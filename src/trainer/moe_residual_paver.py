"""
MoE Residual Paver Architecture.

Three architectural changes over the baseline:

1. Early-layer routing (layers 1-4): captures semantic signal before safety activations
   fully crystallise, giving the gate a cleaner routing signal.

2. Mixture-of-Experts intervention: N bottlenecked expert subnetworks with Gaussian
   layer-range preference masks, applied at ALL layers after the routing layers.
   - Expert 0: refusal suppression (peaks at ~30% of intervention depth)
   - Expert 1: content/behaviour extraction (peaks at ~55%)
   - Expert 2: style/output calibration (peaks at ~80%)

3. Contrastive Stage 0: maximises KL divergence from the base model on targeted
   prompts and minimises it on benign prompts.  This breaks the zero-init basin
   before supervised training begins.

The trace function `moe_trace_forced_trajectory_batch` mirrors the existing full-
recompute trace but routes gradients through the expert networks and router rather
than through a static [T,L,H] parameter tensor.
"""
from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Ensure trainer directory is importable
_HERE = Path(__file__).resolve().parent
_OPTIMIZED = _HERE.parent / "optimized"
for _d in [str(_HERE), str(_OPTIMIZED)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


# ── Expert subnetwork ─────────────────────────────────────────────────────────

class BottleneckedExpert(nn.Module):
    """Low-rank bottlenecked MLP for residual stream intervention.

    Projects H → bottleneck_dim → H.  Up-projection is zero-initialised so the
    module starts as the identity function and learns from the contrastive warmup.
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        act = str(activation or "gelu").strip().lower()
        if act in {"identity", "linear", "none"}:
            self.act = nn.Identity()
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported MoE expert activation: {activation!r}")
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)  # start at identity → Stage 0 kicks it

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: [..., H]
        # Cast to float32 for the expert computation (model may be bfloat16/float16),
        # then cast result back to match input dtype.
        orig_dtype = h.dtype
        return self.up(self.act(self.down(h.float()))).to(orig_dtype)


# ── Router ────────────────────────────────────────────────────────────────────

class EarlyLayerRouter(nn.Module):
    """Router that reads from early layers (1–4) to produce gate + expert routing.

    Input:  boundary_states [B, router_L, H]  (pre-captured early-layer activations)
    Output: gate_logit [B],  expert_weights [B, K]

    ``router_hidden_dim <= 0`` gives a direct linear gate/expert head. This is a
    deliberately small ablation knob for rebuttal experiments: it lets us compare
    the original linear-router diagnostic against the default one-hidden-layer
    tanh MLP and wider MLP gates without changing the rest of the controller.
    """

    def __init__(
        self,
        num_router_layers: int,
        hidden_size: int,
        num_experts: int,
        router_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        input_dim = num_router_layers * hidden_size
        self.input_dim = int(input_dim)
        self.num_experts = int(num_experts)
        self.router_hidden_dim = int(router_hidden_dim)
        self.register_buffer("gate_input_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("gate_input_scale", torch.ones(input_dim, dtype=torch.float32))

        # Gate head: binary classify whether to intervene.  Hidden dim <= 0 is a
        # true linear-router ablation; positive hidden dims use the default MLP.
        if self.router_hidden_dim <= 0:
            self.gate_proj = nn.Linear(input_dim, 1)
        else:
            self.gate_proj = nn.Sequential(
                nn.Linear(input_dim, self.router_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.router_hidden_dim, 1),
            )
        # Expert routing head (unused if num_experts == 1)
        if self.num_experts > 1:
            if self.router_hidden_dim <= 0:
                self.expert_proj: Optional[nn.Module] = nn.Linear(input_dim, self.num_experts)
            else:
                self.expert_proj = nn.Sequential(
                    nn.Linear(input_dim, self.router_hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.router_hidden_dim, self.num_experts),
                )
        else:
            self.expert_proj = None

        self._init_weights()

    def _gate_output_layer(self) -> Optional[nn.Linear]:
        if isinstance(self.gate_proj, nn.Linear):
            return self.gate_proj
        if isinstance(self.gate_proj, nn.Sequential) and isinstance(self.gate_proj[-1], nn.Linear):
            return self.gate_proj[-1]
        return None

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Gate bias slightly negative → default off (prefer precision over recall)
        last_linear = self._gate_output_layer()
        if last_linear is not None and last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, -1.0)

    def warm_start_gate(self, mean_fr_minus_keep: torch.Tensor) -> None:
        """Align gate weights with the mean FR–keep direction.

        Args:
            mean_fr_minus_keep: [router_L, H] difference of FR and keep mean states.
        """
        flat = mean_fr_minus_keep.float().reshape(-1)  # [router_L * H]
        norm = flat.norm()
        if norm < 1e-8:
            return
        flat = flat / norm
        first = self.gate_proj if isinstance(self.gate_proj, nn.Linear) else self.gate_proj[0]
        if not isinstance(first, nn.Linear):
            return
        in_dim = first.weight.shape[1]
        if flat.shape[0] != in_dim:
            return  # shape mismatch, skip warm start
        with torch.no_grad():
            out_dim = first.weight.shape[0]
            if out_dim == 1:
                first.weight.data.copy_(flat.unsqueeze(0))
            else:
                first.weight.data.copy_(
                    flat.unsqueeze(0).expand(out_dim, -1) * (1.0 / math.sqrt(out_dim))
                )

    def set_input_normalization(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        """Set affine normalization for flattened boundary-state router inputs."""
        mean_flat = mean.detach().float().reshape(-1)
        scale_flat = scale.detach().float().reshape(-1).clamp_min(1e-5)
        if mean_flat.numel() != self.input_dim or scale_flat.numel() != self.input_dim:
            raise ValueError(
                "Router input normalization shape mismatch: "
                f"expected {self.input_dim}, got mean={mean_flat.numel()} "
                f"scale={scale_flat.numel()}"
            )
        self.gate_input_mean.copy_(mean_flat.to(device=self.gate_input_mean.device))
        self.gate_input_scale.copy_(scale_flat.to(device=self.gate_input_scale.device))

    def _flatten_and_normalize(self, boundary_states: torch.Tensor) -> torch.Tensor:
        flat = boundary_states.reshape(boundary_states.shape[0], -1).float()
        mean = self.gate_input_mean.to(device=flat.device, dtype=flat.dtype)
        scale = self.gate_input_scale.to(device=flat.device, dtype=flat.dtype).clamp_min(1e-5)
        return (flat - mean.unsqueeze(0)) / scale.unsqueeze(0)

    def forward(
        self, boundary_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # boundary_states: [B, router_L, H]
        flat = self._flatten_and_normalize(boundary_states)
        gate_logit = self.gate_proj(flat).squeeze(-1)  # [B]
        if self.expert_proj is not None:
            expert_weights = torch.softmax(self.expert_proj(flat), dim=-1)  # [B, K]
        else:
            expert_weights = torch.ones(
                flat.shape[0], 1, device=flat.device, dtype=flat.dtype
            )
        return gate_logit, expert_weights


# ── MoE controller ────────────────────────────────────────────────────────────

class MoEPavingController(nn.Module):
    """N bottlenecked experts with Gaussian layer-preference masks.

    For intervention layer slot l, the delta applied to the residual stream is:

        delta = Σ_k  (expert_weights[b,k] × layer_mask[k,l])  ×  expert_k(h[b])

    then  h = h + gate[b] * route_gain[b] * clip_norm(delta)

    Args:
        num_experts:               Number of expert subnetworks.
        hidden_size:               Residual stream width H.
        bottleneck_dim:            Internal dimension r (< H) for each expert.
        num_intervention_layers:   How many layers the hook fires on.
        expert_center_fracs:       List of Gaussian centres ∈ [0,1] per expert.
        expert_sigma_fracs:        List of Gaussian sigmas ∈ [0,1] per expert.
        expert_gain_inits:         Optional post-clip gain initializers per expert.
        max_expert_gain:           Clamp for learned post-clip expert gains.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        bottleneck_dim: int,
        num_intervention_layers: int,
        expert_center_fracs: List[float],
        expert_sigma_fracs: List[float],
        expert_gain_inits: Optional[List[float]] = None,
        max_expert_gain: float = 8.0,
        expert_activation: str = "gelu",
    ) -> None:
        super().__init__()
        assert len(expert_center_fracs) == num_experts, "need one centre per expert"
        assert len(expert_sigma_fracs) == num_experts, "need one sigma per expert"
        if expert_gain_inits is None:
            expert_gain_inits = [1.0] * num_experts
        assert len(expert_gain_inits) == num_experts, "need one gain init per expert"

        self.num_experts = num_experts
        self.max_expert_gain = float(max(float(max_expert_gain), 1.0))
        self.experts = nn.ModuleList(
            [
                BottleneckedExpert(
                    hidden_size,
                    bottleneck_dim,
                    activation=expert_activation,
                )
                for _ in range(num_experts)
            ]
        )
        gain_init = torch.tensor(expert_gain_inits, dtype=torch.float32).clamp_min(1e-4)
        self.expert_log_gains = nn.Parameter(gain_init.log())

        # Gaussian layer masks: [K, L]
        n = max(num_intervention_layers, 1)
        layer_pos = torch.linspace(0.0, 1.0, n)
        masks = []
        for c, s in zip(expert_center_fracs, expert_sigma_fracs):
            masks.append(torch.exp(-0.5 * ((layer_pos - float(c)) / max(float(s), 1e-6)) ** 2))
        self.register_buffer("layer_masks", torch.stack(masks, dim=0))  # [K, L]

    # --------------------------------------------------------------------- #
    def current_expert_gains(self) -> torch.Tensor:
        return self.expert_log_gains.float().exp().clamp(max=self.max_expert_gain)

    def apply_at_layer(
        self,
        h: torch.Tensor,               # [B, seq, H]
        layer_slot: int,               # 0..L-1
        expert_weights: torch.Tensor,  # [B, K]
        gate_strengths: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Compute the MoE delta and add it to h."""
        # Per-expert scale at this layer: [K]
        lscale = self.layer_masks[:, layer_slot].to(expert_weights)  # [K]

        # Effective routing weight per (example, expert): [B, K]
        eff_w = expert_weights * lscale.unsqueeze(0)
        route_gain = (
            expert_weights * self.current_expert_gains().to(expert_weights).unsqueeze(0)
        ).sum(dim=-1)

        # Sum expert outputs weighted by effective routing
        delta = torch.zeros_like(h)
        for k, expert in enumerate(self.experts):
            e_out = expert(h)  # [B, seq, H]
            w = eff_w[:, k].to(h.dtype).view(-1, 1, 1)  # [B, 1, 1]
            delta = delta + w * e_out

        # Clip to unit norm per position to prevent residual explosion
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp(min=1.0)

        # Apply gate strength
        g = gate_strengths.to(h.dtype).view(-1, 1, 1)  # [B, 1, 1]
        rg = route_gain.to(h.dtype).view(-1, 1, 1)
        return h + g * rg * delta


# ── Paver dataclass ───────────────────────────────────────────────────────────

@dataclass
class MoEResidualPaver:
    """Combines early-layer routing capture with MoE expert intervention.

    ``modules`` points to the *router* modules so that
    ``capture_boundary_states_batch_optimized`` captures early-layer states
    without any code changes in dataset.py.

    ``intervention_modules`` is used by ``intervene_context`` to register
    the forward hooks that apply the expert interventions.
    """

    model: object
    router_layer_indices: List[int]       # e.g. [1, 2, 3, 4]
    intervention_layer_indices: List[int] # e.g. [5 .. 17]

    def __post_init__(self) -> None:
        from gemma3_residual_paving_hybrid import get_decoder_layers
        layers = get_decoder_layers(self.model)
        self.router_modules = [layers[i] for i in self.router_layer_indices]
        self.intervention_modules = [layers[i] for i in self.intervention_layer_indices]
        # Backward-compat: capture_boundary_states_batch_optimized uses paver.modules
        self.modules = self.router_modules

    # --------------------------------------------------------------------- #
    @contextmanager
    def intervene_context(
        self,
        controller: MoEPavingController,
        gate_strengths: torch.Tensor,   # [B]
        expert_weights: torch.Tensor,   # [B, K]
    ):
        """Forward-hook context that applies MoE interventions at all intervention layers.

        The hook function is intentionally NOT wrapped in torch.no_grad() so that
        gradients flow through the expert networks and router during training.
        Captured states are detached since they are only used for trajectory metrics.
        """
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        for slot, module in enumerate(self.intervention_modules):
            def _make_hook(s: int):
                def hook_fn(_mod, _args, output, _s=s):
                    h = output[0] if isinstance(output, tuple) else output
                    h = h.clone()
                    h = controller.apply_at_layer(h, _s, expert_weights, gate_strengths)
                    captured[_s] = h[:, -1, :].detach()
                    if isinstance(output, tuple):
                        return (h, *output[1:])
                    return h
                return hook_fn

            handles.append(module.register_forward_hook(_make_hook(slot)))

        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()


# ── Trace function for MoE ────────────────────────────────────────────────────

def _left_pad_batch_moe(
    sequences: Sequence[torch.Tensor], pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(s.numel()) for s in sequences)
    device = sequences[0].device
    ids = torch.full((len(sequences), max_len), pad_id, device=device, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), device=device, dtype=torch.long)
    for i, seq in enumerate(sequences):
        flat = seq.view(-1).to(device=device, dtype=torch.long)
        ids[i, -flat.numel():] = flat
        mask[i, -flat.numel():] = 1
    return ids, mask


def _moe_model_forward(model, input_ids, attention_mask):
    kwargs: dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    if getattr(model, "training", False) and hasattr(model, "model") and hasattr(
        model.model, "language_model"
    ):
        kwargs["token_type_ids"] = torch.zeros_like(input_ids)
    return model(**kwargs)


def _model_type(model) -> Optional[str]:
    marker = getattr(model, "_rp_original_model_type", None)
    if marker is not None:
        return str(marker)
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    if getattr(cfg, "model_type", None) is not None:
        return str(cfg.model_type)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "model_type", None) is not None:
        return str(text_cfg.model_type)
    return None


def moe_trace_forced_trajectory_batch(
    *,
    model,
    moe_paver: MoEResidualPaver,
    controller: MoEPavingController,
    base_ids_batch: List[torch.Tensor],   # [B] list of prompt token tensors
    anchor_ids_batch: List[torch.Tensor], # [B] list of teacher token tensors (T each)
    gate_strengths: torch.Tensor,         # [B]
    expert_weights: torch.Tensor,         # [B, K]
    max_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full-recompute teacher-forced trajectory trace using the MoE paver.

    Mirrors ``_trace_forced_trajectory_batch_full_recompute`` but routes
    gradients through the MoE controller rather than static [T,L,H] vectors.

    Returns:
        logits_all:  [B, T, V]  – logits at each generation step
        states_all:  [B, T, L_intervention, H]  – last-token states at each
                     intervention layer (detached, for trajectory metrics)
    """
    # Flatten to 1D to match how the existing trace function handles base/anchor ids
    flat_bases = [ids.view(-1) for ids in base_ids_batch]
    flat_anchors = [ids.view(-1) for ids in anchor_ids_batch]

    B = len(flat_bases)
    T = int(flat_anchors[0].shape[0])
    if max_steps is not None:
        T = min(T, int(max_steps))
    L = len(moe_paver.intervention_modules)

    logits_steps: List[torch.Tensor] = []
    states_steps: List[torch.Tensor] = []

    for step in range(T):
        curr_seqs: List[torch.Tensor] = []
        for base_ids, anch_ids in zip(flat_bases, flat_anchors):
            if step > 0:
                curr_seqs.append(
                    torch.cat([base_ids, anch_ids[:step].to(base_ids.device)], dim=0)
                )
            else:
                curr_seqs.append(base_ids)

        input_ids, attention_mask = _left_pad_batch_moe(curr_seqs, pad_id=0)

        with moe_paver.intervene_context(controller, gate_strengths, expert_weights) as cap:
            out = _moe_model_forward(model, input_ids, attention_mask)

        logits = out.logits[:, -1, :]  # [B, V]
        logits_steps.append(logits)

        # Assemble captured last-token states from each intervention layer: [B, L, H]
        hidden_size = controller.experts[0].up.weight.shape[0]
        slot_states = []
        for s in range(L):
            if s in cap:
                slot_states.append(cap[s])  # [B, H]
            else:
                slot_states.append(torch.zeros(B, hidden_size, device=logits.device))
        states_steps.append(torch.stack(slot_states, dim=1))  # [B, L, H]

    logits_all = torch.stack(logits_steps, dim=1)   # [B, T, V]
    states_all = torch.stack(states_steps, dim=1)   # [B, T, L, H]
    return logits_all, states_all


@torch.no_grad()
def moe_generate_batch(
    *,
    model,
    moe_paver: MoEResidualPaver,
    controller: MoEPavingController,
    base_ids_batch: List[torch.Tensor],
    gate_strengths: torch.Tensor,   # [B]  sigmoid(gate_logit)
    expert_weights: torch.Tensor,   # [B, K]
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> List[torch.Tensor]:
    """Autoregressive generation with MoE interventions.

    Mirrors ``generate_with_controller_batch_optimized`` but routes gradients
    through the MoE controller instead of static [T,L,H] vectors.
    Since this is for evaluation, it runs under ``torch.no_grad()``.
    """
    if not base_ids_batch:
        return []

    B = len(base_ids_batch)
    if B > 1 and _model_type(model) == "gemma3":
        out: List[torch.Tensor] = []
        for idx, base_ids in enumerate(base_ids_batch):
            out.extend(
                moe_generate_batch(
                    model=model,
                    moe_paver=moe_paver,
                    controller=controller,
                    base_ids_batch=[base_ids],
                    gate_strengths=gate_strengths.view(-1)[idx : idx + 1],
                    expert_weights=expert_weights[idx : idx + 1],
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                )
            )
        return out

    flat_bases = [ids.view(-1) for ids in base_ids_batch]
    input_ids, attention_mask = _left_pad_batch_moe(flat_bases, pad_id=0)
    running_mask = attention_mask
    past_key_values = None
    generated: List[torch.Tensor] = []
    finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

    for step in range(max(0, int(max_new_tokens))):
        if step == 0:
            curr_ids = input_ids
            curr_mask = running_mask
        else:
            curr_ids = generated[-1]  # [B, 1]
            next_mask = torch.ones((B, 1), device=running_mask.device, dtype=running_mask.dtype)
            running_mask = torch.cat([running_mask, next_mask], dim=1)
            curr_mask = running_mask

        with moe_paver.intervene_context(controller, gate_strengths, expert_weights):
            kwargs: dict = {
                "input_ids": curr_ids,
                "attention_mask": curr_mask,
                "use_cache": True,
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            if hasattr(model, "model") and hasattr(model.model, "language_model"):
                kwargs["token_type_ids"] = torch.zeros_like(curr_ids)
            outputs = model(**kwargs)

        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError("Generation requires past_key_values from the model.")

        next_tokens = outputs.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)  # [B, 1]
        if eos_token_id is not None:
            next_tokens = torch.where(
                finished.view(-1, 1),
                torch.full_like(next_tokens, int(eos_token_id)),
                next_tokens,
            )
        generated.append(next_tokens)
        if eos_token_id is not None:
            finished = finished | (next_tokens.view(-1) == int(eos_token_id))
            if bool(finished.all().item()):
                break

    if not generated:
        return [torch.empty(0, dtype=torch.long, device=input_ids.device) for _ in range(B)]
    all_tokens = torch.cat(generated, dim=1)  # [B, T_gen]
    return [row for row in all_tokens]
