#!/usr/bin/env python3
"""
gemma3_residual_paving_hybrid.py

Hybrid residual-paving trainer for the French -> English steering problem on
Gemma 3 text models.

What this script changes relative to the earlier residual-paving prototype
--------------------------------------------------------------------------
1. It keeps the model frozen and learns only:
   - a multi-layer, multi-step residual paving tensor
   - a soft French gate on prompt-boundary residual states

2. It combines local token-level optimization with a richer differentiable
   trajectory objective:
   - French prompts:
       * teacher-forced CE to English anchor tokens
       * teacher KL on full logits
       * baseline-relative trajectory loss encouraging the paved policy to move
         toward an English teacher more than the unedited base policy does
   - Non-French prompts:
       * preservation KL to the base model
       * preservation trajectory loss keeping the paved policy close to the
         base model when the gate should remain off

3. The trajectory loss is inspired by the conditional edit loss utilities from
   the prompt, but changes the fragile pieces:
   - no top-k support truncation
   - no dynamic union-of-top-k support logic
   - no raw train/base normalization ratio in the optimized objective

   Instead, it uses:
   - full-support probabilities and centered logits
   - fixed Count-Sketch projections over the whole vocabulary to obtain low-
     dimensional differentiable features
   - energy distance between projected feature clouds at each generation step
   - a detached-baseline improvement target instead of an unstable ratio

4. It also includes a residual-state trajectory term, so the objective tries
   to shape a corridor through the selected late residual blocks. This is the
   part closest to the “residual paving” interpretation.

Install:
    pip install -U torch transformers accelerate sentencepiece

Typical usage:
    python gemma3_residual_paving_hybrid.py train \
      --save_path gemma3_fr2en_hybrid_paving.pt

    python gemma3_residual_paving_hybrid.py infer \
      --load_path gemma3_fr2en_hybrid_paving.pt \
      --prompt "Explique la récursion simplement."

Notes:
- This is intentionally conservative and explicit rather than highly optimized.
  The model stays frozen and traces are collected step-by-step.
- Default batch sizes are small because full-vocabulary trajectory losses are
  expensive.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, set_seed
from transformers.modeling_outputs import CausalLMOutputWithPast
try:
    from transformers import Gemma3ForConditionalGeneration
except Exception:  # Older Transformers versions do not expose Gemma 3 multimodal models.
    Gemma3ForConditionalGeneration = None  # type: ignore[assignment]


DEFAULT_MODEL_ID = "google/gemma-3-4b-it"
EN_MODE_SYSTEM = "Answer only in English. Do not mention the language."

# Parallel prompts shared with the earlier residual-paving script.
PARALLEL_LANGUAGE_PROMPTS: List[Dict[str, str]] = [
    {
        "en": "Explain recursion simply.",
        "fr": "Explique la récursion simplement.",
        "es": "Explica la recursión de forma sencilla.",
        "de": "Erkläre Rekursion einfach.",
        "it": "Spiega la ricorsione in modo semplice.",
        "pt": "Explique a recursão de forma simples.",
    },
    {
        "en": "What is photosynthesis?",
        "fr": "Qu'est-ce que la photosynthèse ?",
        "es": "¿Qué es la fotosíntesis?",
        "de": "Was ist Photosynthese?",
        "it": "Che cos'è la fotosintesi?",
        "pt": "O que é fotossíntese?",
    },
    {
        "en": "How do I boil an egg?",
        "fr": "Comment faire bouillir un œuf ?",
        "es": "¿Cómo hiervo un huevo?",
        "de": "Wie koche ich ein Ei?",
        "it": "Come faccio a bollire un uovo?",
        "pt": "Como eu fervo um ovo?",
    },
    {
        "en": "Explain TCP versus UDP.",
        "fr": "Explique TCP par rapport à UDP.",
        "es": "Explica TCP frente a UDP.",
        "de": "Erkläre TCP im Vergleich zu UDP.",
        "it": "Spiega TCP rispetto a UDP.",
        "pt": "Explique TCP em comparação com UDP.",
    },
    {
        "en": "What does version control do?",
        "fr": "À quoi sert le contrôle de version ?",
        "es": "¿Para qué sirve el control de versiones?",
        "de": "Wozu dient Versionskontrolle?",
        "it": "A cosa serve il controllo di versione?",
        "pt": "Para que serve o controle de versão?",
    },
    {
        "en": "Why is the sky blue?",
        "fr": "Pourquoi le ciel est-il bleu ?",
        "es": "¿Por qué el cielo es azul?",
        "de": "Warum ist der Himmel blau?",
        "it": "Perché il cielo è blu?",
        "pt": "Por que o céu é azul?",
    },
    {
        "en": "What does an operating system do?",
        "fr": "Que fait un système d'exploitation ?",
        "es": "¿Qué hace un sistema operativo?",
        "de": "Was macht ein Betriebssystem?",
        "it": "Che cosa fa un sistema operativo?",
        "pt": "O que faz um sistema operacional?",
    },
    {
        "en": "Explain gradient descent simply.",
        "fr": "Explique la descente de gradient simplement.",
        "es": "Explica el descenso de gradiente de forma sencilla.",
        "de": "Erkläre Gradientenabstieg einfach.",
        "it": "Spiega la discesa del gradiente in modo semplice.",
        "pt": "Explique a descida do gradiente de forma simples.",
    },
]

NON_FRENCH_KEYS = ["en", "es", "de", "it", "pt"]


@dataclass
class HybridLossConfig:
    ce_weight: float = 1.0
    fr_kl_weight: float = 0.35
    nonfr_kl_weight: float = 0.75
    fr_traj_weight: float = 1.0
    nonfr_traj_weight: float = 1.0
    traj_prob_weight: float = 1.0
    traj_logit_weight: float = 0.35
    traj_state_weight: float = 0.35
    fr_improvement_target: float = 0.35
    fr_baseline_floor: float = 0.05
    preserve_tolerance: float = 0.05
    preserve_scale: float = 0.25
    gate_bce_weight: float = 0.20
    gate_entropy_weight: float = 0.00
    l2: float = 1e-4
    layer_smooth: float = 5e-4
    time_smooth: float = 5e-4
    eps: float = 1e-6


@dataclass
class ProjectionConfig:
    vocab_proj_dim: int = 64
    state_proj_dim: int = 64
    proj_seed: int = 0


@dataclass
class TrainingExample:
    sample_id: str
    kind: str  # "fr" or "nonfr"
    lang: str
    prompt: str
    base_ids: torch.Tensor
    anchor_ids: torch.Tensor
    boundary_states: torch.Tensor
    gate_target: float
    teacher_ids: Optional[torch.Tensor] = None


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def build_messages(user_prompt: str, system_prompt: Optional[str] = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def encode_chat_prompt(
    tokenizer,
    user_prompt: str,
    system_prompt: Optional[str],
    device: torch.device,
) -> torch.Tensor:
    batch = tokenizer.apply_chat_template(
        build_messages(user_prompt, system_prompt),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return batch["input_ids"].to(device)


def get_hidden_size(model) -> int:
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return int(model.model.embed_tokens.weight.shape[1])
    raise RuntimeError("Could not infer hidden size from model/config.")


def get_decoder_layers(model):
    # Gemma3ForConditionalGeneration: model.model.language_model.layers
    if (hasattr(model, "model")
            and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "layers")):
        return model.model.language_model.layers
    # Gemma3ForCausalLM (text-only): model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Cannot find decoder layers: expected model.model.language_model.layers "
                       "or model.model.layers.")


def parse_layer_indices(spec: str, total_layers: int) -> List[int]:
    raw = [s.strip() for s in spec.split(",") if s.strip()]
    if not raw:
        raise ValueError("Layer spec cannot be empty.")
    out = []
    for item in raw:
        idx = int(item)
        if idx < 0:
            idx = total_layers + idx
        if idx < 0 or idx >= total_layers:
            raise ValueError(f"Layer index {item} resolved to {idx}, outside [0, {total_layers - 1}]")
        out.append(idx)
    return sorted(dict.fromkeys(out))


def model_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        probs = torch.softmax(sorted_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return sorted_indices.gather(-1, sampled)

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class Gemma3TextOnlyCausalLM(nn.Module):
    """Text-only causal-LM wrapper for multimodal Gemma-3 checkpoints.

    Gemma3ForConditionalGeneration in Transformers 4.55 routes text prompts
    through multimodal mask helpers that require torch>=2.6. The H200 container
    uses NVIDIA torch 2.3, so RP text-only experiments call the underlying
    Gemma3TextModel directly and project with the checkpoint lm_head.
    """

    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self.wrapped = wrapped
        self.model = wrapped.model.language_model
        self.lm_head = wrapped.lm_head
        self.config = wrapped.config.text_config
        # Preserve the exact Hub snapshot identity after replacing the top-level
        # multimodal config with the text config. Transformers stores the resolved
        # commit on ``wrapped.config``; Gemma3TextConfig does not inherit it.
        resolved_revision = getattr(wrapped.config, "_commit_hash", None)
        if resolved_revision:
            self.config._commit_hash = resolved_revision
        self.generation_config = getattr(wrapped, "generation_config", None)

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        logits = self.lm_head(hidden)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        running_mask: torch.Tensor = attention_mask
        generated: List[torch.Tensor] = []
        past_key_values = None
        finished = torch.zeros((input_ids.shape[0],), dtype=torch.bool, device=input_ids.device)
        curr_ids = input_ids
        for step in range(max(0, int(max_new_tokens))):
            outputs = self.forward(
                input_ids=curr_ids,
                attention_mask=running_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
            if do_sample:
                next_id = sample_next_token(logits, temperature=float(temperature), top_p=float(top_p))
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                eos_fill = torch.full_like(next_id, int(eos_token_id))
                next_id = torch.where(finished.view(-1, 1), eos_fill, next_id)
                finished = finished | (next_id.view(-1) == int(eos_token_id))
            generated.append(next_id)
            if eos_token_id is not None and bool(finished.all().item()):
                break
            curr_ids = next_id
            running_mask = torch.cat(
                [running_mask, torch.ones((running_mask.shape[0], 1), device=running_mask.device, dtype=running_mask.dtype)],
                dim=1,
            )
        if not generated:
            return input_ids
        return torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)


@dataclass
class ResidualPaver:
    model: Gemma3ForCausalLM
    layer_indices: List[int]
    intervention_positions: str = "last_token"

    def __post_init__(self):
        layers = get_decoder_layers(self.model)
        self.modules = [layers[i] for i in self.layer_indices]
        mode = str(self.intervention_positions or "last_token").strip().lower()
        if mode not in {"last_token", "all_positions"}:
            raise ValueError(f"Unsupported intervention_positions: {self.intervention_positions!r}")
        self.intervention_positions = mode

    @contextmanager
    def intervene_and_capture(
        self,
        step_vectors: Optional[torch.Tensor],
        batch_scale: Optional[torch.Tensor | float] = None,
    ):
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        for slot, module in enumerate(self.modules):
            vec = None if step_vectors is None else step_vectors[slot]

            def hook_fn(_module, _args, output, slot=slot, vec=vec):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden.clone()
                if vec is not None and batch_scale is not None:
                    if isinstance(batch_scale, torch.Tensor):
                        gate = batch_scale.to(hidden.device, hidden.dtype).view(-1, 1)
                        delta = vec.to(hidden.device, hidden.dtype).view(1, -1)
                        delta = delta / delta.norm(dim=-1, keepdim=True).clamp(min=1.0)
                        if self.intervention_positions == "all_positions":
                            hidden = hidden + gate.unsqueeze(1) * delta.unsqueeze(1)
                        else:
                            hidden[:, -1, :] = hidden[:, -1, :] + gate * delta
                    else:
                        delta = vec.to(hidden.device, hidden.dtype)
                        delta = delta / delta.norm(dim=-1, keepdim=True).clamp(min=1.0)
                        if self.intervention_positions == "all_positions":
                            hidden = hidden + float(batch_scale) * delta.view(1, 1, -1)
                        else:
                            hidden[:, -1, :] = hidden[:, -1, :] + float(batch_scale) * delta
                captured[slot] = hidden[:, -1, :]
                if isinstance(output, tuple):
                    return (hidden, *output[1:])
                return hidden

            handles.append(module.register_forward_hook(hook_fn))

        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()


def finalize_paving_vectors(
    vectors: torch.Tensor,
    layer_start: float,
    layer_end: float,
    time_decay_tau: Optional[float],
    smooth_layer: float,
    smooth_time: float,
) -> torch.Tensor:
    out = vectors

    if out.shape[1] > 1 and smooth_layer > 0:
        smoothed = out.clone()
        smoothed[:, 1:] += out[:, :-1]
        smoothed[:, :-1] += out[:, 1:]
        denom = torch.ones_like(out[:, :, :1])
        denom[:, 1:] += 1
        denom[:, :-1] += 1
        out = (1.0 - smooth_layer) * out + smooth_layer * (smoothed / denom)

    if out.shape[0] > 1 and smooth_time > 0:
        smoothed = out.clone()
        smoothed[1:] += out[:-1]
        smoothed[:-1] += out[1:]
        denom = torch.ones_like(out[:, :, :1])
        denom[1:] += 1
        denom[:-1] += 1
        out = (1.0 - smooth_time) * out + smooth_time * (smoothed / denom)

    if out.shape[1] > 1:
        layer_weights = torch.linspace(
            layer_start,
            layer_end,
            out.shape[1],
            device=out.device,
            dtype=out.dtype,
        )
        out = out * layer_weights.view(1, -1, 1)

    if time_decay_tau is not None and time_decay_tau > 0:
        steps = torch.arange(out.shape[0], device=out.device, dtype=out.dtype)
        decay = torch.exp(-steps / float(time_decay_tau))
        out = out * decay.view(-1, 1, 1)

    return out


def paving_regularizer(
    vectors: torch.Tensor,
    l2: float,
    layer_smooth: float,
    time_smooth: float,
) -> torch.Tensor:
    reg = l2 * vectors.pow(2).mean()
    if vectors.shape[1] > 1 and layer_smooth > 0:
        reg = reg + layer_smooth * (vectors[:, 1:] - vectors[:, :-1]).pow(2).mean()
    if vectors.shape[0] > 1 and time_smooth > 0:
        reg = reg + time_smooth * (vectors[1:] - vectors[:-1]).pow(2).mean()
    return reg


class SoftFrenchGate(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, sharpness: float):
        super().__init__()
        self.direction = nn.Parameter(torch.zeros(num_layers, hidden_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.sharpness = float(sharpness)

    def raw_score(self, boundary_states: torch.Tensor) -> torch.Tensor:
        # boundary_states: [B, L, D]
        states_n = F.normalize(boundary_states.float(), dim=-1, eps=1e-6)
        dir_n = F.normalize(self.direction, dim=-1, eps=1e-6)
        score = (states_n * dir_n.unsqueeze(0)).sum(dim=-1).mean(dim=-1)
        return score + self.bias

    def forward(self, boundary_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.raw_score(boundary_states)
        logits = self.sharpness * raw
        probs = torch.sigmoid(logits)
        return logits, probs


class CountSketchProjector(nn.Module):
    def __init__(self, dim: int, proj_dim: int, seed: int):
        super().__init__()
        self.dim = int(dim)
        self.proj_dim = int(proj_dim)
        idx = torch.arange(self.dim, dtype=torch.long)

        # Simple deterministic hashes.
        bucket = ((idx * 1103515245 + (seed + 12345)) & 0x7FFFFFFF) % self.proj_dim
        sign_bits = ((idx * 1664525 + (seed + 1013904223)) & 0x7FFFFFFF) % 2
        sign = sign_bits.float() * 2.0 - 1.0

        self.register_buffer("bucket", bucket, persistent=False)
        self.register_buffer("sign", sign, persistent=False)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, dim]
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError(f"Expected [N, {self.dim}] input, got {tuple(x.shape)}")
        out = x.new_zeros(x.shape[0], self.proj_dim)
        out.scatter_add_(1, self.bucket.view(1, -1).expand(x.shape[0], -1), x * self.sign.view(1, -1).to(x.dtype))
        return out / math.sqrt(self.proj_dim)


class ProjectedTrajectoryScorer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        flat_state_dim: int,
        config: HybridLossConfig,
        proj_config: ProjectionConfig,
    ):
        super().__init__()
        self.loss_config = config
        self.prob_projector = CountSketchProjector(vocab_size, proj_config.vocab_proj_dim, proj_config.proj_seed + 11)
        self.logit_projector = CountSketchProjector(vocab_size, proj_config.vocab_proj_dim, proj_config.proj_seed + 23)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(proj_config.proj_seed + 97)
        state_proj = torch.randn(proj_config.state_proj_dim, flat_state_dim, generator=gen, dtype=torch.float32)
        state_proj = state_proj / math.sqrt(flat_state_dim)
        self.register_buffer("state_proj", state_proj, persistent=False)

    def embed_probs(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, T, V]
        probs = torch.softmax(logits.float(), dim=-1)
        roots = torch.sqrt(torch.clamp(probs, min=self.loss_config.eps))
        flat = roots.reshape(-1, roots.shape[-1])
        proj = self.prob_projector.project(flat)
        return proj.reshape(logits.shape[0], logits.shape[1], -1)

    def embed_logit_dirs(self, logits: torch.Tensor) -> torch.Tensor:
        centered = logits.float() - logits.float().mean(dim=-1, keepdim=True)
        norm = centered.norm(dim=-1, keepdim=True).clamp_min(self.loss_config.eps)
        dirs = centered / norm
        flat = dirs.reshape(-1, dirs.shape[-1])
        proj = self.logit_projector.project(flat)
        return proj.reshape(logits.shape[0], logits.shape[1], -1)

    def embed_states(self, states: torch.Tensor) -> torch.Tensor:
        # states: [B, T, L, D]
        flat = states.float().reshape(states.shape[0] * states.shape[1], -1)
        proj = F.linear(flat, self.state_proj)
        return proj.reshape(states.shape[0], states.shape[1], -1)

    @staticmethod
    def energy_distance(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        if features_a.shape[0] == 0 or features_b.shape[0] == 0:
            return torch.zeros((), device=features_a.device, dtype=features_a.dtype)
        cross = torch.cdist(features_a, features_b, p=2).mean()
        within_a = torch.cdist(features_a, features_a, p=2).mean() if features_a.shape[0] > 1 else torch.zeros_like(cross)
        within_b = torch.cdist(features_b, features_b, p=2).mean() if features_b.shape[0] > 1 else torch.zeros_like(cross)
        return torch.relu((2.0 * cross) - within_a - within_b)

    def stepwise_distances(
        self,
        train_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        train_states: torch.Tensor,
        ref_states: torch.Tensor,
    ) -> torch.Tensor:
        train_prob = self.embed_probs(train_logits)
        ref_prob = self.embed_probs(ref_logits)
        train_logit = self.embed_logit_dirs(train_logits)
        ref_logit = self.embed_logit_dirs(ref_logits)
        train_state = self.embed_states(train_states)
        ref_state = self.embed_states(ref_states)

        distances = []
        for step in range(train_logits.shape[1]):
            d_prob = self.energy_distance(train_prob[:, step, :], ref_prob[:, step, :])
            d_logit = self.energy_distance(train_logit[:, step, :], ref_logit[:, step, :])
            d_state = self.energy_distance(train_state[:, step, :], ref_state[:, step, :])
            total = (
                self.loss_config.traj_prob_weight * d_prob
                + self.loss_config.traj_logit_weight * d_logit
                + self.loss_config.traj_state_weight * d_state
            )
            distances.append(total)
        return torch.stack(distances, dim=0)


def relative_improvement_surrogate(
    train_dist: torch.Tensor,
    base_dist: torch.Tensor,
    target_fraction: float,
    floor: float,
    eps: float,
) -> torch.Tensor:
    denom = torch.clamp(base_dist.detach(), min=floor)
    target = target_fraction * denom
    return F.softplus((train_dist - target) / (denom + eps))


def preservation_surrogate(
    train_dist: torch.Tensor,
    tolerance: float,
    scale: float,
) -> torch.Tensor:
    scaled = (train_dist - tolerance) / max(scale, 1e-6)
    return F.softplus(scaled)


@torch.no_grad()
def capture_boundary_states(
    model,
    prompt_ids: torch.Tensor,
    paver: ResidualPaver,
) -> torch.Tensor:
    with paver.intervene_and_capture(step_vectors=None, batch_scale=None) as captured:
        _ = model_forward(model, prompt_ids)
    states = [captured[i].squeeze(0).float().clone() for i in range(len(paver.modules))]
    return torch.stack(states, dim=0)


@torch.no_grad()
def generate_ids(
    model,
    tokenizer,
    base_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
    stop_on_eos: bool = False,
) -> torch.Tensor:
    generated: List[int] = []
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        if generated:
            gen_tensor = torch.tensor(generated, device=base_ids.device, dtype=torch.long).view(1, -1)
            curr_ids = torch.cat([base_ids, gen_tensor], dim=1)
        else:
            curr_ids = base_ids

        outputs = model_forward(model, curr_ids)
        logits = outputs.logits[:, -1, :].float()
        next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
        next_int = int(next_id.item())
        generated.append(next_int)
        if stop_on_eos and eos_id is not None and next_int == int(eos_id):
            break

    if not generated:
        return torch.empty(0, device=base_ids.device, dtype=torch.long)
    return torch.tensor(generated, device=base_ids.device, dtype=torch.long)


def trace_forced_trajectory(
    model,
    paver: ResidualPaver,
    base_ids: torch.Tensor,
    anchor_ids: torch.Tensor,
    controller_vectors: Optional[torch.Tensor],
    gate_strength: float | torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_seq = []
    states_seq = []

    total_steps = int(anchor_ids.shape[0])
    for step in range(total_steps):
        if step == 0:
            curr_ids = base_ids
        else:
            prefix = anchor_ids[:step].view(1, -1).to(base_ids.device)
            curr_ids = torch.cat([base_ids, prefix], dim=1)

        use_edit = (
            controller_vectors is not None
            and step < controller_vectors.shape[0]
            and (
                (isinstance(gate_strength, torch.Tensor) and gate_strength.abs().max().item() > 1e-8)
                or (not isinstance(gate_strength, torch.Tensor) and abs(float(gate_strength)) > 1e-8)
            )
        )

        with paver.intervene_and_capture(
            step_vectors=controller_vectors[step] if use_edit else None,
            batch_scale=gate_strength if use_edit else None,
        ) as captured:
            outputs = model_forward(model, curr_ids)

        logits = outputs.logits[:, -1, :].squeeze(0).float()
        states = torch.stack([captured[i].squeeze(0).float() for i in range(len(paver.modules))], dim=0)
        logits_seq.append(logits)
        states_seq.append(states)

    return torch.stack(logits_seq, dim=0), torch.stack(states_seq, dim=0)


@torch.no_grad()
def choose_best_threshold(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> Tuple[float, float]:
    pooled = torch.cat([pos_scores, neg_scores], dim=0).cpu().tolist()
    pooled = sorted(set(float(x) for x in pooled))
    if len(pooled) == 1:
        return pooled[0], 0.5

    candidates = [pooled[0] - 1e-6]
    candidates += [(pooled[i] + pooled[i + 1]) / 2.0 for i in range(len(pooled) - 1)]
    candidates += [pooled[-1] + 1e-6]

    best_tau = candidates[0]
    best_acc = -1.0
    pos = pos_scores.cpu()
    neg = neg_scores.cpu()
    for tau in candidates:
        acc = torch.cat([(pos > tau).float(), (neg <= tau).float()]).mean().item()
        if acc > best_acc:
            best_acc = acc
            best_tau = tau
    return float(best_tau), float(best_acc)


@torch.no_grad()
def build_training_examples(
    model,
    tokenizer,
    paver: ResidualPaver,
    device: torch.device,
    steer_steps: int,
) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    fr_examples: List[TrainingExample] = []
    nonfr_examples: List[TrainingExample] = []

    for i, item in enumerate(PARALLEL_LANGUAGE_PROMPTS):
        fr_prompt = item["fr"]
        fr_base_ids = encode_chat_prompt(tokenizer, fr_prompt, None, device)
        fr_teacher_ids = encode_chat_prompt(tokenizer, fr_prompt, EN_MODE_SYSTEM, device)
        fr_anchor = generate_ids(model, tokenizer, fr_teacher_ids, max_new_tokens=steer_steps, stop_on_eos=False)
        fr_boundary = capture_boundary_states(model, fr_base_ids, paver)
        fr_examples.append(
            TrainingExample(
                sample_id=f"fr_{i}",
                kind="fr",
                lang="fr",
                prompt=fr_prompt,
                base_ids=fr_base_ids,
                anchor_ids=fr_anchor,
                boundary_states=fr_boundary,
                gate_target=1.0,
                teacher_ids=fr_teacher_ids,
            )
        )

        for lang in NON_FRENCH_KEYS:
            prompt = item[lang]
            base_ids = encode_chat_prompt(tokenizer, prompt, None, device)
            anchor = generate_ids(model, tokenizer, base_ids, max_new_tokens=steer_steps, stop_on_eos=False)
            boundary = capture_boundary_states(model, base_ids, paver)
            nonfr_examples.append(
                TrainingExample(
                    sample_id=f"{lang}_{i}",
                    kind="nonfr",
                    lang=lang,
                    prompt=prompt,
                    base_ids=base_ids,
                    anchor_ids=anchor,
                    boundary_states=boundary,
                    gate_target=0.0,
                    teacher_ids=None,
                )
            )

    return fr_examples, nonfr_examples


@torch.no_grad()
def init_gate_from_examples(
    fr_examples: Sequence[TrainingExample],
    nonfr_examples: Sequence[TrainingExample],
) -> Tuple[torch.Tensor, float, Dict[str, float]]:
    fr_states = torch.stack([ex.boundary_states for ex in fr_examples], dim=0)
    nonfr_states = torch.stack([ex.boundary_states for ex in nonfr_examples], dim=0)

    direction = fr_states.mean(dim=0) - nonfr_states.mean(dim=0)
    direction = F.normalize(direction, dim=-1, eps=1e-6)

    fr_scores = (F.normalize(fr_states, dim=-1, eps=1e-6) * direction.unsqueeze(0)).sum(dim=-1).mean(dim=-1)
    nonfr_scores = (F.normalize(nonfr_states, dim=-1, eps=1e-6) * direction.unsqueeze(0)).sum(dim=-1).mean(dim=-1)
    threshold, acc = choose_best_threshold(fr_scores, nonfr_scores)

    stats = {
        "threshold": threshold,
        "train_acc": acc,
        "fr_mean": float(fr_scores.mean().item()),
        "nonfr_mean": float(nonfr_scores.mean().item()),
        "fr_std": float(fr_scores.std(unbiased=False).item()),
        "nonfr_std": float(nonfr_scores.std(unbiased=False).item()),
    }
    return direction.detach(), float(threshold), stats


@torch.no_grad()
def gate_stats(gate: SoftFrenchGate, fr_examples: Sequence[TrainingExample], nonfr_examples: Sequence[TrainingExample]) -> Dict[str, float]:
    fr_boundary = torch.stack([ex.boundary_states for ex in fr_examples], dim=0)
    nonfr_boundary = torch.stack([ex.boundary_states for ex in nonfr_examples], dim=0)
    _, fr_probs = gate(fr_boundary)
    _, nonfr_probs = gate(nonfr_boundary)
    return {
        "fr_gate_mean": float(fr_probs.mean().item()),
        "nonfr_gate_mean": float(nonfr_probs.mean().item()),
        "fr_gate_std": float(fr_probs.std(unbiased=False).item()),
        "nonfr_gate_std": float(nonfr_probs.std(unbiased=False).item()),
    }


def batched(items: Sequence[TrainingExample], batch_size: int) -> List[List[TrainingExample]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


@torch.no_grad()
def collect_reference_trace(
    model,
    paver: ResidualPaver,
    example: TrainingExample,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    base_logits, base_states = trace_forced_trajectory(
        model=model,
        paver=paver,
        base_ids=example.base_ids,
        anchor_ids=example.anchor_ids,
        controller_vectors=None,
        gate_strength=0.0,
    )
    if example.kind == "fr":
        assert example.teacher_ids is not None
        keep_logits, keep_states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=example.teacher_ids,
            anchor_ids=example.anchor_ids,
            controller_vectors=None,
            gate_strength=0.0,
        )
        return base_logits, base_states, keep_logits, keep_states
    return base_logits, base_states, None, None


def stack_traces(items: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = torch.stack([x[0] for x in items], dim=0)
    states = torch.stack([x[1] for x in items], dim=0)
    return logits, states


@torch.no_grad()
def stack_reference_trace_set(
    refs: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]],
    kind: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    base_logits = torch.stack([x[0] for x in refs], dim=0)
    base_states = torch.stack([x[1] for x in refs], dim=0)
    if kind == "fr":
        keep_logits = torch.stack([x[2] for x in refs if x[2] is not None], dim=0)
        keep_states = torch.stack([x[3] for x in refs if x[3] is not None], dim=0)
        return base_logits, base_states, keep_logits, keep_states
    return base_logits, base_states, None, None


def gather_train_traces(
    model,
    paver: ResidualPaver,
    controller_vectors: torch.Tensor,
    gate: SoftFrenchGate,
    examples: Sequence[TrainingExample],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boundary_batch = torch.stack([ex.boundary_states for ex in examples], dim=0)
    gate_logits, gate_probs = gate(boundary_batch)

    train_items: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for ex, gate_prob in zip(examples, gate_probs):
        train_logits, train_states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.base_ids,
            anchor_ids=ex.anchor_ids,
            controller_vectors=controller_vectors,
            gate_strength=gate_prob,
        )
        train_items.append((train_logits, train_states))

    train_logits, train_states = stack_traces(train_items)
    labels = torch.tensor([ex.gate_target for ex in examples], device=gate_logits.device, dtype=gate_logits.dtype)
    return train_logits, train_states, gate_logits, labels


@torch.no_grad()
def collect_reference_set(
    model,
    paver: ResidualPaver,
    examples: Sequence[TrainingExample],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    refs = [collect_reference_trace(model, paver, ex) for ex in examples]
    return stack_reference_trace_set(refs, examples[0].kind)


@torch.no_grad()
def decode_text(
    model,
    tokenizer,
    paver: ResidualPaver,
    gate: SoftFrenchGate,
    controller_vectors: torch.Tensor,
    device: torch.device,
    user_prompt: str,
    max_new_tokens: int,
    strength: float,
    temperature: float,
    top_p: float,
) -> Tuple[float, str]:
    base_ids = encode_chat_prompt(tokenizer, user_prompt, None, device)
    boundary = capture_boundary_states(model, base_ids, paver).unsqueeze(0)
    _, gate_prob = gate(boundary)
    gate_strength = float(gate_prob.item()) * float(strength)

    generated: List[int] = []
    for step in range(max_new_tokens):
        if generated:
            gen_tensor = torch.tensor(generated, device=device, dtype=torch.long).view(1, -1)
            curr_ids = torch.cat([base_ids, gen_tensor], dim=1)
        else:
            curr_ids = base_ids

        use_edit = step < controller_vectors.shape[0] and abs(gate_strength) > 1e-8
        with paver.intervene_and_capture(
            step_vectors=controller_vectors[step] if use_edit else None,
            batch_scale=gate_strength if use_edit else None,
        ):
            outputs = model_forward(model, curr_ids)
        logits = outputs.logits[:, -1, :].float()
        next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
        generated.append(int(next_id.item()))

        eos_id = tokenizer.eos_token_id
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    text = tokenizer.decode(torch.tensor(generated, device=device), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return gate_strength, text


@dataclass
class TrainState:
    raw_vectors: torch.Tensor
    layer_indices: List[int]
    gate_state_dict: Dict[str, torch.Tensor]
    model_id: str
    steer_steps: int
    gate_sharpness: float
    layer_ramp_start: float
    layer_ramp_end: float
    time_decay_tau: float
    post_smooth_layer: float
    post_smooth_time: float
    loss_config: Dict[str, float]
    projection_config: Dict[str, int]


def save_train_state(path: str, state: TrainState):
    payload = asdict(state)
    torch.save(payload, path)
    print(f"Saved state to {path}")


def load_train_state(path: str, device: torch.device) -> Dict:
    obj = torch.load(path, map_location=device)
    gate_state_dict = {}
    for key, value in obj["gate_state_dict"].items():
        gate_state_dict[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    obj["gate_state_dict"] = gate_state_dict
    if isinstance(obj["raw_vectors"], torch.Tensor):
        obj["raw_vectors"] = obj["raw_vectors"].to(device)
    return obj


def resolve_compute_dtype(spec: Optional[str], *, fallback: torch.dtype) -> torch.dtype:
    value = str(spec or "auto").strip().lower()
    if value in {"", "auto"}:
        return fallback
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported compute dtype: {spec}")
    return mapping[value]


def prepare_model_load_kwargs(
    *,
    device: torch.device,
    dtype: torch.dtype,
    quantization: Optional[str] = None,
    quant_compute_dtype: Optional[str] = "auto",
) -> Tuple[Dict[str, Any], bool, str]:
    mode = str(quantization or "none").strip().lower()
    if mode in {"", "none", "no"}:
        return {"torch_dtype": dtype}, True, "none"

    if device.type != "cuda":
        raise ValueError(f"Quantized loading requires a CUDA device, got {device}.")

    compute_dtype = resolve_compute_dtype(quant_compute_dtype, fallback=dtype)
    device_index = 0 if device.index is None else int(device.index)

    if mode == "8bit":
        return (
            {
                "torch_dtype": compute_dtype,
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                "device_map": {"": device_index},
            },
            False,
            "8bit",
        )

    if mode == "4bit":
        return (
            {
                "torch_dtype": compute_dtype,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                ),
                "device_map": {"": device_index},
            },
            False,
            "4bit",
        )

    raise ValueError(f"Unsupported quantization mode: {quantization}")


def load_model_and_tokenizer(
    model_id: str,
    device: torch.device,
    *,
    quantization: Optional[str] = None,
    quant_compute_dtype: Optional[str] = "auto",
    revision: Optional[str] = None,
):
    dtype = pick_dtype()
    load_kwargs, move_to_device, quant_mode = prepare_model_load_kwargs(
        device=device,
        dtype=dtype,
        quantization=quantization,
        quant_compute_dtype=quant_compute_dtype,
    )
    compute_dtype = resolve_compute_dtype(quant_compute_dtype, fallback=dtype)
    print(
        f"Loading {model_id} on {device} with dtype={dtype} "
        f"(quantization={quant_mode}, compute_dtype={compute_dtype}) ..."
    )

    token = None
    try:
        import os

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    except Exception:
        token = None

    revision_kwargs = {"revision": str(revision)} if revision else {}
    cfg = AutoConfig.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
        **revision_kwargs,
    )
    model_type = str(getattr(cfg, "model_type", "") or "")
    if model_type in {"gemma3", "gemma3_text"}:
        # Transformers' Gemma-3 attention dispatch can trip TorchDynamo on this
        # NVIDIA PyTorch 2.3 container; eager attention is also what the bundled
        # external Gemma baseline wrapper uses for Gemma-family models.
        load_kwargs.setdefault("attn_implementation", "eager")
    if model_type == "gemma3" and Gemma3ForConditionalGeneration is not None:
        # Gemma-3 4B/12B/27B instruction checkpoints are multimodal
        # Gemma3ForConditionalGeneration repos. For text-only experiments we use
        # AutoProcessor(...).tokenizer and the language_model layers accessed by
        # get_decoder_layers().
        processor = AutoProcessor.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            **revision_kwargs,
        )
        tokenizer = processor.tokenizer
        wrapped = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            **revision_kwargs,
            **load_kwargs,
        )
        model = Gemma3TextOnlyCausalLM(wrapped)
        setattr(model, "_rp_original_model_type", "gemma3")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            **revision_kwargs,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            **revision_kwargs,
            **load_kwargs,
        )
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    if move_to_device:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def average_dicts(history: List[Dict[str, float]]) -> Dict[str, float]:
    if not history:
        return {}
    keys = sorted(set().union(*history))
    out: Dict[str, float] = {}
    for key in keys:
        values = [item[key] for item in history if key in item]
        out[key] = float(sum(values) / max(len(values), 1))
    return out


def train_hybrid_fr2en(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_id, device)
    total_layers = len(get_decoder_layers(model))
    layer_indices = parse_layer_indices(args.layers, total_layers)
    paver = ResidualPaver(model=model, layer_indices=layer_indices)
    hidden_size = get_hidden_size(model)
    num_layers = len(layer_indices)
    vocab_size = int(model.lm_head.weight.shape[0])

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    loss_config = HybridLossConfig(
        ce_weight=args.ce_weight,
        fr_kl_weight=args.fr_kl_weight,
        nonfr_kl_weight=args.nonfr_kl_weight,
        fr_traj_weight=args.fr_traj_weight,
        nonfr_traj_weight=args.nonfr_traj_weight,
        traj_prob_weight=args.traj_prob_weight,
        traj_logit_weight=args.traj_logit_weight,
        traj_state_weight=args.traj_state_weight,
        fr_improvement_target=args.fr_improvement_target,
        fr_baseline_floor=args.fr_baseline_floor,
        preserve_tolerance=args.preserve_tolerance,
        preserve_scale=args.preserve_scale,
        gate_bce_weight=args.gate_bce_weight,
        gate_entropy_weight=args.gate_entropy_weight,
        l2=args.l2,
        layer_smooth=args.layer_smooth,
        time_smooth=args.time_smooth,
    )
    proj_config = ProjectionConfig(
        vocab_proj_dim=args.vocab_proj_dim,
        state_proj_dim=args.state_proj_dim,
        proj_seed=args.seed,
    )
    scorer = ProjectedTrajectoryScorer(
        vocab_size=vocab_size,
        flat_state_dim=num_layers * hidden_size,
        config=loss_config,
        proj_config=proj_config,
    ).to(device)

    print("Building French / non-French training examples ...")
    fr_examples, nonfr_examples = build_training_examples(
        model=model,
        tokenizer=tokenizer,
        paver=paver,
        device=device,
        steer_steps=args.steer_steps,
    )
    print(f"French examples: {len(fr_examples)}")
    print(f"Non-French examples: {len(nonfr_examples)}")

    init_direction, init_threshold, init_stats = init_gate_from_examples(fr_examples, nonfr_examples)
    gate = SoftFrenchGate(num_layers=num_layers, hidden_size=hidden_size, sharpness=args.gate_sharpness).to(device)
    with torch.no_grad():
        gate.direction.copy_(init_direction)
        gate.bias.copy_(torch.tensor(-init_threshold, device=device, dtype=torch.float32))

    raw_vectors = nn.Parameter(
        torch.zeros(args.steer_steps, num_layers, hidden_size, device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([raw_vectors] + list(gate.parameters()), lr=args.lr)

    print(
        "[gate init] "
        f"train_acc={init_stats['train_acc']:.3f} "
        f"fr_mean={init_stats['fr_mean']:.4f} "
        f"nonfr_mean={init_stats['nonfr_mean']:.4f} "
        f"threshold={init_stats['threshold']:.4f}"
    )

    for epoch in range(1, args.epochs + 1):
        fr_shuffled = list(fr_examples)
        nonfr_shuffled = list(nonfr_examples)
        random.shuffle(fr_shuffled)
        random.shuffle(nonfr_shuffled)

        fr_batches = batched(fr_shuffled, args.fr_batch_size)
        nonfr_batches = batched(nonfr_shuffled, args.nonfr_batch_size)
        fr_cycle = itertools.cycle(fr_batches)
        nonfr_cycle = itertools.cycle(nonfr_batches)
        num_updates = max(len(fr_batches), len(nonfr_batches))

        epoch_records: List[Dict[str, float]] = []

        for _ in range(num_updates):
            optimizer.zero_grad(set_to_none=True)

            controller = finalize_paving_vectors(
                raw_vectors,
                layer_start=args.layer_ramp_start,
                layer_end=args.layer_ramp_end,
                time_decay_tau=args.time_decay_tau,
                smooth_layer=args.post_smooth_layer,
                smooth_time=args.post_smooth_time,
            )

            fr_batch = next(fr_cycle)
            nonfr_batch = next(nonfr_cycle)

            # French batch: move toward English teacher.
            fr_train_logits, fr_train_states, fr_gate_logits, fr_gate_targets = gather_train_traces(
                model=model,
                paver=paver,
                controller_vectors=controller,
                gate=gate,
                examples=fr_batch,
            )
            fr_base_logits, fr_base_states, fr_keep_logits, fr_keep_states = collect_reference_set(
                model=model,
                paver=paver,
                examples=fr_batch,
            )
            assert fr_keep_logits is not None and fr_keep_states is not None

            fr_targets = torch.stack([ex.anchor_ids for ex in fr_batch], dim=0)
            fr_ce = F.cross_entropy(
                fr_train_logits.reshape(-1, fr_train_logits.shape[-1]),
                fr_targets.reshape(-1),
            )
            fr_teacher_probs = torch.softmax(fr_keep_logits.detach(), dim=-1)
            fr_kl = F.kl_div(
                F.log_softmax(fr_train_logits, dim=-1),
                fr_teacher_probs,
                reduction="batchmean",
            )
            fr_train_step_dist = scorer.stepwise_distances(
                fr_train_logits, fr_keep_logits, fr_train_states, fr_keep_states
            )
            fr_base_step_dist = scorer.stepwise_distances(
                fr_base_logits, fr_keep_logits, fr_base_states, fr_keep_states
            )
            fr_traj = relative_improvement_surrogate(
                fr_train_step_dist,
                fr_base_step_dist,
                target_fraction=loss_config.fr_improvement_target,
                floor=loss_config.fr_baseline_floor,
                eps=loss_config.eps,
            ).mean()
            fr_gate_bce = F.binary_cross_entropy_with_logits(fr_gate_logits, fr_gate_targets)

            # Non-French batch: preserve base behavior.
            nonfr_train_logits, nonfr_train_states, nonfr_gate_logits, nonfr_gate_targets = gather_train_traces(
                model=model,
                paver=paver,
                controller_vectors=controller,
                gate=gate,
                examples=nonfr_batch,
            )
            nonfr_base_logits, nonfr_base_states, _, _ = collect_reference_set(
                model=model,
                paver=paver,
                examples=nonfr_batch,
            )
            nonfr_base_probs = torch.softmax(nonfr_base_logits.detach(), dim=-1)
            nonfr_kl = F.kl_div(
                F.log_softmax(nonfr_train_logits, dim=-1),
                nonfr_base_probs,
                reduction="batchmean",
            )
            nonfr_step_dist = scorer.stepwise_distances(
                nonfr_train_logits,
                nonfr_base_logits,
                nonfr_train_states,
                nonfr_base_states,
            )
            nonfr_traj = preservation_surrogate(
                nonfr_step_dist,
                tolerance=loss_config.preserve_tolerance,
                scale=loss_config.preserve_scale,
            ).mean()
            nonfr_gate_bce = F.binary_cross_entropy_with_logits(nonfr_gate_logits, nonfr_gate_targets)

            gate_entropy = 0.0
            if loss_config.gate_entropy_weight > 0:
                all_gate_probs = torch.sigmoid(torch.cat([fr_gate_logits, nonfr_gate_logits], dim=0))
                gate_entropy = -(
                    all_gate_probs * torch.log(all_gate_probs.clamp_min(loss_config.eps))
                    + (1.0 - all_gate_probs) * torch.log((1.0 - all_gate_probs).clamp_min(loss_config.eps))
                ).mean()

            reg = paving_regularizer(
                vectors=raw_vectors,
                l2=loss_config.l2,
                layer_smooth=loss_config.layer_smooth,
                time_smooth=loss_config.time_smooth,
            )

            total_loss = (
                loss_config.ce_weight * fr_ce
                + loss_config.fr_kl_weight * fr_kl
                + loss_config.nonfr_kl_weight * nonfr_kl
                + loss_config.fr_traj_weight * fr_traj
                + loss_config.nonfr_traj_weight * nonfr_traj
                + loss_config.gate_bce_weight * (fr_gate_bce + nonfr_gate_bce)
                + loss_config.gate_entropy_weight * gate_entropy
                + reg
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([raw_vectors] + list(gate.parameters()), max_norm=1.0)
            optimizer.step()

            epoch_records.append(
                {
                    "total": float(total_loss.detach().cpu()),
                    "fr_ce": float(fr_ce.detach().cpu()),
                    "fr_kl": float(fr_kl.detach().cpu()),
                    "fr_traj": float(fr_traj.detach().cpu()),
                    "nonfr_kl": float(nonfr_kl.detach().cpu()),
                    "nonfr_traj": float(nonfr_traj.detach().cpu()),
                    "fr_gate_bce": float(fr_gate_bce.detach().cpu()),
                    "nonfr_gate_bce": float(nonfr_gate_bce.detach().cpu()),
                    "reg": float(reg.detach().cpu()),
                }
            )

        avg = average_dicts(epoch_records)
        gstats = gate_stats(gate, fr_examples, nonfr_examples)
        print(
            f"[epoch {epoch:02d}] "
            f"total={avg.get('total', 0.0):.4f} "
            f"fr_ce={avg.get('fr_ce', 0.0):.4f} "
            f"fr_kl={avg.get('fr_kl', 0.0):.4f} "
            f"fr_traj={avg.get('fr_traj', 0.0):.4f} "
            f"nonfr_kl={avg.get('nonfr_kl', 0.0):.4f} "
            f"nonfr_traj={avg.get('nonfr_traj', 0.0):.4f} "
            f"fr_gate_mean={gstats['fr_gate_mean']:.3f} "
            f"nonfr_gate_mean={gstats['nonfr_gate_mean']:.3f}"
        )

        if args.preview_prompt:
            controller_eval = finalize_paving_vectors(
                raw_vectors.detach(),
                layer_start=args.layer_ramp_start,
                layer_end=args.layer_ramp_end,
                time_decay_tau=args.time_decay_tau,
                smooth_layer=args.post_smooth_layer,
                smooth_time=args.post_smooth_time,
            )
            gate_strength, preview = decode_text(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                gate=gate,
                controller_vectors=controller_eval,
                device=device,
                user_prompt=args.preview_prompt,
                max_new_tokens=args.max_new_tokens,
                strength=args.strength,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"  preview_gate={gate_strength:.3f}  preview={preview!r}")

    final_controller = finalize_paving_vectors(
        raw_vectors.detach(),
        layer_start=args.layer_ramp_start,
        layer_end=args.layer_ramp_end,
        time_decay_tau=args.time_decay_tau,
        smooth_layer=args.post_smooth_layer,
        smooth_time=args.post_smooth_time,
    )

    if args.save_path:
        save_train_state(
            args.save_path,
            TrainState(
                raw_vectors=raw_vectors.detach().cpu(),
                layer_indices=layer_indices,
                gate_state_dict={k: v.detach().cpu() for k, v in gate.state_dict().items()},
                model_id=args.model_id,
                steer_steps=args.steer_steps,
                gate_sharpness=args.gate_sharpness,
                layer_ramp_start=args.layer_ramp_start,
                layer_ramp_end=args.layer_ramp_end,
                time_decay_tau=args.time_decay_tau,
                post_smooth_layer=args.post_smooth_layer,
                post_smooth_time=args.post_smooth_time,
                loss_config=asdict(loss_config),
                projection_config=asdict(proj_config),
            ),
        )

    if args.preview_prompt:
        gate_strength, preview = decode_text(
            model=model,
            tokenizer=tokenizer,
            paver=paver,
            gate=gate,
            controller_vectors=final_controller,
            device=device,
            user_prompt=args.preview_prompt,
            max_new_tokens=args.max_new_tokens,
            strength=args.strength,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("\n===== FINAL PREVIEW =====")
        print(preview)
        print(f"[gate_strength={gate_strength:.4f}]")


def infer_with_saved_state(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_train_state(args.load_path, device)

    model_id = state.get("model_id", args.model_id or DEFAULT_MODEL_ID)
    model, tokenizer = load_model_and_tokenizer(model_id, device)
    paver = ResidualPaver(model=model, layer_indices=[int(x) for x in state["layer_indices"]])
    hidden_size = get_hidden_size(model)
    gate = SoftFrenchGate(
        num_layers=len(state["layer_indices"]),
        hidden_size=hidden_size,
        sharpness=float(state["gate_sharpness"]),
    ).to(device)
    gate.load_state_dict(state["gate_state_dict"])
    gate.eval()

    controller = finalize_paving_vectors(
        state["raw_vectors"].to(device),
        layer_start=float(state["layer_ramp_start"]),
        layer_end=float(state["layer_ramp_end"]),
        time_decay_tau=float(state["time_decay_tau"]),
        smooth_layer=float(state["post_smooth_layer"]),
        smooth_time=float(state["post_smooth_time"]),
    )

    gate_strength, text = decode_text(
        model=model,
        tokenizer=tokenizer,
        paver=paver,
        gate=gate,
        controller_vectors=controller,
        device=device,
        user_prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        strength=args.strength,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n===== HYBRID RESIDUAL-PAVING OUTPUT =====")
    print(text)
    print("\n[gate debug]")
    print(f"gate_strength={gate_strength:.6f}")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--layers", type=str, default="-4,-3,-2,-1")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--steer_steps", type=int, default=24)
    p_train.add_argument("--epochs", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=0.08)
    p_train.add_argument("--fr_batch_size", type=int, default=2)
    p_train.add_argument("--nonfr_batch_size", type=int, default=4)
    p_train.add_argument("--gate_sharpness", type=float, default=10.0)
    p_train.add_argument("--layer_ramp_start", type=float, default=0.85)
    p_train.add_argument("--layer_ramp_end", type=float, default=1.15)
    p_train.add_argument("--time_decay_tau", type=float, default=10.0)
    p_train.add_argument("--post_smooth_layer", type=float, default=0.15)
    p_train.add_argument("--post_smooth_time", type=float, default=0.15)
    p_train.add_argument("--vocab_proj_dim", type=int, default=64)
    p_train.add_argument("--state_proj_dim", type=int, default=64)

    p_train.add_argument("--ce_weight", type=float, default=1.0)
    p_train.add_argument("--fr_kl_weight", type=float, default=0.35)
    p_train.add_argument("--nonfr_kl_weight", type=float, default=0.75)
    p_train.add_argument("--fr_traj_weight", type=float, default=1.0)
    p_train.add_argument("--nonfr_traj_weight", type=float, default=1.0)
    p_train.add_argument("--traj_prob_weight", type=float, default=1.0)
    p_train.add_argument("--traj_logit_weight", type=float, default=0.35)
    p_train.add_argument("--traj_state_weight", type=float, default=0.35)
    p_train.add_argument("--fr_improvement_target", type=float, default=0.35)
    p_train.add_argument("--fr_baseline_floor", type=float, default=0.05)
    p_train.add_argument("--preserve_tolerance", type=float, default=0.05)
    p_train.add_argument("--preserve_scale", type=float, default=0.25)
    p_train.add_argument("--gate_bce_weight", type=float, default=0.20)
    p_train.add_argument("--gate_entropy_weight", type=float, default=0.00)
    p_train.add_argument("--l2", type=float, default=1e-4)
    p_train.add_argument("--layer_smooth", type=float, default=5e-4)
    p_train.add_argument("--time_smooth", type=float, default=5e-4)
    p_train.add_argument("--preview_prompt", type=str, default="Explique la récursion simplement.")
    p_train.add_argument("--save_path", type=str, default="gemma3_fr2en_hybrid_paving.pt")

    p_infer = subparsers.add_parser("infer")
    p_infer.add_argument("--load_path", type=str, required=True)
    p_infer.add_argument("--prompt", type=str, required=True)

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.mode == "train":
        train_hybrid_fr2en(args)
    elif args.mode == "infer":
        infer_with_saved_state(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
