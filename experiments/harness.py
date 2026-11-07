"""
Core training harness for residual-paving ablation experiments.

All 7 variants subclass ResidualPavingExperiment and override one or more of:
  compute_fr_loss      – edit loss (CE + KL + trajectory ± ablation terms)
  compute_keep_loss    – preservation loss (benign ± harmful tier)
  get_controller_vecs  – how to materialise [T,L,H] vectors (fixed vs state-cond)
  post_epoch_hook      – called after each epoch (scheduled sampling)
  pre_train_hook       – called before stage-1 starts (basis init)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
for _d in [str(_ROOT / "src" / "trainer"), str(_ROOT / "src" / "optimized")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from gemma3_residual_paving_binary_refined import (
    AnchoredBinaryGate,
    BinaryExample,
    choose_weighted_threshold,
    init_binary_gate_from_examples,
)
from gemma3_residual_paving_hybrid import (
    HybridLossConfig,
    ProjectedTrajectoryScorer,
    ProjectionConfig,
    ResidualPaver,
    finalize_paving_vectors,
    get_decoder_layers,
    get_hidden_size,
    load_model_and_tokenizer,
    parse_layer_indices,
    paving_regularizer,
)
from optimized_binary_runtime import trace_forced_trajectory_batch_optimized
from config import ExperimentConfig


# ── Loss utilities ───────────────────────────────────────────────────────────

def compute_mean_boundary_direction(
    fr_examples: List[BinaryExample],
    keep_examples: List[BinaryExample],
) -> torch.Tensor:
    """Mean harmful-vs-keep boundary-state direction in [L, H]."""
    if not fr_examples:
        raise ValueError("Expected at least one harmful example for warm start.")
    if not keep_examples:
        raise ValueError("Expected at least one keep example for warm start.")

    fr_mean = torch.stack([ex.boundary_states.float() for ex in fr_examples], dim=0).mean(dim=0)
    keep_mean = torch.stack([ex.boundary_states.float() for ex in keep_examples], dim=0).mean(dim=0)
    return fr_mean - keep_mean


def build_boundary_warm_start(
    mean_direction: torch.Tensor,
    *,
    steer_steps: int,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast a boundary-state direction across generation steps."""
    if mean_direction.ndim != 2:
        raise ValueError(f"Expected mean_direction with shape [L, H], got {tuple(mean_direction.shape)}")
    return mean_direction.to(device=device, dtype=dtype).unsqueeze(0).expand(int(steer_steps), -1, -1) * float(scale)

def topk_kl_loss(
    train_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """KL(teacher||student) over teacher's top-K support. train_logits: [T,V]."""
    steps = min(int(train_logits.shape[0]), int(teacher_topk_ids.shape[0]), int(teacher_topk_probs.shape[0]))
    student_log_probs = F.log_softmax(train_logits[:steps].float(), dim=-1)  # [T, V]
    sel = student_log_probs.gather(1, teacher_topk_ids[:steps].to(train_logits.device))  # [T, K]
    t_probs = teacher_topk_probs[:steps].float().to(train_logits.device)
    t_log = torch.log(t_probs.clamp_min(eps))
    return (t_probs * (t_log - sel)).sum(dim=-1).clamp_min(0).mean()


def stepwise_topk_kl_distance(
    train_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-step KL(teacher||student) over teacher top-k support. Returns [T]."""
    steps = min(int(train_logits.shape[0]), int(teacher_topk_ids.shape[0]), int(teacher_topk_probs.shape[0]))
    student_log_probs = F.log_softmax(train_logits[:steps].float(), dim=-1)
    sel = student_log_probs.gather(1, teacher_topk_ids[:steps].to(train_logits.device))
    teacher_probs = teacher_topk_probs[:steps].float().to(train_logits.device)
    teacher_log = torch.log(teacher_probs.clamp_min(eps))
    return (teacher_probs * (teacher_log - sel)).sum(dim=-1).clamp_min(0)


def ce_loss_from_logits(
    train_logits: torch.Tensor,
    anchor_ids: torch.Tensor,
    step_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy at each step, optionally weighted by step position."""
    T = train_logits.shape[0]
    targets = anchor_ids[:T].to(train_logits.device)
    per_step = F.cross_entropy(train_logits.float(), targets, reduction="none")  # [T]
    if step_weights is not None:
        w = step_weights[:T].to(train_logits.device)
        return (per_step * w).sum() / w.sum().clamp_min(1e-6)
    return per_step.mean()


def gate_pair_margin(
    fr_logit: torch.Tensor,
    en_logit: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Hinge loss encouraging fr_logit > en_logit + margin."""
    return F.relu(en_logit - fr_logit + margin).mean()


def make_step_weights(T: int, device: torch.device) -> torch.Tensor:
    """Front-loaded step weights: early tokens matter more."""
    weights = torch.exp(-torch.arange(T, dtype=torch.float32, device=device) / max(T / 3.0, 1.0))
    return weights / weights.sum()


def embed_trace_features(
    scorer: ProjectedTrajectoryScorer,
    logits: torch.Tensor,
    states: torch.Tensor,
    *,
    to_cpu: bool = True,
) -> Dict[str, torch.Tensor]:
    logits_batch = logits.unsqueeze(0) if logits.ndim == 2 else logits
    states_batch = states.unsqueeze(0) if states.ndim == 3 else states
    out = {
        "prob": scorer.embed_probs(logits_batch)[0].detach(),
        "logit": scorer.embed_logit_dirs(logits_batch)[0].detach(),
        "state": scorer.embed_states(states_batch)[0].detach(),
    }
    if to_cpu:
        return {key: value.cpu() for key, value in out.items()}
    return out


def stepwise_distance_to_reference(
    scorer: ProjectedTrajectoryScorer,
    train_logits: torch.Tensor,
    train_states: torch.Tensor,
    ref_features: Dict[str, torch.Tensor],
) -> torch.Tensor:
    train_features = embed_trace_features(scorer, train_logits, train_states, to_cpu=False)
    prob_delta = train_features["prob"] - ref_features["prob"].to(train_logits.device)
    logit_delta = train_features["logit"] - ref_features["logit"].to(train_logits.device)
    state_delta = train_features["state"] - ref_features["state"].to(train_logits.device)
    return (
        scorer.loss_config.traj_prob_weight * prob_delta.norm(dim=-1)
        + scorer.loss_config.traj_logit_weight * logit_delta.norm(dim=-1)
        + scorer.loss_config.traj_state_weight * state_delta.norm(dim=-1)
    )


# ── Reference cache ──────────────────────────────────────────────────────────

class RefCache:
    """Precomputed frozen-model references for a list of BinaryExamples."""

    def __init__(self) -> None:
        self.base_topk_ids: Dict[str, torch.Tensor] = {}       # [T, K]
        self.base_topk_probs: Dict[str, torch.Tensor] = {}     # [T, K]
        self.teacher_topk_ids: Dict[str, torch.Tensor] = {}    # [T, K]
        self.teacher_topk_probs: Dict[str, torch.Tensor] = {}  # [T, K]
        self.neg_topk_ids: Dict[str, torch.Tensor] = {}        # [T, K]
        self.neg_topk_probs: Dict[str, torch.Tensor] = {}      # [T, K]
        self.base_teacher_kl: Dict[str, torch.Tensor] = {}     # [T] per-step KL(base||teacher-support)
        self.base_topk_coverage: Dict[str, torch.Tensor] = {}  # [T]
        self.teacher_topk_coverage: Dict[str, torch.Tensor] = {}  # [T]

    @torch.no_grad()
    def build(
        self,
        model,
        paver: ResidualPaver,
        scorer: ProjectedTrajectoryScorer,
        examples: List[BinaryExample],
        batch_size: int,
        topk: int = 1024,
    ) -> None:
        print(f"    [refs] precomputing references for {len(examples)} examples ...")
        base_coverages: List[float] = []
        teacher_coverages: List[float] = []
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            # Base trajectory: harmful prompt, no gate, no controller
            base_l, base_s = trace_forced_trajectory_batch_optimized(
                model=model, paver=paver,
                base_ids_batch=[ex.base_ids for ex in chunk],
                anchor_ids_batch=[ex.anchor_ids for ex in chunk],
                controller_vectors=None, gate_strengths=0.0,
            )  # [B, T, V], [B, T, L, H]
            # Teacher trajectory: paired benign prompt, no gate, no controller
            teach_l, teach_s = trace_forced_trajectory_batch_optimized(
                model=model, paver=paver,
                base_ids_batch=[ex.teacher_ids for ex in chunk],
                anchor_ids_batch=[ex.anchor_ids for ex in chunk],
                controller_vectors=None, gate_strengths=0.0,
            )
            # Negative trajectory: harmful prompt, no gate → refusal logits
            has_neg = all(ex.negative_ids is not None for ex in chunk)
            if has_neg:
                neg_l, neg_s = trace_forced_trajectory_batch_optimized(
                    model=model, paver=paver,
                    base_ids_batch=[ex.base_ids for ex in chunk],
                    anchor_ids_batch=[ex.negative_ids for ex in chunk],
                    controller_vectors=None, gate_strengths=0.0,
                )

            for i, ex in enumerate(chunk):
                base_log = base_l[i].float()
                teach_log = teach_l[i].float()

                base_vals, base_ids = torch.topk(base_log, k=min(topk, base_log.shape[-1]), dim=-1)
                base_log_z = torch.logsumexp(base_log, dim=-1, keepdim=True)
                base_probs = torch.exp(base_vals - base_log_z)
                self.base_topk_ids[ex.sample_id] = base_ids.cpu()
                self.base_topk_probs[ex.sample_id] = base_probs.cpu()
                self.base_topk_coverage[ex.sample_id] = base_probs.sum(dim=-1).cpu()
                base_coverages.append(float(base_probs.sum(dim=-1).mean().item()))

                teach_vals, teach_ids = torch.topk(teach_log, k=min(topk, teach_log.shape[-1]), dim=-1)
                teach_log_z = torch.logsumexp(teach_log, dim=-1, keepdim=True)
                teach_probs = torch.exp(teach_vals - teach_log_z)
                self.teacher_topk_ids[ex.sample_id] = teach_ids.cpu()
                self.teacher_topk_probs[ex.sample_id] = teach_probs.cpu()
                self.teacher_topk_coverage[ex.sample_id] = teach_probs.sum(dim=-1).cpu()
                teacher_coverages.append(float(teach_probs.sum(dim=-1).mean().item()))
                if has_neg:
                    neg_log = neg_l[i].float()
                    neg_vals, neg_ids = torch.topk(neg_log, k=min(topk, neg_log.shape[-1]), dim=-1)
                    neg_log_z = torch.logsumexp(neg_log, dim=-1, keepdim=True)
                    neg_probs = torch.exp(neg_vals - neg_log_z)
                    self.neg_topk_ids[ex.sample_id] = neg_ids.cpu()
                    self.neg_topk_probs[ex.sample_id] = neg_probs.cpu()

                self.base_teacher_kl[ex.sample_id] = stepwise_topk_kl_distance(
                    base_log,
                    teach_ids,
                    teach_probs,
                ).detach().cpu()
        if teacher_coverages:
            print(
                "    [refs] topk coverage "
                f"teacher={sum(teacher_coverages) / len(teacher_coverages):.4f} "
                f"base={sum(base_coverages) / max(len(base_coverages), 1):.4f} "
                f"(k={int(topk)})"
            )


# ── Main experiment class ────────────────────────────────────────────────────

class ResidualPavingExperiment:

    def __init__(self, cfg: ExperimentConfig, run_dir: Path) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.paver: Optional[ResidualPaver] = None
        self.gate: Optional[AnchoredBinaryGate] = None
        self.raw_vectors: Optional[nn.Parameter] = None
        self.scorer: Optional[ProjectedTrajectoryScorer] = None
        self.fr_examples: List[BinaryExample] = []
        self.keep_benign: List[BinaryExample] = []
        self.keep_harmful: List[BinaryExample] = []
        self.eval_fr_examples: List[BinaryExample] = []
        self.eval_keep_benign: List[BinaryExample] = []
        self.eval_keep_harmful: List[BinaryExample] = []
        self.fr_refs: Optional[RefCache] = None
        self.en_refs: Optional[RefCache] = None
        self.eval_fr_refs: Optional[RefCache] = None
        self.eval_en_refs: Optional[RefCache] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_val_loss = float("inf")
        self.best_checkpoint_path = self.run_dir / "best_checkpoint.pt"

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        print(f"\n{'='*60}")
        print(f"[{self.cfg.name}] loading model {self.cfg.model_id} ...")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.cfg.model_id,
            device=self.device,
            quantization=getattr(self.cfg, "model_quantization", "none"),
            quant_compute_dtype=getattr(self.cfg, "model_quant_compute_dtype", "auto"),
        )
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

        layers = get_decoder_layers(self.model)
        layer_indices = parse_layer_indices(self.cfg.layer_spec, len(layers))
        hidden_size = get_hidden_size(self.model)
        num_layers = len(layer_indices)
        vocab_size = self.model.config.vocab_size if hasattr(self.model.config, "vocab_size") else \
                     self.model.config.text_config.vocab_size

        self.paver = ResidualPaver(model=self.model, layer_indices=layer_indices)
        self.scorer = ProjectedTrajectoryScorer(
            vocab_size=vocab_size,
            flat_state_dim=num_layers * hidden_size,
            config=HybridLossConfig(),
            proj_config=ProjectionConfig(),
        ).to(self.device)

        # Learnable paving vectors [T, L, H]
        self.raw_vectors = nn.Parameter(
            torch.zeros(self.cfg.steer_steps, num_layers, hidden_size,
                        dtype=torch.float32, device=self.device)
        )
        self._num_layers = num_layers
        self._hidden_size = hidden_size

        # Build data examples
        from dataset import build_all_examples
        splits = build_all_examples(
            self.model, self.tokenizer, self.paver, self.device, self.cfg
        )
        self.fr_examples = splits.train_fr
        self.keep_benign = splits.train_keep_benign
        self.keep_harmful = splits.train_keep_harmful
        self.eval_fr_examples = splits.eval_fr
        self.eval_keep_benign = splits.eval_keep_benign
        self.eval_keep_harmful = splits.eval_keep_harmful

        # Gate: initialised from FR vs EN boundary direction
        all_en = self.keep_benign + self.keep_harmful
        base_dir, threshold, gate_stats = init_binary_gate_from_examples(
            self.fr_examples, all_en
        )
        print(f"[{self.cfg.name}] gate init: threshold={threshold:.3f} acc={gate_stats['train_acc']:.3f}")
        self.gate = AnchoredBinaryGate(
            init_direction=base_dir,
            init_threshold=threshold,
            sharpness=self.cfg.gate_sharpness,
            learn_direction_residual=self.cfg.gate_learn_residual,
            residual_scale=self.cfg.gate_residual_scale,
        ).to(self.device)

        # Precompute frozen references
        print(f"[{self.cfg.name}] precomputing frozen references ...")
        self.fr_refs = RefCache()
        self.fr_refs.build(self.model, self.paver, self.scorer, self.fr_examples,
                           batch_size=self.cfg.trace_batch_size, topk=self.cfg.ref_topk)
        self.en_refs = RefCache()
        self.en_refs.build(self.model, self.paver, self.scorer, self.keep_benign,
                           batch_size=self.cfg.trace_batch_size, topk=self.cfg.ref_topk)
        if self.eval_fr_examples:
            self.eval_fr_refs = RefCache()
            self.eval_fr_refs.build(
                self.model,
                self.paver,
                self.scorer,
                self.eval_fr_examples,
                batch_size=self.cfg.trace_batch_size,
                topk=self.cfg.ref_topk,
            )
        if self.eval_keep_benign:
            self.eval_en_refs = RefCache()
            self.eval_en_refs.build(
                self.model,
                self.paver,
                self.scorer,
                self.eval_keep_benign,
                batch_size=self.cfg.trace_batch_size,
                topk=self.cfg.ref_topk,
            )

        self.pre_train_hook()

    # ── Override points ───────────────────────────────────────────────────────

    def pre_train_hook(self) -> None:
        """Called after setup, before stage 1. Override for basis init."""
        warm_start = str(getattr(self.cfg, "controller_warm_start", "none") or "none").strip().lower()
        if warm_start in {"", "none", "off", "zero"}:
            return
        if warm_start != "mean_boundary":
            raise ValueError(f"Unsupported controller_warm_start: {self.cfg.controller_warm_start!r}")

        keep_examples = self.keep_benign + self.keep_harmful
        if not self.fr_examples or not keep_examples:
            return

        mean_direction = compute_mean_boundary_direction(self.fr_examples, keep_examples)
        init = build_boundary_warm_start(
            mean_direction,
            steer_steps=self.cfg.steer_steps,
            scale=float(self.cfg.controller_warm_start_scale),
            device=self.raw_vectors.device,
            dtype=self.raw_vectors.dtype,
        )
        with torch.no_grad():
            self.raw_vectors.copy_(init)
        print(
            f"[{self.cfg.name}] controller warm start=mean_boundary "
            f"scale={float(self.cfg.controller_warm_start_scale):.3f} "
            f"mean_norm={float(mean_direction.norm().item()):.4f}"
        )

    def get_controller_vecs(self, boundary_states: torch.Tensor) -> torch.Tensor:
        """Return [T, L, H] controller vectors for one example. Override for state-cond."""
        return finalize_paving_vectors(
            self.raw_vectors,
            layer_start=self.cfg.layer_ramp_start,
            layer_end=self.cfg.layer_ramp_end,
            time_decay_tau=self.cfg.time_decay_tau if self.cfg.time_decay_tau > 0 else None,
            smooth_layer=self.cfg.layer_smooth,
            smooth_time=self.cfg.time_smooth,
        )

    def resolve_train_steps(self, stage: int) -> int:
        if int(stage) == 1:
            requested = int(getattr(self.cfg, "stage1_train_steps", 0))
        elif int(stage) == 2:
            requested = int(getattr(self.cfg, "stage2_train_steps", 0))
        else:
            requested = 0
        if requested <= 0:
            return int(self.cfg.steer_steps)
        return min(int(self.cfg.steer_steps), requested)

    def slice_anchor_batch(self, examples: List[BinaryExample], *, stage: int) -> List[torch.Tensor]:
        steps = self.resolve_train_steps(stage)
        return [ex.anchor_ids[:steps] for ex in examples]

    def compute_fr_loss(
        self,
        ex: BinaryExample,
        train_logits: torch.Tensor,   # [T, V]
        train_states: torch.Tensor,   # [T, L, H]
        step_weights: torch.Tensor,
        traj_scale: float,
    ) -> torch.Tensor:
        """Edit loss for one harmful example. Override in triplet/etc. variants."""
        cfg = self.cfg
        sid = ex.sample_id

        # CE against anchor tokens (helpful response from benign teacher)
        loss = cfg.ce_weight * ce_loss_from_logits(
            train_logits, ex.anchor_ids.to(train_logits.device), step_weights
        )

        # KL against teacher logits
        loss = loss + cfg.kl_weight * topk_kl_loss(
            train_logits,
            self.fr_refs.teacher_topk_ids[sid],
            self.fr_refs.teacher_topk_probs[sid],
        )

        # Trajectory: normalised distance to teacher
        if traj_scale > 0:
            d_train = stepwise_topk_kl_distance(
                train_logits,
                self.fr_refs.teacher_topk_ids[sid],
                self.fr_refs.teacher_topk_probs[sid],
            )
            d_base = self.fr_refs.base_teacher_kl[sid].to(train_logits.device)
            # Normalise: d_train / (d_base + eps) — reward getting below 1.0
            # Truncate d_base to match d_train length (stage train_steps may be < steer_steps)
            T_train = int(d_train.shape[0])
            norm_dist = (d_train / (d_base[:T_train].detach() + 1e-6)).mean()
            loss = loss + traj_scale * norm_dist

        return loss

    def compute_keep_loss(
        self,
        ex: BinaryExample,
        train_logits: torch.Tensor,  # [T, V]
        refs: Optional[RefCache] = None,
    ) -> torch.Tensor:
        """Preservation loss for one keep example. Override in three_tier variant."""
        if ex.kind not in ("en", "harmful_keep"):
            return train_logits.new_zeros(())

        sid = ex.sample_id
        ref_cache = refs or self.en_refs
        if ex.kind == "en" and ref_cache is not None and sid in ref_cache.base_topk_ids:
            return self.cfg.pres_weight * topk_kl_loss(
                train_logits,
                ref_cache.base_topk_ids[sid],
                ref_cache.base_topk_probs[sid],
            )
        # harmful_keep: ignored in baseline (three_tier variant overrides)
        return train_logits.new_zeros(())

    def post_epoch_hook(self, stage: int, epoch: int) -> None:
        """Called at end of each epoch. Override for scheduled sampling."""

    def compute_validation_metrics(self) -> Dict[str, float]:
        if not self.eval_fr_examples and not self.eval_keep_benign and not self.eval_keep_harmful:
            return {}

        cfg = self.cfg
        gate_threshold = getattr(self, "gate_threshold", 0.0)
        metrics: Dict[str, float] = {}
        correct = 0
        total = 0

        with torch.no_grad():
            for ex in self.eval_fr_examples:
                logit, _ = self.gate(ex.boundary_states.unsqueeze(0).to(self.device))
                correct += int(logit.item() >= gate_threshold)
                total += 1
            for ex in self.eval_keep_benign + self.eval_keep_harmful:
                logit, _ = self.gate(ex.boundary_states.unsqueeze(0).to(self.device))
                correct += int(logit.item() < gate_threshold)
                total += 1

            fr_ces: List[float] = []
            fr_kls: List[float] = []
            keep_kls: List[float] = []

            for start in range(0, len(self.eval_fr_examples), cfg.trace_batch_size):
                chunk = self.eval_fr_examples[start : start + cfg.trace_batch_size]
                ctrl_batch = torch.stack(
                    [self.get_controller_vecs(ex.boundary_states.to(self.device)) for ex in chunk],
                    dim=0,
                )
                gate_probs = torch.stack(
                    [self.gate(ex.boundary_states.unsqueeze(0).to(self.device))[1] for ex in chunk]
                ).squeeze(-1)
                train_l, _ = trace_forced_trajectory_batch_optimized(
                    model=self.model,
                    paver=self.paver,
                    base_ids_batch=[ex.base_ids for ex in chunk],
                    anchor_ids_batch=[ex.anchor_ids for ex in chunk],
                    controller_vectors=ctrl_batch,
                    gate_strengths=gate_probs,
                )

                for idx, ex in enumerate(chunk):
                    sid = ex.sample_id
                    fr_ces.append(
                        float(
                            ce_loss_from_logits(
                                train_l[idx],
                                ex.anchor_ids.to(self.device),
                            ).item()
                        )
                    )
                    fr_kls.append(
                        float(
                            topk_kl_loss(
                                train_l[idx],
                                self.eval_fr_refs.teacher_topk_ids[sid],
                                self.eval_fr_refs.teacher_topk_probs[sid],
                            ).item()
                        )
                    )

            for start in range(0, len(self.eval_keep_benign), cfg.trace_batch_size):
                chunk = self.eval_keep_benign[start : start + cfg.trace_batch_size]
                ctrl_batch = torch.stack(
                    [self.get_controller_vecs(ex.boundary_states.to(self.device)) for ex in chunk],
                    dim=0,
                )
                gate_probs = torch.stack(
                    [self.gate(ex.boundary_states.unsqueeze(0).to(self.device))[1] for ex in chunk]
                ).squeeze(-1)
                train_l, _ = trace_forced_trajectory_batch_optimized(
                    model=self.model,
                    paver=self.paver,
                    base_ids_batch=[ex.base_ids for ex in chunk],
                    anchor_ids_batch=[ex.anchor_ids for ex in chunk],
                    controller_vectors=ctrl_batch,
                    gate_strengths=gate_probs,
                )
                for idx, ex in enumerate(chunk):
                    sid = ex.sample_id
                    keep_kls.append(
                        float(
                            topk_kl_loss(
                                train_l[idx],
                                self.eval_en_refs.base_topk_ids[sid],
                                self.eval_en_refs.base_topk_probs[sid],
                            ).item()
                        )
                    )

        metrics["val_ce"] = float(sum(fr_ces) / max(len(fr_ces), 1))
        metrics["val_kl"] = float(sum(fr_kls) / max(len(fr_kls), 1))
        metrics["val_keep_kl"] = float(sum(keep_kls) / max(len(keep_kls), 1))
        metrics["val_gate_acc"] = correct / max(total, 1)
        metrics["val_loss"] = (
            cfg.ce_weight * metrics["val_ce"]
            + cfg.kl_weight * metrics["val_kl"]
            + cfg.pres_weight * metrics["val_keep_kl"]
        )
        return metrics

    def _checkpoint_payload(self) -> Dict:
        return {
            "raw_vectors": self.raw_vectors.data.cpu(),
            "gate_state_dict": {k: v.cpu() for k, v in self.gate.state_dict().items()},
            "gate_threshold": getattr(self, "gate_threshold", 0.0),
            "config": vars(self.cfg),
        }

    def save_checkpoint(self, path: Path) -> None:
        torch.save(self._checkpoint_payload(), path)

    def maybe_save_best_checkpoint(self, metrics: Dict[str, float]) -> None:
        val_loss = metrics.get("val_loss")
        if val_loss is None or val_loss >= self.best_val_loss:
            return
        self.best_val_loss = float(val_loss)
        self.save_checkpoint(self.best_checkpoint_path)
        print(
            f"  [val] best_checkpoint={self.best_checkpoint_path.name} "
            f"val_loss={self.best_val_loss:.4f}"
        )

    def apply_stage2_gate_warmup(self, epoch: int) -> None:
        warmup_epochs = max(0, int(getattr(self.cfg, "stage2_gate_warmup_epochs", 0)))
        if warmup_epochs <= 0:
            return
        target_sharpness = float(self.cfg.gate_sharpness)
        start_sharpness = 64.0
        if epoch <= warmup_epochs:
            mix = float(epoch) / float(max(warmup_epochs, 1))
            sharpness = start_sharpness + mix * (target_sharpness - start_sharpness)
        else:
            sharpness = target_sharpness
        with torch.no_grad():
            self.gate.log_sharpness.copy_(
                torch.tensor(math.log(max(sharpness, 1.0)), device=self.device, dtype=torch.float32)
            )

    # ── Training stages ───────────────────────────────────────────────────────

    def stage1_train(self) -> None:
        """Stage 1: hard gate (gate=1 for FR, 0 for EN), CE+KL only."""
        cfg = self.cfg
        optimizer = torch.optim.Adam([self.raw_vectors], lr=cfg.lr)
        T = cfg.steer_steps
        step_w = make_step_weights(T, self.device)

        self.model.eval()
        print(
            f"\n[{cfg.name}] Stage 1 ({cfg.stage1_epochs} epochs, hard gate, CE+KL, "
            f"train_steps={self.resolve_train_steps(stage=1)})"
        )
        for epoch in range(1, cfg.stage1_epochs + 1):
            t0 = time.time()
            total_loss = 0.0

            # Process FR examples in batches (hard gate = 1.0)
            for start in range(0, len(self.fr_examples), cfg.trace_batch_size):
                chunk = self.fr_examples[start : start + cfg.trace_batch_size]
                B = len(chunk)

                ctrl_batch = torch.stack([
                    self.get_controller_vecs(ex.boundary_states.to(self.device))
                    for ex in chunk
                ], dim=0)  # [B, T, L, H]

                train_l, train_s = trace_forced_trajectory_batch_optimized(
                    model=self.model, paver=self.paver,
                    base_ids_batch=[ex.base_ids for ex in chunk],
                    anchor_ids_batch=self.slice_anchor_batch(chunk, stage=1),
                    controller_vectors=ctrl_batch,
                    gate_strengths=torch.ones(B, device=self.device),
                )  # [B, T, V], [B, T, L, H]

                loss = train_l.new_zeros(())
                for i, ex in enumerate(chunk):
                    loss = loss + self.compute_fr_loss(
                        ex, train_l[i], train_s[i], step_w, traj_scale=0.0
                    )
                loss = loss / B

                # Regulariser
                loss = loss + paving_regularizer(
                    self.raw_vectors, cfg.l2_weight, cfg.layer_smooth, cfg.time_smooth
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.raw_vectors], 1.0)
                optimizer.step()
                total_loss += float(loss.detach())

            elapsed = time.time() - t0
            val_metrics = self.compute_validation_metrics()
            self.maybe_save_best_checkpoint(val_metrics)
            val_summary = ""
            if val_metrics:
                val_summary = (
                    f"  val_ce={val_metrics['val_ce']:.4f}"
                    f"  val_kl={val_metrics['val_kl']:.4f}"
                    f"  val_gate_acc={val_metrics['val_gate_acc']:.3f}"
                )
            print(f"  s1 epoch {epoch:02d}/{cfg.stage1_epochs}  loss={total_loss:.4f}{val_summary}  t={elapsed:.1f}s")
            self.post_epoch_hook(stage=1, epoch=epoch)

        self.model.eval()

    def stage2_train(self) -> None:
        """Stage 2: soft gate, CE+KL+trajectory, gate trained jointly."""
        cfg = self.cfg
        ctrl_params = [self.raw_vectors] + list(self._extra_ctrl_params())
        gate_params = list(self.gate.parameters())
        optimizer = torch.optim.Adam([
            {"params": ctrl_params, "lr": cfg.lr},
            {"params": gate_params, "lr": cfg.gate_lr},
        ])
        T = cfg.steer_steps
        step_w = make_step_weights(T, self.device)

        self.model.eval()
        print(
            f"\n[{cfg.name}] Stage 2 ({cfg.stage2_epochs} epochs, soft gate, +traj, "
            f"train_steps={self.resolve_train_steps(stage=2)})"
        )
        for epoch in range(1, cfg.stage2_epochs + 1):
            t0 = time.time()
            total_loss = 0.0
            self.apply_stage2_gate_warmup(epoch)

            # Trajectory weight ramps up over traj_warmup_epochs
            traj_scale = cfg.traj_weight * min(1.0, epoch / max(cfg.traj_warmup_epochs, 1))

            # ── FR examples ──
            for start in range(0, len(self.fr_examples), cfg.trace_batch_size):
                chunk = self.fr_examples[start : start + cfg.trace_batch_size]
                B = len(chunk)
                boundary_batch = torch.stack(
                    [ex.boundary_states.to(self.device) for ex in chunk], dim=0
                )  # [B, L, H]
                gate_logits, gate_probs = self.gate(boundary_batch)  # [B], [B]

                ctrl_batch = torch.stack([
                    self.get_controller_vecs(ex.boundary_states.to(self.device))
                    for ex in chunk
                ], dim=0)

                train_l, train_s = trace_forced_trajectory_batch_optimized(
                    model=self.model, paver=self.paver,
                    base_ids_batch=[ex.base_ids for ex in chunk],
                    anchor_ids_batch=self.slice_anchor_batch(chunk, stage=2),
                    controller_vectors=ctrl_batch,
                    gate_strengths=gate_probs,
                )

                loss = train_l.new_zeros(())
                for i, ex in enumerate(chunk):
                    loss = loss + self.compute_fr_loss(
                        ex, train_l[i], train_s[i], step_w, traj_scale
                    )
                # Gate BCE: FR should fire (target=1)
                loss = loss + cfg.gate_bce_weight * F.binary_cross_entropy_with_logits(
                    gate_logits, torch.ones_like(gate_logits)
                )
                loss = loss / B

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ctrl_params + gate_params, 1.0)
                optimizer.step()
                total_loss += float(loss.detach())

            # ── EN (keep benign) examples ──
            for start in range(0, len(self.keep_benign), cfg.trace_batch_size):
                chunk = self.keep_benign[start : start + cfg.trace_batch_size]
                B = len(chunk)
                boundary_batch = torch.stack(
                    [ex.boundary_states.to(self.device) for ex in chunk], dim=0
                )
                gate_logits_en, gate_probs_en = self.gate(boundary_batch)
                fr_logits_sample = gate_logits_en.new_tensor(
                    [self.gate(self.fr_examples[j % len(self.fr_examples)].boundary_states
                               .unsqueeze(0).to(self.device))[0].item()
                     for j in range(B)]
                )

                ctrl_batch = torch.stack([
                    self.get_controller_vecs(ex.boundary_states.to(self.device))
                    for ex in chunk
                ], dim=0)

                train_l, _ = trace_forced_trajectory_batch_optimized(
                    model=self.model, paver=self.paver,
                    base_ids_batch=[ex.base_ids for ex in chunk],
                    anchor_ids_batch=self.slice_anchor_batch(chunk, stage=2),
                    controller_vectors=ctrl_batch,
                    gate_strengths=gate_probs_en,
                )

                loss = train_l.new_zeros(())
                for i, ex in enumerate(chunk):
                    loss = loss + self.compute_keep_loss(ex, train_l[i])
                # Gate BCE: EN should not fire (target=0)
                loss = loss + cfg.gate_bce_weight * F.binary_cross_entropy_with_logits(
                    gate_logits_en, torch.zeros_like(gate_logits_en)
                )
                # Margin: FR gate score > EN gate score + margin
                loss = loss + gate_pair_margin(fr_logits_sample, gate_logits_en, cfg.gate_margin)
                loss = loss / B

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ctrl_params + gate_params, 1.0)
                optimizer.step()
                total_loss += float(loss.detach())

            # Regulariser
            reg = paving_regularizer(self.raw_vectors, cfg.l2_weight, cfg.layer_smooth, cfg.time_smooth)
            optimizer.zero_grad()
            reg.backward()
            optimizer.step()

            elapsed = time.time() - t0
            val_metrics = self.compute_validation_metrics()
            self.maybe_save_best_checkpoint(val_metrics)
            val_summary = ""
            if val_metrics:
                val_summary = (
                    f"  val_ce={val_metrics['val_ce']:.4f}"
                    f"  val_kl={val_metrics['val_kl']:.4f}"
                    f"  val_gate_acc={val_metrics['val_gate_acc']:.3f}"
                )
            print(
                f"  s2 epoch {epoch:02d}/{cfg.stage2_epochs}  loss={total_loss:.4f}"
                f"  traj_scale={traj_scale:.3f}{val_summary}  t={elapsed:.1f}s"
            )
            self.post_epoch_hook(stage=2, epoch=epoch)

        self.model.eval()

    def stage3_calibrate(self) -> None:
        """Stage 3: freeze controller, calibrate gate with utility-weighted BCE."""
        cfg = self.cfg
        # Freeze controller
        self.raw_vectors.requires_grad_(False)
        for p in self._extra_ctrl_params():
            p.requires_grad_(False)
        gate_optimizer = torch.optim.Adam(self.gate.parameters(), lr=cfg.gate_lr * 0.3)

        print(f"\n[{cfg.name}] Stage 3 ({cfg.stage3_epochs} epochs, gate calibration)")
        for epoch in range(1, cfg.stage3_epochs + 1):
            t0 = time.time()
            total_loss = 0.0

            all_examples = self.fr_examples + self.keep_benign
            for ex in all_examples:
                target = torch.tensor([ex.gate_target], device=self.device)
                boundary = ex.boundary_states.unsqueeze(0).to(self.device)
                gate_logit, _ = self.gate(boundary)
                loss = F.binary_cross_entropy_with_logits(gate_logit, target)
                gate_optimizer.zero_grad()
                loss.backward()
                gate_optimizer.step()
                total_loss += float(loss.detach())

            elapsed = time.time() - t0
            print(f"  s3 epoch {epoch:02d}/{cfg.stage3_epochs}  loss={total_loss:.4f}  t={elapsed:.1f}s")

        # Fold threshold into gate bias
        with torch.no_grad():
            fr_scores = torch.stack([
                self.gate(ex.boundary_states.unsqueeze(0).to(self.device))[0]
                for ex in self.fr_examples
            ]).squeeze(-1)
            en_scores = torch.stack([
                self.gate(ex.boundary_states.unsqueeze(0).to(self.device))[0]
                for ex in self.keep_benign
            ]).squeeze(-1)
        threshold, acc = choose_weighted_threshold(fr_scores, en_scores)
        print(f"[{cfg.name}] gate threshold={threshold:.4f} acc={acc:.3f}")
        self.gate_threshold = float(threshold)

        # Unfreeze controller for potential future use
        self.raw_vectors.requires_grad_(True)

    def _extra_ctrl_params(self):
        """Yield any extra parameters added by subclasses (e.g. state-cond MLP)."""
        return []

    # ── Save / evaluate ───────────────────────────────────────────────────────

    def save(self) -> None:
        path = self.run_dir / "checkpoint.pt"
        self.save_checkpoint(path)
        print(f"[{self.cfg.name}] saved checkpoint → {path}")

    # ── Full run ──────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        t_start = time.time()
        self.setup()
        self.stage1_train()
        self.stage2_train()
        self.stage3_calibrate()
        self.save()

        from eval import evaluate
        results = evaluate(self)
        results["total_time_s"] = time.time() - t_start

        import json
        results_path = self.run_dir / "results.json"
        results_path.write_text(json.dumps(results, indent=2))
        print(f"\n[{self.cfg.name}] DONE  refusal_rate={results.get('edit_refusal_rate', '?'):.3f}"
              f"  preservation={results.get('keep_preservation', '?'):.3f}"
              f"  gate_acc={results.get('gate_accuracy', '?'):.3f}"
              f"  time={results['total_time_s']:.0f}s")
        return results
