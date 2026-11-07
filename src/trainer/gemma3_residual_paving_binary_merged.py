#!/usr/bin/env python3
"""
gemma3_residual_paving_binary_merged.py

Merged binary residual-paving trainer for the French -> English steering task
on Gemma 3 text models.

Why this variant exists
-----------------------
The earlier binary-refined trainer produced an excellent gate, but the steering
controller regressed because the trajectory proxy became too dominant. An
earlier curriculum trainer did flip French prompts into English, but its gate
was not selective enough.

This script merges the strongest parts of both:

1. Controller recipe from the successful curriculum-parallel-teacher setup
   - strong local token objectives on French prompts
   - hard-gated stage 1 for learning the residual corridor first
   - parallel English prompt used as the positive teacher

2. Binary gate calibration from the refined trainer
   - anchored binary gate for French vs English
   - stage 3 utility-weighted gate-only calibration
   - weighted threshold folding into the final bias

3. Residual-paving objective kept as a *small, front-loaded auxiliary*
   - full-support probability matching with no top-k support transitions
   - centered-logit alignment instead of sparse logit support matching
   - direct residual-state corridor shaping across paved layers
   - baseline-relative close term and English-vs-French separation term
   - trajectory weight ramps up only after the local objectives start producing
     English behavior

The intent is to stay faithful to the "residual paving" idea without letting a
representation-space proxy overwhelm the output-aligned objectives.

Dependencies
------------
This script expects these files to exist in the same directory:
- gemma3_residual_paving_hybrid.py
- gemma3_residual_paving_binary_refined.py

Install:
    pip install -U torch transformers accelerate sentencepiece

Example:
    python gemma3_residual_paving_binary_merged.py train \
      --save_path gemma3_fr_en_binary_merged.pt \
      --preview_hard_gate

    python gemma3_residual_paving_binary_merged.py infer \
      --load_path gemma3_fr_en_binary_merged.pt \
      --prompt "Explique la récursion simplement." \
      --hard_gate
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from gemma3_residual_paving_hybrid import (
    DEFAULT_MODEL_ID,
    PARALLEL_LANGUAGE_PROMPTS,
    HybridLossConfig,
    ProjectionConfig,
    ProjectedTrajectoryScorer,
    ResidualPaver,
    average_dicts,
    batched,
    capture_boundary_states,
    encode_chat_prompt,
    finalize_paving_vectors,
    generate_ids,
    get_decoder_layers,
    get_hidden_size,
    load_model_and_tokenizer,
    parse_layer_indices,
    paving_regularizer,
    preservation_surrogate,
    relative_improvement_surrogate,
    set_seed,
    trace_forced_trajectory,
)
from gemma3_residual_paving_binary_refined import (
    AnchoredBinaryGate,
    BinaryTrainState,
    binary_gate_stats,
    build_binary_examples,
    choose_weighted_threshold,
    compute_gate_utility_weights,
    decode_with_binary_gate,
    english_fraction,
    gather_train_traces_with_gate,
    gather_train_traces_with_strengths,
    gate_pair_margin_loss,
    gate_saturation_penalty,
    load_binary_state,
    collect_en_references,
    collect_fr_references,
    build_prompt_variant_examples,
    make_piecewise_step_weights,
    make_step_weights,
    preview_teacher_pairs,
    save_binary_state,
    step_weighted_ce,
    step_weighted_kl,
    weighted_mean,
    weighted_token_margin_loss,
    init_binary_gate_from_examples,
)
from harmbench_dataset import (
    ALPACA_DEFAULT_SOURCE,
    build_harmbench_alpaca_prompt_records,
    build_harmbench_two_questions_prompt_records,
    build_saladbench_alpaca_prompt_records,
    build_saladbench_two_questions_prompt_records,
    default_harmbench_source,
    load_alpaca_rows,
    load_harmbench_rows,
    load_saladbench_rows,
    load_or_build_harmbench_parallel_records,
    filter_harmbench_rows,
    split_alpaca_rows,
)

DEFAULT_PROMPT_DATA_PATH = Path(__file__).resolve().parent / "data" / "french_parallel_prompts.txt"
DEFAULT_HARMBENCH_CACHE_DIR = Path(__file__).resolve().parent / "data" / "harmbench_parallel_cache"
DEFAULT_NUM_CONTROLLER_EXPERTS = 4
DEFAULT_HARMBENCH_NUM_CONTROLLER_EXPERTS = 5
DEFAULT_SALADBENCH_NUM_CONTROLLER_EXPERTS = 6
DEFAULT_ROUTE_EXPERT_NAMES = ("chat", "explanation", "instruction", "refusal")
HARMBENCH_ROUTE_BUCKETS: Tuple[str, ...] = (
    "cybercrime_intrusion",
    "chemical_biological",
    "illegal",
    "misinformation_disinformation",
    "harmful",
)
SALADBENCH_ROUTE_BUCKETS: Tuple[str, ...] = (
    "o1_representation_toxicity",
    "o2_misinformation_harms",
    "o3_socioeconomic_harms",
    "o4_information_safety",
    "o5_malicious_use",
    "o6_human_autonomy_integrity",
)
DEFAULT_BUCKET_TO_EXPERT_NAME = {
    "greeting_social_chat": "chat",
    "short_factual_qa": "explanation",
    "long_explanation": "explanation",
    "instruction_following": "instruction",
    "safety_refusal": "refusal",
    "code_help": "instruction",
    "opinion_open_ended": "chat",
    "summarization_rewrite": "explanation",
    "list_generation": "instruction",
    "ambiguous_slang_typo": "chat",
}
DEFAULT_METRIC_WINDOWS: Tuple[Tuple[int, int], ...] = ((1, 8), (9, 24), (25, 40))


def _is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).strip().lower()
    return "out of memory" in message and "cuda" in message


def truncate_token_ids(token_ids: torch.Tensor, max_steps: Optional[int]) -> torch.Tensor:
    flat = token_ids.view(-1)
    if max_steps is None:
        return flat
    resolved = int(max_steps)
    if resolved <= 0:
        return flat
    return flat[:resolved]


def resolve_stage_train_steps(args, *, stage: int, for_eval: bool) -> int:
    full_steps = max(1, int(args.steer_steps))
    if for_eval:
        return full_steps
    if int(stage) == 1:
        requested = int(getattr(args, "stage1_train_steps", 0))
    elif int(stage) == 2:
        requested = int(getattr(args, "stage2_train_steps", 0))
    else:
        raise ValueError(f"Unsupported training stage: {stage}")
    if requested <= 0:
        return full_steps
    return min(full_steps, requested)


def flatten_controller_traces(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim < 2:
        raise ValueError(f"Expected controller traces with at least 2 dims, got {tuple(vectors.shape)}")
    return vectors.reshape(vectors.shape[0], -1)


def resolve_num_controller_experts(dataset_track: str, requested_num_experts: int) -> int:
    track = str(dataset_track).strip().lower()
    requested = max(int(requested_num_experts), 1)
    if track in {"harmbench", "harmbench_parallel", "harmbench_2questions"} and requested == DEFAULT_NUM_CONTROLLER_EXPERTS:
        return DEFAULT_HARMBENCH_NUM_CONTROLLER_EXPERTS
    if track in {"saladbench", "saladbench_2questions"} and requested == DEFAULT_NUM_CONTROLLER_EXPERTS:
        return DEFAULT_SALADBENCH_NUM_CONTROLLER_EXPERTS
    return requested


def resolve_stage1_route_init(dataset_track: str, requested_init: str) -> str:
    track = str(dataset_track).strip().lower()
    init_key = str(requested_init or "auto").strip().lower() or "auto"
    if init_key == "auto":
        if track in {"saladbench", "saladbench_2questions"}:
            return "boundary_kmeans"
        return "bucket_prior"
    if init_key not in {"bucket_prior", "uniform", "boundary_kmeans"}:
        raise ValueError(f"Unsupported stage1_route_init: {requested_init!r}")
    return init_key


def compute_language_direction_basis(
    diff_traces: torch.Tensor,
    *,
    rank: int,
) -> Dict[str, torch.Tensor]:
    if diff_traces.ndim != 4:
        raise ValueError(f"Expected [N, T, L, H] diff_traces, got {tuple(diff_traces.shape)}")
    if diff_traces.shape[0] < 1:
        raise ValueError("At least one trace is required to build a basis.")

    flat = flatten_controller_traces(diff_traces.float())
    mean_flat = flat.mean(dim=0, keepdim=True)
    centered = flat - mean_flat
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    max_rank = min(int(rank), vh.shape[0])
    basis_flat = vh[:max_rank]
    total_energy = singular_values.pow(2).sum().clamp_min(1e-6)
    explained = singular_values[:max_rank].pow(2) / total_energy
    trace_shape = diff_traces.shape[1:]
    return {
        "basis_vectors": basis_flat.reshape(max_rank, *trace_shape).contiguous(),
        "mean_direction": mean_flat.reshape(*trace_shape).contiguous(),
        "explained_energy": explained.contiguous(),
        "singular_values": singular_values[:max_rank].contiguous(),
    }


def compute_boundary_mean_direction_from_examples(
    fr_examples: Sequence,
    en_examples: Sequence,
) -> torch.Tensor:
    if len(fr_examples) <= 0:
        raise ValueError("Expected at least one FR example for controller warm start.")
    if len(en_examples) <= 0:
        raise ValueError("Expected at least one EN example for controller warm start.")
    fr_mean = torch.stack([ex.boundary_states.float() for ex in fr_examples], dim=0).mean(dim=0)
    en_mean = torch.stack([ex.boundary_states.float() for ex in en_examples], dim=0).mean(dim=0)
    return fr_mean - en_mean


def build_boundary_warm_start_vectors(
    mean_direction: torch.Tensor,
    *,
    steer_steps: int,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mean_direction.ndim != 2:
        raise ValueError(f"Expected mean_direction with shape [L, H], got {tuple(mean_direction.shape)}")
    return mean_direction.to(device=device, dtype=dtype).unsqueeze(0).expand(int(steer_steps), -1, -1) * float(scale)


def maybe_warm_start_free_controller(
    controller_bank: "PavingControllerBank",
    *,
    fr_examples: Sequence,
    en_examples: Sequence,
    init_mode: str,
    init_scale: float,
) -> bool:
    mode = str(init_mode or "none").strip().lower()
    if mode in {"", "none", "off", "zero"}:
        return False
    if controller_bank.controller_mode != "free" or controller_bank.expert_residual_vectors is None:
        return False
    if mode != "mean_boundary":
        raise ValueError(f"Unsupported free_controller_init: {init_mode!r}")

    mean_direction = compute_boundary_mean_direction_from_examples(fr_examples, en_examples)
    init = build_boundary_warm_start_vectors(
        mean_direction,
        steer_steps=controller_bank.steer_steps,
        scale=float(init_scale),
        device=controller_bank.expert_residual_vectors.device,
        dtype=controller_bank.expert_residual_vectors.dtype,
    )
    expanded = init.unsqueeze(0).expand(controller_bank.num_experts, -1, -1, -1)
    with torch.no_grad():
        controller_bank.expert_residual_vectors.copy_(expanded)
    print(
        "[controller warm start] "
        f"mode=mean_boundary scale={float(init_scale):.3f} "
        f"mean_norm={float(mean_direction.norm().item()):.4f}"
    )
    return True


def maybe_warm_start_gate(
    gate: AnchoredBinaryGate,
    *,
    fr_examples: Sequence,
    en_examples: Sequence,
    steps: int,
    lr: float,
) -> bool:
    if getattr(gate, "gate_type", "anchored_linear") != "anchored_mlp":
        return False
    if int(steps) <= 0 or not fr_examples or not en_examples:
        return False
    boundary = torch.stack(
        [ex.boundary_states.float() for ex in list(fr_examples) + list(en_examples)],
        dim=0,
    ).to(device=gate.bias.device)
    labels = torch.cat(
        [
            torch.ones(len(fr_examples), device=gate.bias.device, dtype=torch.float32),
            torch.zeros(len(en_examples), device=gate.bias.device, dtype=torch.float32),
        ],
        dim=0,
    )
    opt = torch.optim.Adam(gate.parameters(), lr=float(lr))
    final_loss = 0.0
    for _ in range(max(int(steps), 0)):
        opt.zero_grad(set_to_none=True)
        logits, _ = gate(boundary)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())
    with torch.no_grad():
        logits, probs = gate(boundary)
        acc = (((logits > 0.0).float() == labels).float().mean().item()) if logits.numel() > 0 else 0.0
        print(
            "[gate warm start] "
            f"type={gate.gate_type} steps={int(steps)} lr={float(lr):.4f} "
            f"loss={final_loss:.4f} acc={acc:.3f} mean_prob={float(probs.mean().item()):.3f}"
        )
    return True


def route_bucket_index(bucket: str, num_experts: int) -> int:
    n = max(int(num_experts), 1)
    if n == 1:
        return 0
    bucket_key = (bucket or "general").strip().lower()
    if bucket_key in HARMBENCH_ROUTE_BUCKETS:
        if n >= len(HARMBENCH_ROUTE_BUCKETS):
            return HARMBENCH_ROUTE_BUCKETS.index(bucket_key)
        if n == 4:
            grouped = {
                "cybercrime_intrusion": 0,
                "chemical_biological": 1,
                "illegal": 2,
                "harmful": 2,
                "misinformation_disinformation": 3,
            }
            return grouped.get(bucket_key, 0)
        checksum = HARMBENCH_ROUTE_BUCKETS.index(bucket_key)
        return checksum % n
    if bucket_key in SALADBENCH_ROUTE_BUCKETS:
        checksum = SALADBENCH_ROUTE_BUCKETS.index(bucket_key)
        return checksum % n
    expert_name = DEFAULT_BUCKET_TO_EXPERT_NAME.get(bucket_key, DEFAULT_ROUTE_EXPERT_NAMES[0])
    if expert_name in DEFAULT_ROUTE_EXPERT_NAMES:
        idx = DEFAULT_ROUTE_EXPERT_NAMES.index(expert_name)
        if idx < n:
            return idx
    checksum = sum(ord(ch) for ch in bucket_key)
    return checksum % n


def fixed_bucket_route_weights(
    examples: Sequence,
    *,
    num_experts: int,
    device: torch.device,
    smoothing: float,
) -> torch.Tensor:
    n = max(int(num_experts), 1)
    smooth = float(max(min(smoothing, 0.25), 0.0))
    base = torch.full((len(examples), n), smooth / max(n - 1, 1), device=device, dtype=torch.float32)
    if n == 1:
        base.fill_(1.0)
        return base
    for row, ex in enumerate(examples):
        idx = route_bucket_index(example_bucket(ex), n)
        base[row, idx] = 1.0 - smooth
    return base / base.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class BoundaryExpertRouter(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        hidden_size: int,
        num_experts: int,
        hidden_dim: int,
        temperature: float,
        topk: int,
    ):
        super().__init__()
        self.num_experts = max(int(num_experts), 1)
        self.temperature = float(max(temperature, 1e-6))
        self.topk = int(max(topk, 0))
        if self.num_experts == 1:
            self.net = None
            return
        input_dim = int(hidden_size)
        if int(hidden_dim) > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, int(hidden_dim)),
                nn.Tanh(),
                nn.Linear(int(hidden_dim), self.num_experts),
            )
        else:
            self.net = nn.Linear(input_dim, self.num_experts)

    def forward(self, boundary_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = boundary_states.shape[0]
        if self.num_experts == 1:
            logits = boundary_states.new_zeros((batch_size, 1), dtype=torch.float32)
            weights = boundary_states.new_ones((batch_size, 1), dtype=torch.float32)
            return logits, weights
        pooled = boundary_states.float().mean(dim=1)
        logits = self.net(pooled)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        if 0 < self.topk < self.num_experts:
            topk_values, topk_idx = torch.topk(weights, k=self.topk, dim=-1)
            masked = torch.zeros_like(weights)
            masked.scatter_(1, topk_idx, topk_values)
            weights = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return logits, weights


class PavingControllerBank(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        steer_steps: int,
        num_layers: int,
        hidden_size: int,
        controller_mode: str,
        basis_bundle: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.num_experts = max(int(num_experts), 1)
        self.steer_steps = int(steer_steps)
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.controller_mode = str(controller_mode).strip().lower()
        if self.controller_mode not in {"free", "basis", "basis_plus_residual"}:
            raise ValueError(f"Unsupported controller_mode: {controller_mode}")

        self.register_buffer("basis_vectors", None, persistent=False)
        self.register_buffer("mean_direction", None, persistent=False)
        self.expert_basis_coeffs: Optional[nn.Parameter]
        self.expert_residual_vectors: Optional[nn.Parameter]

        if self.controller_mode == "free":
            self.expert_basis_coeffs = None
            self.expert_residual_vectors = nn.Parameter(
                torch.zeros(self.num_experts, self.steer_steps, self.num_layers, self.hidden_size, dtype=torch.float32)
            )
            self.basis_rank = 0
        else:
            if basis_bundle is None:
                raise ValueError("basis_bundle is required for basis controller modes.")
            basis_vectors = basis_bundle["basis_vectors"].float()
            mean_direction = basis_bundle["mean_direction"].float()
            expected_shape = (self.steer_steps, self.num_layers, self.hidden_size)
            if tuple(mean_direction.shape) != expected_shape:
                raise ValueError(f"basis mean_direction shape mismatch: expected {expected_shape}, got {tuple(mean_direction.shape)}")
            if basis_vectors.ndim != 4 or tuple(basis_vectors.shape[1:]) != expected_shape:
                raise ValueError(
                    f"basis_vectors must have shape [R, {self.steer_steps}, {self.num_layers}, {self.hidden_size}], "
                    f"got {tuple(basis_vectors.shape)}"
                )
            self.register_buffer("basis_vectors", basis_vectors.contiguous())
            self.register_buffer("mean_direction", mean_direction.contiguous())
            self.basis_rank = int(basis_vectors.shape[0])
            self.expert_basis_coeffs = nn.Parameter(torch.zeros(self.num_experts, self.basis_rank, dtype=torch.float32))
            if self.controller_mode == "basis_plus_residual":
                self.expert_residual_vectors = nn.Parameter(
                    torch.zeros(self.num_experts, self.steer_steps, self.num_layers, self.hidden_size, dtype=torch.float32)
                )
            else:
                self.expert_residual_vectors = None

    def materialize_raw_expert_vectors(self) -> torch.Tensor:
        if self.controller_mode == "free":
            assert self.expert_residual_vectors is not None
            return self.expert_residual_vectors

        assert self.basis_vectors is not None and self.mean_direction is not None and self.expert_basis_coeffs is not None
        basis_flat = self.basis_vectors.reshape(self.basis_vectors.shape[0], -1)
        coeff_flat = self.expert_basis_coeffs @ basis_flat
        raw = coeff_flat.reshape(self.num_experts, self.steer_steps, self.num_layers, self.hidden_size)
        raw = raw + self.mean_direction.unsqueeze(0)
        if self.expert_residual_vectors is not None:
            raw = raw + self.expert_residual_vectors
        return raw

    def expert_vectors(
        self,
        *,
        layer_start: float,
        layer_end: float,
        time_decay_tau: float,
        smooth_layer: float,
        smooth_time: float,
    ) -> torch.Tensor:
        raw = self.materialize_raw_expert_vectors()
        return torch.stack(
            [
                finalize_paving_vectors(
                    raw[idx],
                    layer_start=layer_start,
                    layer_end=layer_end,
                    time_decay_tau=time_decay_tau,
                    smooth_layer=smooth_layer,
                    smooth_time=smooth_time,
                )
                for idx in range(self.num_experts)
            ],
            dim=0,
        )

    def controller_batch(
        self,
        route_weights: torch.Tensor,
        *,
        layer_start: float,
        layer_end: float,
        time_decay_tau: float,
        smooth_layer: float,
        smooth_time: float,
    ) -> torch.Tensor:
        experts = self.expert_vectors(
            layer_start=layer_start,
            layer_end=layer_end,
            time_decay_tau=time_decay_tau,
            smooth_layer=smooth_layer,
            smooth_time=smooth_time,
        )
        return torch.einsum("bk,ktlh->btlh", route_weights.float(), experts)

    def regularizer(
        self,
        *,
        l2: float,
        layer_smooth: float,
        time_smooth: float,
        coeff_l2: float,
    ) -> torch.Tensor:
        raw = self.materialize_raw_expert_vectors()
        reg = torch.stack(
            [paving_regularizer(raw[idx], l2=l2, layer_smooth=layer_smooth, time_smooth=time_smooth) for idx in range(raw.shape[0])],
            dim=0,
        ).mean()
        if self.expert_basis_coeffs is not None and coeff_l2 > 0:
            reg = reg + float(coeff_l2) * self.expert_basis_coeffs.pow(2).mean()
        return reg


def route_entropy(route_weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    weights = route_weights.float().clamp_min(float(eps))
    entropy = -(weights * weights.log()).sum(dim=-1)
    return entropy.mean()


def route_usage_metrics(route_weights: torch.Tensor) -> Dict[str, float]:
    weights = route_weights.float()
    mean_usage = weights.mean(dim=0)
    stats = {f"route_{idx:02d}": float(mean_usage[idx].item()) for idx in range(mean_usage.shape[0])}
    stats["route_entropy"] = float(route_entropy(weights).item())
    return stats


def route_balance_penalty(route_weights: torch.Tensor) -> torch.Tensor:
    weights = route_weights.float()
    target = weights.new_full((weights.shape[-1],), 1.0 / max(weights.shape[-1], 1))
    return (weights.mean(dim=0) - target).pow(2).mean()


def pooled_output_features(states: torch.Tensor) -> torch.Tensor:
    if states.ndim == 4:
        return states.float().mean(dim=(1, 2))
    if states.ndim == 3:
        return states.float().mean(dim=(0, 1))
    raise ValueError(f"Expected [B, T, L, H] or [T, L, H] states, got {tuple(states.shape)}")


def pooled_boundary_features(boundary_states: torch.Tensor) -> torch.Tensor:
    if boundary_states.ndim == 3:
        return boundary_states.float().mean(dim=1)
    if boundary_states.ndim == 2:
        return boundary_states.float().mean(dim=0)
    raise ValueError(f"Expected [B, L, H] or [L, H] boundary states, got {tuple(boundary_states.shape)}")


def uniform_route_weights(
    batch_size: int,
    *,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    n = max(int(num_experts), 1)
    return torch.full((int(batch_size), n), 1.0 / float(n), device=device, dtype=torch.float32)


def smoothed_one_hot_route_weights(
    indices: torch.Tensor,
    *,
    num_experts: int,
    device: torch.device,
    smoothing: float,
) -> torch.Tensor:
    n = max(int(num_experts), 1)
    smooth = float(max(min(smoothing, 0.25), 0.0))
    if n == 1:
        return torch.ones((indices.shape[0], 1), device=device, dtype=torch.float32)
    base = torch.full((indices.shape[0], n), smooth / max(n - 1, 1), device=device, dtype=torch.float32)
    base.scatter_(1, indices.view(-1, 1), 1.0 - smooth)
    return base / base.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def assign_boundary_centroid_routes(
    boundary_batch: torch.Tensor,
    *,
    centroids: torch.Tensor,
    num_experts: int,
    device: torch.device,
    smoothing: float,
) -> torch.Tensor:
    pooled = pooled_boundary_features(boundary_batch)
    pooled = F.normalize(pooled.float(), dim=-1, eps=1e-6)
    ref = F.normalize(centroids.to(device=device).float(), dim=-1, eps=1e-6)
    route_idx = (pooled @ ref.t()).argmax(dim=-1)
    return smoothed_one_hot_route_weights(route_idx, num_experts=num_experts, device=device, smoothing=smoothing)


def compute_boundary_kmeans_centroids(
    examples: Sequence,
    *,
    num_experts: int,
    seed: int,
    max_iters: int = 24,
) -> Optional[torch.Tensor]:
    n = max(int(num_experts), 1)
    if not examples or n <= 1:
        return None

    features = torch.stack(
        [pooled_boundary_features(ex.boundary_states).detach().float().cpu() for ex in examples],
        dim=0,
    )
    features = F.normalize(features, dim=-1, eps=1e-6)
    if features.shape[0] == 0:
        return None

    rng = random.Random(int(seed))
    base_indices = list(range(features.shape[0]))
    rng.shuffle(base_indices)
    if not base_indices:
        return None
    chosen = [base_indices[idx % len(base_indices)] for idx in range(n)]
    centroids = features[chosen].clone()

    for _ in range(max(int(max_iters), 1)):
        similarities = features @ centroids.t()
        assignments = similarities.argmax(dim=-1)
        updated = centroids.clone()
        for idx in range(n):
            mask = assignments == idx
            if mask.any():
                updated[idx] = features[mask].mean(dim=0)
            else:
                updated[idx] = features[chosen[idx % len(chosen)]]
        updated = F.normalize(updated, dim=-1, eps=1e-6)
        if torch.allclose(updated, centroids, atol=1e-4, rtol=1e-4):
            centroids = updated
            break
        centroids = updated
    return centroids


def build_centroid_bank(features: torch.Tensor, labels: Sequence[str]) -> Optional[Dict[str, Any]]:
    if features.ndim != 2:
        raise ValueError(f"Expected [N, H] features, got {tuple(features.shape)}")
    if features.shape[0] != len(labels):
        raise ValueError("Feature rows must match label count.")
    unique_labels = sorted({str(label) for label in labels})
    if not unique_labels:
        return None
    centroids = []
    for label in unique_labels:
        mask = [idx for idx, item in enumerate(labels) if str(item) == label]
        label_feats = features[mask]
        centroid = F.normalize(label_feats.mean(dim=0), dim=-1, eps=1e-6)
        centroids.append(centroid)
    return {"labels": unique_labels, "centroids": torch.stack(centroids, dim=0)}


def build_semantic_anchor_bank(examples: Sequence) -> Dict[str, Optional[Dict[str, Any]]]:
    if not examples:
        return {"answer_mode": None, "length_bucket": None, "refusal": None}
    features = torch.stack([pooled_boundary_features(ex.boundary_states) for ex in examples], dim=0)
    answer_mode_labels = [str(getattr(ex, "answer_mode", example_bucket(ex))) for ex in examples]
    length_labels = [str(getattr(ex, "length_bucket", "medium")) for ex in examples]
    refusal_labels = ["refusal" if float(getattr(ex, "refusal_target", 0.0)) > 0.5 else "non_refusal" for ex in examples]
    return {
        "answer_mode": build_centroid_bank(features, answer_mode_labels),
        "length_bucket": build_centroid_bank(features, length_labels),
        "refusal": build_centroid_bank(features, refusal_labels),
    }


def centroid_classification_loss(
    features: torch.Tensor,
    targets: Sequence[str],
    bank: Optional[Dict[str, Any]],
    *,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    if bank is None or len(bank["labels"]) <= 1:
        return features.new_zeros(())
    norm_features = F.normalize(features.float(), dim=-1, eps=float(eps))
    centroids = F.normalize(bank["centroids"].detach().float(), dim=-1, eps=float(eps))
    logits = norm_features @ centroids.t()
    logits = logits / max(float(temperature), float(eps))
    label_to_index = {label: idx for idx, label in enumerate(bank["labels"])}
    known_pairs = [
        (row_idx, label_to_index[str(item)])
        for row_idx, item in enumerate(targets)
        if str(item) in label_to_index
    ]
    if not known_pairs:
        return features.new_zeros(())
    row_index = torch.tensor([row_idx for row_idx, _ in known_pairs], device=features.device, dtype=torch.long)
    target_idx = torch.tensor([label_idx for _, label_idx in known_pairs], device=features.device, dtype=torch.long)
    return F.cross_entropy(logits.index_select(0, row_index), target_idx)


def semantic_anchor_terms(
    *,
    train_states: torch.Tensor,
    ref_states: torch.Tensor,
    examples: Sequence,
    bank: Dict[str, Optional[Dict[str, Any]]],
    temperature: float,
    eps: float,
) -> Dict[str, torch.Tensor]:
    train_features = pooled_output_features(train_states)
    ref_features = pooled_output_features(ref_states)
    content = 1.0 - F.cosine_similarity(train_features, ref_features.detach(), dim=-1, eps=float(eps))
    answer_mode_targets = [str(getattr(ex, "answer_mode", example_bucket(ex))) for ex in examples]
    length_targets = [str(getattr(ex, "length_bucket", "medium")) for ex in examples]
    refusal_targets = ["refusal" if float(getattr(ex, "refusal_target", 0.0)) > 0.5 else "non_refusal" for ex in examples]
    return {
        "content": content.mean(),
        "mode": centroid_classification_loss(train_features, answer_mode_targets, bank.get("answer_mode"), temperature=temperature, eps=eps),
        "length": centroid_classification_loss(train_features, length_targets, bank.get("length_bucket"), temperature=temperature, eps=eps),
        "refusal": centroid_classification_loss(train_features, refusal_targets, bank.get("refusal"), temperature=temperature, eps=eps),
    }


def consistency_terms(
    *,
    base_logits: torch.Tensor,
    base_states: torch.Tensor,
    variant_logits: torch.Tensor,
    variant_states: torch.Tensor,
    max_steps: int,
    base_gate_logits: Optional[torch.Tensor] = None,
    variant_gate_logits: Optional[torch.Tensor] = None,
    base_gate_probs: Optional[torch.Tensor] = None,
    variant_gate_probs: Optional[torch.Tensor] = None,
    base_route_weights: Optional[torch.Tensor] = None,
    variant_route_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    k = min(int(max_steps), base_logits.shape[1], variant_logits.shape[1], base_states.shape[1], variant_states.shape[1])
    if k <= 0:
        zero = base_logits.new_zeros(())
        return {"logit": zero, "state": zero, "gate_raw": zero, "gate_prob": zero, "route": zero}
    logit_loss = F.mse_loss(base_logits[:, :k, :].float(), variant_logits[:, :k, :].float())
    base_pooled = pooled_output_features(base_states[:, :k])
    variant_pooled = pooled_output_features(variant_states[:, :k])
    state_loss = (1.0 - F.cosine_similarity(base_pooled, variant_pooled, dim=-1, eps=float(eps))).mean()
    zero = logit_loss.new_zeros(())
    gate_raw_loss = F.mse_loss(base_gate_logits, variant_gate_logits) if base_gate_logits is not None and variant_gate_logits is not None else zero
    gate_prob_loss = F.mse_loss(base_gate_probs, variant_gate_probs) if base_gate_probs is not None and variant_gate_probs is not None else zero
    route_loss = F.mse_loss(base_route_weights, variant_route_weights) if base_route_weights is not None and variant_route_weights is not None else zero
    return {
        "logit": logit_loss,
        "state": state_loss,
        "gate_raw": gate_raw_loss,
        "gate_prob": gate_prob_loss,
        "route": route_loss,
    }


def english_fraction_window_metrics(
    logits: torch.Tensor,
    english_ids: torch.Tensor,
    negative_id_sets: Sequence[torch.Tensor],
    windows: Sequence[Tuple[int, int]],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for start, end in windows:
        zero_start = max(int(start) - 1, 0)
        zero_end = max(int(end), zero_start)
        k = min(zero_end, logits.shape[1], english_ids.shape[1], *(neg.shape[1] for neg in negative_id_sets))
        if k <= zero_start:
            metrics[f"eng_minus_fr_{start}_{end}"] = 0.0
            continue
        pred = logits[:, zero_start:k, :].argmax(dim=-1)
        pos = (pred == english_ids[:, zero_start:k]).float().mean().item()
        neg = max((pred == neg_ids[:, zero_start:k]).float().mean().item() for neg_ids in negative_id_sets)
        metrics[f"eng_minus_fr_{start}_{end}"] = float(pos - neg)
    return metrics


def weighted_step_mean(values: torch.Tensor, step_weights: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return weighted_mean(values, step_weights)
    if values.ndim != 2:
        raise ValueError(f"weighted_step_mean expects [B, T] or [T], got shape={tuple(values.shape)}")
    k = min(values.shape[1], step_weights.shape[0])
    if k <= 0:
        return values.new_zeros(())
    w = step_weights[:k].view(1, k)
    return (values[:, :k] * w).sum() / (values.shape[0] * w.sum().clamp_min(1e-6))


def make_layer_weights(
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
    late_boost: float,
) -> torch.Tensor:
    n = max(int(num_layers), 1)
    idx = torch.linspace(0.0, 1.0, steps=n, device=device, dtype=dtype)
    weights = 1.0 + float(max(late_boost, 0.0)) * idx
    return weights / weights.sum().clamp_min(1e-6)


def hellinger_distance_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    p = torch.softmax(logits_a.float(), dim=-1).clamp_min(float(eps))
    q = torch.softmax(logits_b.detach().float(), dim=-1).clamp_min(float(eps))
    return torch.linalg.vector_norm(torch.sqrt(p) - torch.sqrt(q), ord=2, dim=-1) / math.sqrt(2.0)


def centered_logit_cosine_distance(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    a = logits_a.float()
    b = logits_b.detach().float()
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    return 1.0 - F.cosine_similarity(a, b, dim=-1, eps=float(eps))


def residual_state_cosine_distance(
    states_a: torch.Tensor,
    states_b: torch.Tensor,
    layer_weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    a = states_a.float()
    b = states_b.detach().float()
    cos = F.cosine_similarity(a, b, dim=-1, eps=float(eps))
    dist = 1.0 - cos
    lw = layer_weights.view(1, 1, -1)
    return (dist * lw).sum(dim=-1) / lw.sum().clamp_min(float(eps))


def pairwise_paving_distance(
    *,
    scorer: ProjectedTrajectoryScorer,
    train_logits: torch.Tensor,
    train_states: torch.Tensor,
    ref_logits: torch.Tensor,
    ref_states: torch.Tensor,
    layer_weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    d_prob = hellinger_distance_from_logits(train_logits, ref_logits, eps=eps)
    d_logit = centered_logit_cosine_distance(train_logits, ref_logits, eps=eps)
    d_state = residual_state_cosine_distance(train_states, ref_states, layer_weights=layer_weights, eps=eps)
    total = (
        float(scorer.loss_config.traj_prob_weight) * d_prob
        + float(scorer.loss_config.traj_logit_weight) * d_logit
        + float(scorer.loss_config.traj_state_weight) * d_state
    )
    norm = max(
        float(scorer.loss_config.traj_prob_weight)
        + float(scorer.loss_config.traj_logit_weight)
        + float(scorer.loss_config.traj_state_weight),
        float(eps),
    )
    return total / norm


def weighted_logprob_margin_loss(
    logits: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    weights: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    pos_lp = log_probs.gather(-1, pos_ids.unsqueeze(-1)).squeeze(-1)
    neg_lp = log_probs.gather(-1, neg_ids.unsqueeze(-1)).squeeze(-1)
    loss = F.softplus(float(margin) - (pos_lp - neg_lp))
    return weighted_step_mean(loss, weights)


def fr_aux_trajectory_terms(
    *,
    scorer: ProjectedTrajectoryScorer,
    train_logits: torch.Tensor,
    train_states: torch.Tensor,
    pos_logits: torch.Tensor,
    pos_states: torch.Tensor,
    base_logits: torch.Tensor,
    base_states: torch.Tensor,
    neg_logits: torch.Tensor,
    neg_states: torch.Tensor,
    hard_neg_logits: Optional[torch.Tensor],
    hard_neg_states: Optional[torch.Tensor],
    step_weights: torch.Tensor,
    layer_weights: torch.Tensor,
    target_fraction: float,
    floor: float,
    sep_margin: float,
    mono_tol: float,
    eps: float,
) -> Dict[str, torch.Tensor]:
    """Support-free residual-paving auxiliary.

    This keeps the spirit of the richer conditional loss while making it much
    easier to optimize:
    - full-support Hellinger distance on probabilities instead of top-k support
    - centered-logit cosine distance instead of sparse logit support matching
    - direct residual-state cosine distance across paved layers

    The result is a paired, per-sample, per-step corridor objective. It stays
    distributional, but it is local enough to provide useful gradients.
    """
    d_pos_train = pairwise_paving_distance(
        scorer=scorer,
        train_logits=train_logits,
        train_states=train_states,
        ref_logits=pos_logits,
        ref_states=pos_states,
        layer_weights=layer_weights,
        eps=eps,
    )
    d_pos_base = pairwise_paving_distance(
        scorer=scorer,
        train_logits=base_logits,
        train_states=base_states,
        ref_logits=pos_logits,
        ref_states=pos_states,
        layer_weights=layer_weights,
        eps=eps,
    )
    d_neg_train = pairwise_paving_distance(
        scorer=scorer,
        train_logits=train_logits,
        train_states=train_states,
        ref_logits=neg_logits,
        ref_states=neg_states,
        layer_weights=layer_weights,
        eps=eps,
    )
    if hard_neg_logits is not None and hard_neg_states is not None:
        d_hard_neg_train = pairwise_paving_distance(
            scorer=scorer,
            train_logits=train_logits,
            train_states=train_states,
            ref_logits=hard_neg_logits,
            ref_states=hard_neg_states,
            layer_weights=layer_weights,
            eps=eps,
        )
        d_neg_train = torch.minimum(d_neg_train, d_hard_neg_train)

    denom = torch.clamp(d_pos_base.detach(), min=float(floor))
    target = float(target_fraction) * denom
    close = F.softplus((d_pos_train - target) / (denom + float(eps)))
    gap = (d_neg_train - d_pos_train) / (denom + float(eps))
    sep = F.softplus(float(sep_margin) - gap)

    return {
        "close": weighted_step_mean(close, step_weights),
        "sep": weighted_step_mean(sep, step_weights),
        "mono": d_pos_train.new_zeros(()),
        "d_pos_train": d_pos_train.detach(),
        "d_pos_base": d_pos_base.detach(),
        "d_neg_train": d_neg_train.detach(),
    }


def en_aux_trajectory_terms(
    *,
    scorer: ProjectedTrajectoryScorer,
    train_logits: torch.Tensor,
    train_states: torch.Tensor,
    base_logits: torch.Tensor,
    base_states: torch.Tensor,
    step_weights: torch.Tensor,
    layer_weights: torch.Tensor,
    tolerance: float,
    scale: float,
    mono_tol: float,
    eps: float,
) -> Dict[str, torch.Tensor]:
    d_base = pairwise_paving_distance(
        scorer=scorer,
        train_logits=train_logits,
        train_states=train_states,
        ref_logits=base_logits,
        ref_states=base_states,
        layer_weights=layer_weights,
        eps=eps,
    )
    preserve = preservation_surrogate(d_base, tolerance=float(tolerance), scale=float(scale))
    return {
        "preserve": weighted_step_mean(preserve, step_weights),
        "mono": d_base.new_zeros(()),
        "d_base": d_base.detach(),
    }


@torch.no_grad()
def choose_traj_scale(
    *,
    epoch: int,
    total_epochs: int,
    max_scale: float,
    warmup_epochs: int,
    last_metric: float,
    unlock_metric: float,
    cold_scale: float,
) -> float:
    if total_epochs <= 0:
        return 0.0
    if epoch <= max(int(warmup_epochs), 0):
        progress = float(epoch) / float(max(int(warmup_epochs), 1))
        return float(max_scale) * progress * 0.5
    if float(last_metric) < float(unlock_metric):
        return float(cold_scale)
    progress = float(epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
    progress = max(0.0, min(1.0, progress))
    return float(max_scale) * progress


def load_controller_basis_bundle(
    basis_path: str,
    *,
    device: torch.device,
    steer_steps: int,
    num_layers: int,
    hidden_size: int,
) -> Dict[str, torch.Tensor]:
    obj = torch.load(basis_path, map_location=device)
    if not isinstance(obj, dict) or "basis_vectors" not in obj or "mean_direction" not in obj:
        raise ValueError(f"Basis file must contain 'basis_vectors' and 'mean_direction': {basis_path}")
    basis_vectors = obj["basis_vectors"].to(device).float()
    mean_direction = obj["mean_direction"].to(device).float()
    expected_shape = (int(steer_steps), int(num_layers), int(hidden_size))
    if tuple(mean_direction.shape) != expected_shape:
        raise ValueError(f"Basis mean_direction shape mismatch: expected {expected_shape}, got {tuple(mean_direction.shape)}")
    if basis_vectors.ndim != 4 or tuple(basis_vectors.shape[1:]) != expected_shape:
        raise ValueError(f"Basis vectors shape mismatch: expected [R, {expected_shape}], got {tuple(basis_vectors.shape)}")
    bundle = {
        "basis_vectors": basis_vectors,
        "mean_direction": mean_direction,
    }
    if "explained_energy" in obj and isinstance(obj["explained_energy"], torch.Tensor):
        bundle["explained_energy"] = obj["explained_energy"].to(device).float()
    if "singular_values" in obj and isinstance(obj["singular_values"], torch.Tensor):
        bundle["singular_values"] = obj["singular_values"].to(device).float()
    return bundle


def resolve_prompt_data_path(prompt_data_path: Optional[str]) -> Optional[Path]:
    if prompt_data_path:
        return Path(prompt_data_path).expanduser()
    if DEFAULT_PROMPT_DATA_PATH.exists():
        return DEFAULT_PROMPT_DATA_PATH
    return None


def load_binary_prompt_records(
    prompt_data_path: Optional[str],
    *,
    max_records: Optional[int],
) -> Tuple[List[Dict[str, str]], str]:
    if max_records is not None and int(max_records) <= 0:
        raise ValueError("--max_prompt_records must be positive when provided.")

    resolved_path = resolve_prompt_data_path(prompt_data_path)
    if resolved_path is None:
        prompt_records = [dict(item) for item in PARALLEL_LANGUAGE_PROMPTS]
        if max_records is not None:
            prompt_records = prompt_records[: int(max_records)]
        return prompt_records, "built-in PARALLEL_LANGUAGE_PROMPTS"

    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Prompt data file is missing a header row: {resolved_path}")
        if "fr" not in reader.fieldnames or "en" not in reader.fieldnames:
            raise ValueError(f"Prompt data file must include 'fr' and 'en' columns: {resolved_path}")

        prompt_records: List[Dict[str, str]] = []
        for line_no, row in enumerate(reader, start=2):
            record = {
                str(key).strip(): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            record = {key: value for key, value in record.items() if value}
            if not record:
                continue
            if not record.get("fr") or not record.get("en"):
                raise ValueError(f"Prompt record on line {line_no} must include non-empty 'fr' and 'en' values.")
            prompt_records.append(record)
            if max_records is not None and len(prompt_records) >= int(max_records):
                break

    if not prompt_records:
        raise ValueError(f"No prompt records were loaded from {resolved_path}")
    return prompt_records, str(resolved_path.resolve())


def resolve_harmbench_cache_path(cache_dir: Optional[str], split_name: str) -> Path:
    base_dir = Path(cache_dir).expanduser() if cache_dir else DEFAULT_HARMBENCH_CACHE_DIR
    return base_dir / f"harmbench_{split_name}_parallel.tsv"


@torch.no_grad()
def translate_prompt_to_french(
    model,
    tokenizer,
    device: torch.device,
    english_prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    translation_request = (
        "Translate the following user request into French. Preserve meaning, structure, and formatting. "
        "Output only the French translation.\n\n"
        f"{english_prompt}"
    )
    base_ids = encode_chat_prompt(
        tokenizer,
        translation_request,
        "You are a careful translator that outputs only French.",
        device,
    )
    translated_ids = generate_ids(
        model,
        tokenizer,
        base_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_on_eos=True,
    )
    translated = tokenizer.decode(
        translated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return translated


def load_dataset_track_prompt_records(
    *,
    args,
    model,
    tokenizer,
    device: torch.device,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, str], bool]:
    dataset_track = str(args.dataset_track).strip().lower()
    if dataset_track == "parallel_french":
        prompt_records, prompt_data_source = load_binary_prompt_records(
            args.prompt_data_path,
            max_records=args.max_prompt_records,
        )
        return prompt_records, [], {"train": prompt_data_source}, False

    if dataset_track == "harmbench":
        train_source = args.harmbench_train_source or default_harmbench_source(args.harmbench_train_split)
        test_source = args.harmbench_test_source or default_harmbench_source(args.harmbench_test_split)
        train_rows = filter_harmbench_rows(
            load_harmbench_rows(train_source),
            exclude_copyright=True,
            exclude_contextual=True,
        )
        test_rows = filter_harmbench_rows(
            load_harmbench_rows(test_source),
            exclude_copyright=True,
            exclude_contextual=True,
        )
        if args.max_prompt_records is not None:
            train_rows = train_rows[: int(args.max_prompt_records)]
        if args.harmbench_max_test_records is not None:
            test_rows = test_rows[: int(args.harmbench_max_test_records)]

        alpaca_rows = load_alpaca_rows(args.alpaca_source or ALPACA_DEFAULT_SOURCE)
        alpaca_train_rows, alpaca_test_rows = split_alpaca_rows(
            alpaca_rows,
            train_count=len(train_rows),
            test_count=len(test_rows),
            seed=args.alpaca_split_seed,
        )
        train_records = build_harmbench_alpaca_prompt_records(
            train_rows,
            alpaca_train_rows,
            bucket_field=args.harmbench_bucket_field,
        )
        test_records = build_harmbench_alpaca_prompt_records(
            test_rows,
            alpaca_test_rows,
            bucket_field=args.harmbench_bucket_field,
        )
        train_desc = f"harmbench:{train_source} + alpaca:{args.alpaca_source or ALPACA_DEFAULT_SOURCE}"
        test_desc = f"harmbench:{test_source} + alpaca_holdout:{args.alpaca_source or ALPACA_DEFAULT_SOURCE}"
        return train_records, test_records, {"train": train_desc, "test": test_desc}, True

    if dataset_track == "harmbench_2questions":
        train_source = args.harmbench_train_source or default_harmbench_source(args.harmbench_train_split)
        test_source = args.harmbench_test_source or default_harmbench_source(args.harmbench_test_split)
        train_rows = filter_harmbench_rows(
            load_harmbench_rows(train_source),
            exclude_copyright=True,
            exclude_contextual=True,
        )
        test_rows = filter_harmbench_rows(
            load_harmbench_rows(test_source),
            exclude_copyright=True,
            exclude_contextual=True,
        )
        if args.max_prompt_records is not None:
            train_rows = train_rows[: int(args.max_prompt_records)]
        if args.harmbench_max_test_records is not None:
            test_rows = test_rows[: int(args.harmbench_max_test_records)]

        alpaca_rows = load_alpaca_rows(args.alpaca_source or ALPACA_DEFAULT_SOURCE)
        alpaca_train_rows, alpaca_test_rows = split_alpaca_rows(
            alpaca_rows,
            train_count=len(train_rows),
            test_count=len(test_rows),
            seed=args.alpaca_split_seed,
        )
        train_records = build_harmbench_two_questions_prompt_records(
            train_rows,
            alpaca_train_rows,
            bucket_field=args.harmbench_bucket_field,
        )
        test_records = build_harmbench_two_questions_prompt_records(
            test_rows,
            alpaca_test_rows,
            bucket_field=args.harmbench_bucket_field,
        )
        train_desc = f"alpaca + harmbench_2questions:{train_source}"
        test_desc = f"alpaca_holdout + harmbench_2questions:{test_source}"
        return train_records, test_records, {"train": train_desc, "test": test_desc}, True

    if dataset_track == "saladbench":
        if args.saladbench_split_from_base:
            base_source = args.saladbench_base_source or args.saladbench_train_source
            base_rows = load_saladbench_rows(base_source, split="base")
            base_rows = stratified_prompt_record_subset(
                base_rows,
                sample_count=args.saladbench_base_total_records,
                seed=args.saladbench_base_split_seed,
            )
            train_rows, test_rows = split_prompt_records_stratified(
                base_rows,
                test_fraction=args.saladbench_base_test_fraction,
                seed=args.saladbench_base_split_seed,
            )
        else:
            train_rows = load_saladbench_rows(args.saladbench_train_source, split=args.saladbench_train_split)
            test_rows = load_saladbench_rows(args.saladbench_test_source, split=args.saladbench_test_split)
            if args.max_prompt_records is not None:
                train_rows = train_rows[: int(args.max_prompt_records)]
            if args.saladbench_max_test_records is not None:
                test_rows = test_rows[: int(args.saladbench_max_test_records)]

        alpaca_rows = load_alpaca_rows(args.alpaca_source or ALPACA_DEFAULT_SOURCE)
        alpaca_train_rows, alpaca_test_rows = split_alpaca_rows(
            alpaca_rows,
            train_count=len(train_rows),
            test_count=len(test_rows),
            seed=args.alpaca_split_seed,
        )
        train_records = build_saladbench_alpaca_prompt_records(
            train_rows,
            alpaca_train_rows,
            bucket_level=args.saladbench_bucket_level,
        )
        test_records = build_saladbench_alpaca_prompt_records(
            test_rows,
            alpaca_test_rows,
            bucket_level=args.saladbench_bucket_level,
        )
        if args.saladbench_split_from_base:
            base_source = args.saladbench_base_source or args.saladbench_train_source or "base"
            split_desc = (
                f"base[{len(train_rows) + len(test_rows)}]"
                f"/test_frac={float(args.saladbench_base_test_fraction):.3f}"
                f"/seed={int(args.saladbench_base_split_seed)}"
            )
            train_source = f"{base_source}:{split_desc}:train"
            test_source = f"{base_source}:{split_desc}:test"
        else:
            train_source = args.saladbench_train_source or f"{args.saladbench_train_split}"
            test_source = args.saladbench_test_source or f"{args.saladbench_test_split}"
        train_desc = f"saladbench:{train_source} + alpaca:{args.alpaca_source or ALPACA_DEFAULT_SOURCE}"
        test_desc = f"saladbench:{test_source} + alpaca_holdout:{args.alpaca_source or ALPACA_DEFAULT_SOURCE}"
        return train_records, test_records, {"train": train_desc, "test": test_desc}, True

    if dataset_track == "saladbench_2questions":
        if args.saladbench_split_from_base:
            base_source = args.saladbench_base_source or args.saladbench_train_source
            base_rows = load_saladbench_rows(base_source, split="base")
            base_rows = stratified_prompt_record_subset(
                base_rows,
                sample_count=args.saladbench_base_total_records,
                seed=args.saladbench_base_split_seed,
            )
            train_rows, test_rows = split_prompt_records_stratified(
                base_rows,
                test_fraction=args.saladbench_base_test_fraction,
                seed=args.saladbench_base_split_seed,
            )
        else:
            train_rows = load_saladbench_rows(args.saladbench_train_source, split=args.saladbench_train_split)
            test_rows = load_saladbench_rows(args.saladbench_test_source, split=args.saladbench_test_split)
            if args.max_prompt_records is not None:
                train_rows = train_rows[: int(args.max_prompt_records)]
            if args.saladbench_max_test_records is not None:
                test_rows = test_rows[: int(args.saladbench_max_test_records)]

        alpaca_rows = load_alpaca_rows(args.alpaca_source or ALPACA_DEFAULT_SOURCE)
        alpaca_train_rows, alpaca_test_rows = split_alpaca_rows(
            alpaca_rows,
            train_count=len(train_rows),
            test_count=len(test_rows),
            seed=args.alpaca_split_seed,
        )
        train_records = build_saladbench_two_questions_prompt_records(
            train_rows,
            alpaca_train_rows,
            bucket_level=args.saladbench_bucket_level,
        )
        test_records = build_saladbench_two_questions_prompt_records(
            test_rows,
            alpaca_test_rows,
            bucket_level=args.saladbench_bucket_level,
        )
        if args.saladbench_split_from_base:
            base_source = args.saladbench_base_source or args.saladbench_train_source or "base"
            split_desc = (
                f"base[{len(train_rows) + len(test_rows)}]"
                f"/test_frac={float(args.saladbench_base_test_fraction):.3f}"
                f"/seed={int(args.saladbench_base_split_seed)}"
            )
            train_source = f"{base_source}:{split_desc}:train"
            test_source = f"{base_source}:{split_desc}:test"
        else:
            train_source = args.saladbench_train_source or f"{args.saladbench_train_split}"
            test_source = args.saladbench_test_source or f"{args.saladbench_test_split}"
        train_desc = f"alpaca + saladbench_2questions:{train_source}"
        test_desc = f"alpaca_holdout + saladbench_2questions:{test_source}"
        return train_records, test_records, {"train": train_desc, "test": test_desc}, True

    if dataset_track == "harmbench_parallel":
        def translator(prompt_text: str) -> str:
            return translate_prompt_to_french(
                model,
                tokenizer,
                device,
                prompt_text,
                max_new_tokens=args.harmbench_translation_max_new_tokens,
                temperature=args.harmbench_translation_temperature,
                top_p=args.harmbench_translation_top_p,
            )

        train_source = args.harmbench_train_source or default_harmbench_source(args.harmbench_train_split)
        test_source = args.harmbench_test_source or default_harmbench_source(args.harmbench_test_split)
        train_cache = str(resolve_harmbench_cache_path(args.harmbench_cache_dir, "train"))
        test_cache = str(resolve_harmbench_cache_path(args.harmbench_cache_dir, "test"))

        train_records, train_desc = load_or_build_harmbench_parallel_records(
            source=train_source,
            cache_path=train_cache,
            translator=translator,
            bucket_field=args.harmbench_bucket_field,
            max_records=args.max_prompt_records,
            exclude_copyright=True,
            exclude_contextual=True,
            force_refresh=bool(args.harmbench_refresh_cache),
        )
        test_records, test_desc = load_or_build_harmbench_parallel_records(
            source=test_source,
            cache_path=test_cache,
            translator=translator,
            bucket_field=args.harmbench_bucket_field,
            max_records=args.harmbench_max_test_records,
            exclude_copyright=True,
            exclude_contextual=True,
            force_refresh=bool(args.harmbench_refresh_cache),
        )
        return train_records, test_records, {"train": train_desc, "test": test_desc}, True

    if dataset_track not in {"harmbench", "harmbench_parallel", "harmbench_2questions", "saladbench", "saladbench_2questions"}:
        raise ValueError(f"Unsupported dataset track: {args.dataset_track!r}")
    raise ValueError(f"Unsupported dataset track: {args.dataset_track!r}")


def format_bucket_counts(records: Sequence[Dict[str, str]]) -> str:
    counts: Dict[str, int] = {}
    for record in records:
        bucket = str(record.get("bucket", "general")).strip() or "general"
        counts[bucket] = counts.get(bucket, 0) + 1
    return " ".join(f"{bucket}={count}" for bucket, count in sorted(counts.items()))


def prompt_record_bucket(record: Mapping[str, Any]) -> str:
    return str(record.get("bucket", "general")).strip() or "general"


def prompt_record_stratify_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    bucket = prompt_record_bucket(record)
    source = str(record.get("source", "")).strip().lower() or "unknown"
    return bucket, source


def allocate_stratified_counts(group_sizes: Mapping[Any, int], sample_count: int) -> Dict[Any, int]:
    total_available = sum(max(int(size), 0) for size in group_sizes.values())
    target = max(0, min(int(sample_count), total_available))
    if target <= 0 or total_available <= 0:
        return {key: 0 for key in group_sizes}

    positive_keys = [key for key, size in group_sizes.items() if int(size) > 0]
    if target >= total_available:
        return {key: int(group_sizes[key]) for key in group_sizes}

    counts = {key: 0 for key in group_sizes}
    remaining = target
    if target >= len(positive_keys):
        for key in positive_keys:
            counts[key] = 1
        remaining -= len(positive_keys)

    if remaining <= 0:
        return counts

    fractional_parts: List[Tuple[float, int, Any]] = []
    for order, key in enumerate(sorted(positive_keys, key=str)):
        size = int(group_sizes[key])
        available = max(size - counts[key], 0)
        if available <= 0:
            continue
        expected = (size / float(total_available)) * target
        additional_expected = max(expected - counts[key], 0.0)
        base = min(int(math.floor(additional_expected)), available)
        counts[key] += base
        remaining -= base
        fractional_parts.append((additional_expected - base, order, key))

    if remaining <= 0:
        return counts

    fractional_parts.sort(key=lambda item: (-item[0], item[1]))
    for _, _, key in fractional_parts:
        if remaining <= 0:
            break
        size = int(group_sizes[key])
        if counts[key] >= size:
            continue
        counts[key] += 1
        remaining -= 1

    if remaining <= 0:
        return counts

    for key in sorted(positive_keys, key=str):
        if remaining <= 0:
            break
        size = int(group_sizes[key])
        available = size - counts[key]
        if available <= 0:
            continue
        extra = min(available, remaining)
        counts[key] += extra
        remaining -= extra

    return counts


def stratified_prompt_record_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    sample_count: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    items = [dict(record) for record in records]
    if sample_count is None or int(sample_count) >= len(items):
        return items
    target = int(sample_count)
    if target <= 0:
        raise ValueError("Requested stratified prompt-record subset must contain at least one record.")

    rng = random.Random(seed)
    group_to_items: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in items:
        group_to_items.setdefault(prompt_record_stratify_key(record), []).append(record)

    for group_items in group_to_items.values():
        rng.shuffle(group_items)

    group_sizes = {key: len(group_items) for key, group_items in group_to_items.items()}
    counts = allocate_stratified_counts(group_sizes, target)

    selected: List[Dict[str, Any]] = []
    for key in sorted(group_to_items.keys(), key=str):
        selected.extend(group_to_items[key][: counts.get(key, 0)])

    rng.shuffle(selected)
    return selected


def split_prompt_records_stratified(
    records: Sequence[Mapping[str, Any]],
    *,
    test_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items = [dict(record) for record in records]
    if not items:
        raise ValueError("At least one prompt record is required.")

    frac = float(test_fraction)
    if frac <= 0.0:
        return items, []
    if frac >= 1.0:
        raise ValueError("Stratified prompt-record test fraction must be less than 1.0.")
    if len(items) < 2:
        return items, []

    rng = random.Random(seed)
    group_to_indices: Dict[Tuple[str, str], List[int]] = {}
    for idx, record in enumerate(items):
        group_to_indices.setdefault(prompt_record_stratify_key(record), []).append(idx)

    test_index_set = set()
    for group_key in sorted(group_to_indices.keys(), key=str):
        group_indices = list(group_to_indices[group_key])
        if len(group_indices) < 2:
            continue
        rng.shuffle(group_indices)
        group_test_count = int(round(len(group_indices) * frac))
        group_test_count = max(1, min(group_test_count, len(group_indices) - 1))
        test_index_set.update(group_indices[:group_test_count])

    if not test_index_set:
        indices = list(range(len(items)))
        rng.shuffle(indices)
        test_index_set.add(indices[0])

    train_records = [dict(record) for idx, record in enumerate(items) if idx not in test_index_set]
    test_records = [dict(record) for idx, record in enumerate(items) if idx in test_index_set]
    return train_records, test_records


def example_bucket(example) -> str:
    bucket = getattr(example, "bucket", "general")
    return str(bucket).strip() or "general"


def make_example_batches(
    examples: Sequence,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    bucket_batches: bool,
) -> List[List]:
    items = list(examples)
    if not bucket_batches:
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(items)
        return batched(items, batch_size)

    grouped: Dict[str, List] = {}
    for ex in items:
        grouped.setdefault(example_bucket(ex), []).append(ex)

    rng = random.Random(seed)
    bucket_names = sorted(grouped.keys())
    bucket_batches_out: List[List] = []
    for bucket in bucket_names:
        bucket_items = list(grouped[bucket])
        if shuffle:
            rng.shuffle(bucket_items)
        bucket_batches_out.extend(batched(bucket_items, batch_size))

    if shuffle:
        rng.shuffle(bucket_batches_out)
    return bucket_batches_out


def _weighted_average_metric_records(weighted_records: Sequence[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    if not weighted_records:
        return {}
    total_weight = float(sum(max(1, int(weight)) for weight, _ in weighted_records))
    accum: Dict[str, float] = {}
    for weight, metrics in weighted_records:
        resolved_weight = float(max(1, int(weight)))
        for key, value in metrics.items():
            accum[key] = accum.get(key, 0.0) + resolved_weight * float(value)
    return {key: value / total_weight for key, value in accum.items()}


def _split_batch_for_oom_retry(batch: Sequence) -> List[List]:
    items = list(batch)
    if len(items) <= 1:
        return [items]
    midpoint = max(1, len(items) // 2)
    return [items[:midpoint], items[midpoint:]]


def _split_paired_batches_for_oom_retry(fr_batch: Sequence, en_batch: Sequence) -> List[Tuple[List, List]]:
    fr_parts = _split_batch_for_oom_retry(fr_batch)
    en_parts = _split_batch_for_oom_retry(en_batch)
    if len(fr_parts) == 1 and len(en_parts) == 1:
        return [(fr_parts[0], en_parts[0])]
    if len(fr_parts) == 1:
        return [(fr_parts[0], part) for part in en_parts if part]
    if len(en_parts) == 1:
        return [(part, en_parts[0]) for part in fr_parts if part]
    paired = []
    for fr_part, en_part in zip(fr_parts, en_parts):
        if fr_part and en_part:
            paired.append((fr_part, en_part))
    return paired


def _run_stage1_batches_with_oom_backoff(
    *,
    batches: Sequence[Sequence],
    label: str,
    compute_metrics,
    optimizer=None,
    trainable_params: Optional[Sequence[nn.Parameter]] = None,
) -> Dict[str, float]:
    pending_batches: List[List] = [list(batch) for batch in batches if batch]
    weighted_records: List[Tuple[int, Dict[str, float]]] = []
    while pending_batches:
        batch = pending_batches.pop(0)
        try:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            total, metrics = compute_metrics(batch)
            if optimizer is not None:
                total.backward()
                if trainable_params is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
            weighted_records.append((len(batch), metrics))
        except Exception as exc:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if not _is_cuda_oom_error(exc) or len(batch) <= 1:
                raise
            smaller_batches = _split_batch_for_oom_retry(batch)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sizes = ",".join(str(len(part)) for part in smaller_batches if part)
            print(f"[{label}] CUDA OOM at batch_size={len(batch)}; retrying with batch_sizes={sizes}")
            pending_batches = [part for part in smaller_batches if part] + pending_batches
    return _weighted_average_metric_records(weighted_records)


def _run_stage2_batches_with_oom_backoff(
    *,
    batch_pairs: Sequence[Tuple[Sequence, Sequence]],
    label: str,
    compute_metrics,
    optimizer=None,
    trainable_params: Optional[Sequence[nn.Parameter]] = None,
) -> Dict[str, float]:
    pending_pairs: List[Tuple[List, List]] = [
        (list(fr_batch), list(en_batch))
        for fr_batch, en_batch in batch_pairs
        if fr_batch and en_batch
    ]
    weighted_records: List[Tuple[int, Dict[str, float]]] = []
    while pending_pairs:
        fr_batch, en_batch = pending_pairs.pop(0)
        try:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            total, metrics = compute_metrics(fr_batch, en_batch)
            if optimizer is not None:
                total.backward()
                if trainable_params is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
            weighted_records.append((max(len(fr_batch), len(en_batch)), metrics))
        except Exception as exc:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if not _is_cuda_oom_error(exc) or (len(fr_batch) <= 1 and len(en_batch) <= 1):
                raise
            smaller_pairs = _split_paired_batches_for_oom_retry(fr_batch, en_batch)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sizes = ",".join(f"{len(fr_part)}x{len(en_part)}" for fr_part, en_part in smaller_pairs)
            print(
                f"[{label}] CUDA OOM at fr_batch_size={len(fr_batch)} "
                f"en_batch_size={len(en_batch)}; retrying with batch_pairs={sizes}"
            )
            pending_pairs = smaller_pairs + pending_pairs
    return _weighted_average_metric_records(weighted_records)


def stack_negative_id_sets(fr_batch: Sequence, *, max_steps: Optional[int] = None) -> List[torch.Tensor]:
    negative_sets = [torch.stack([truncate_token_ids(ex.negative_ids, max_steps) for ex in fr_batch], dim=0)]
    if all(ex.hard_negative_ids is not None for ex in fr_batch):
        negative_sets.append(torch.stack([truncate_token_ids(ex.hard_negative_ids, max_steps) for ex in fr_batch], dim=0))
    return negative_sets


def hardest_weighted_logprob_margin_loss(
    logits: torch.Tensor,
    pos_ids: torch.Tensor,
    negative_id_sets: Sequence[torch.Tensor],
    weights: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    losses = [
        weighted_logprob_margin_loss(logits, pos_ids, neg_ids, weights, margin)
        for neg_ids in negative_id_sets
    ]
    return torch.stack(losses, dim=0).max()


@torch.no_grad()
def english_fraction_with_negatives(
    logits: torch.Tensor,
    english_ids: torch.Tensor,
    negative_id_sets: Sequence[torch.Tensor],
    steps: int,
) -> float:
    k = min(int(steps), logits.shape[1], english_ids.shape[1], *(neg.shape[1] for neg in negative_id_sets))
    if k <= 0:
        return 0.0
    pred = logits[:, :k, :].argmax(dim=-1)
    pos = (pred == english_ids[:, :k]).float().mean().item()
    neg = max((pred == neg_ids[:, :k]).float().mean().item() for neg_ids in negative_id_sets)
    return float(pos - neg)


@torch.no_grad()
def collect_hard_negative_references(
    model,
    paver: ResidualPaver,
    examples: Sequence,
    max_steps: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not examples or any(ex.hard_negative_ids is None for ex in examples):
        return None, None

    logits_list = []
    states_list = []
    for ex in examples:
        logits, states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.base_ids,
            anchor_ids=truncate_token_ids(ex.hard_negative_ids, max_steps),
            controller_vectors=None,
            gate_strength=0.0,
        )
        logits_list.append(logits)
        states_list.append(states)
    return torch.stack(logits_list, dim=0), torch.stack(states_list, dim=0)


def gather_train_traces_with_controller_batch(
    model,
    paver: ResidualPaver,
    controller_batch: torch.Tensor,
    examples: Sequence,
    gate_strengths: torch.Tensor,
    max_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_list = []
    states_list = []
    for ex, controller_vectors, gate_strength in zip(examples, controller_batch, gate_strengths):
        logits, states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.base_ids,
            anchor_ids=truncate_token_ids(ex.anchor_ids, max_steps),
            controller_vectors=controller_vectors,
            gate_strength=gate_strength,
        )
        logits_list.append(logits)
        states_list.append(states)
    return torch.stack(logits_list, dim=0), torch.stack(states_list, dim=0)


def build_route_batch(
    *,
    boundary_batch: torch.Tensor,
    examples: Sequence,
    router: Optional[BoundaryExpertRouter],
    num_experts: int,
    bucket_smoothing: float,
    fallback_mode: str = "bucket_prior",
    fallback_centroids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if router is None or max(int(num_experts), 1) == 1:
        mode = str(fallback_mode).strip().lower() or "bucket_prior"
        if mode == "uniform":
            weights = uniform_route_weights(
                len(examples),
                num_experts=num_experts,
                device=boundary_batch.device,
            )
        elif mode == "boundary_kmeans" and fallback_centroids is not None:
            weights = assign_boundary_centroid_routes(
                boundary_batch,
                centroids=fallback_centroids,
                num_experts=num_experts,
                device=boundary_batch.device,
                smoothing=bucket_smoothing,
            )
        else:
            weights = fixed_bucket_route_weights(
                examples,
                num_experts=num_experts,
                device=boundary_batch.device,
                smoothing=bucket_smoothing,
            )
        logits = torch.log(weights.clamp_min(1e-6))
        return logits, weights
    return router(boundary_batch)


def maybe_warm_start_router(
    router: Optional[BoundaryExpertRouter],
    *,
    examples: Sequence,
    num_experts: int,
    bucket_smoothing: float,
    fallback_mode: str,
    fallback_centroids: Optional[torch.Tensor],
    steps: int,
    lr: float,
) -> bool:
    if router is None or not examples or max(int(num_experts), 1) <= 1 or int(steps) <= 0:
        return False
    boundary_batch = torch.stack([ex.boundary_states.float() for ex in examples], dim=0)
    with torch.no_grad():
        _, target_weights = build_route_batch(
            boundary_batch=boundary_batch,
            examples=examples,
            router=None,
            num_experts=num_experts,
            bucket_smoothing=bucket_smoothing,
            fallback_mode=fallback_mode,
            fallback_centroids=fallback_centroids,
        )
        target_idx = target_weights.argmax(dim=-1)
    opt = torch.optim.Adam(router.parameters(), lr=float(lr))
    final_loss = 0.0
    for _ in range(max(int(steps), 0)):
        opt.zero_grad(set_to_none=True)
        logits, weights = router(boundary_batch)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), target_weights.detach(), reduction="batchmean")
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())
    with torch.no_grad():
        _, warmed_weights = router(boundary_batch)
        match = ((warmed_weights.argmax(dim=-1) == target_idx).float().mean().item()) if target_idx.numel() > 0 else 0.0
        print(
            "[router warm start] "
            f"mode={fallback_mode} steps={int(steps)} lr={float(lr):.4f} "
            f"loss={final_loss:.4f} match={match:.3f} entropy={float(route_entropy(warmed_weights).item()):.3f}"
        )
    return True


def split_binary_example_pairs(
    fr_examples: Sequence,
    en_examples: Sequence,
    *,
    test_fraction: float,
    seed: int,
) -> Tuple[List, List, List, List]:
    if len(fr_examples) != len(en_examples):
        raise ValueError("French and English example sets must have the same length for paired splitting.")
    num_pairs = len(fr_examples)
    if num_pairs == 0:
        raise ValueError("At least one example pair is required.")

    frac = float(test_fraction)
    if frac <= 0.0 or num_pairs < 2:
        return list(fr_examples), list(en_examples), [], []

    rng = random.Random(seed)
    bucket_to_indices: Dict[str, List[int]] = {}
    for idx, ex in enumerate(fr_examples):
        bucket_to_indices.setdefault(example_bucket(ex), []).append(idx)

    test_index_set = set()
    for bucket in sorted(bucket_to_indices.keys()):
        bucket_indices = list(bucket_to_indices[bucket])
        if len(bucket_indices) < 2:
            continue
        rng.shuffle(bucket_indices)
        bucket_test_count = int(round(len(bucket_indices) * frac))
        bucket_test_count = max(1, min(bucket_test_count, len(bucket_indices) - 1))
        test_index_set.update(bucket_indices[:bucket_test_count])

    if not test_index_set:
        indices = list(range(num_pairs))
        rng.shuffle(indices)
        test_index_set.add(indices[0])

    train_fr = [ex for i, ex in enumerate(fr_examples) if i not in test_index_set]
    train_en = [ex for i, ex in enumerate(en_examples) if i not in test_index_set]
    test_fr = [ex for i, ex in enumerate(fr_examples) if i in test_index_set]
    test_en = [ex for i, ex in enumerate(en_examples) if i in test_index_set]
    return train_fr, train_en, test_fr, test_en


@torch.no_grad()
def maybe_binary_gate_stats(
    gate: AnchoredBinaryGate,
    fr_examples: Sequence,
    en_examples: Sequence,
) -> Optional[Dict[str, float]]:
    if not fr_examples or not en_examples:
        return None
    return binary_gate_stats(gate, fr_examples, en_examples)


def append_metric_fields(parts: List[str], metrics: Dict[str, float], keys: Sequence[str]) -> None:
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")


def format_stage1_metrics(
    *,
    tag: str,
    epoch: int,
    traj_scale: float,
    metrics: Dict[str, float],
    gate_stats: Optional[Dict[str, float]],
) -> str:
    parts = [
        f"[{tag} epoch {epoch:02d}]",
        f"traj_scale={traj_scale:.4f}",
        f"total={metrics.get('total', 0.0):.4f}",
        f"fr_ce={metrics.get('fr_ce', 0.0):.4f}",
        f"fr_ce_front={metrics.get('fr_ce_front', 0.0):.4f}",
        f"fr_kl={metrics.get('fr_kl', 0.0):.4f}",
        f"fr_kl_front={metrics.get('fr_kl_front', 0.0):.4f}",
        f"fr_tok_margin={metrics.get('fr_tok_margin', 0.0):.4f}",
        f"fr_traj_close={metrics.get('fr_traj_close', 0.0):.4f}",
        f"fr_traj_sep={metrics.get('fr_traj_sep', 0.0):.4f}",
        f"fr_traj_mono={metrics.get('fr_traj_mono', 0.0):.4f}",
        f"eng_minus_fr={metrics.get('eng_minus_fr', 0.0):.4f}",
    ]
    append_metric_fields(
        parts,
        metrics,
        [
            "eng_minus_fr_1_8",
            "eng_minus_fr_9_24",
            "eng_minus_fr_25_40",
            "fr_sem_content",
            "fr_sem_mode",
            "fr_sem_length",
            "fr_sem_refusal",
            "route_reg",
            "route_entropy",
            "route_00",
            "route_01",
            "route_02",
            "route_03",
            "cons_logit",
            "cons_state",
            "cons_gate_raw",
            "cons_gate_prob",
            "cons_route",
        ],
    )
    if gate_stats is not None:
        parts.extend(
            [
                f"fr_gate_mean={gate_stats['fr_gate_mean']:.3f}",
                f"en_gate_mean={gate_stats['en_gate_mean']:.3f}",
                f"thr_acc={gate_stats['threshold_acc']:.3f}",
            ]
        )
    return " ".join(parts)


def format_stage2_metrics(
    *,
    tag: str,
    epoch: int,
    traj_scale: float,
    metrics: Dict[str, float],
    gate_stats: Optional[Dict[str, float]],
) -> str:
    parts = [
        f"[{tag} epoch {epoch:02d}]",
        f"traj_scale={traj_scale:.4f}",
        f"total={metrics.get('total', 0.0):.4f}",
        f"fr_ce={metrics.get('fr_ce', 0.0):.4f}",
        f"fr_ce_front={metrics.get('fr_ce_front', 0.0):.4f}",
        f"fr_kl={metrics.get('fr_kl', 0.0):.4f}",
        f"fr_kl_front={metrics.get('fr_kl_front', 0.0):.4f}",
        f"fr_tok_margin={metrics.get('fr_tok_margin', 0.0):.4f}",
        f"fr_traj_close={metrics.get('fr_traj_close', 0.0):.4f}",
        f"fr_traj_sep={metrics.get('fr_traj_sep', 0.0):.4f}",
        f"fr_traj_mono={metrics.get('fr_traj_mono', 0.0):.4f}",
        f"en_kl={metrics.get('en_kl', 0.0):.4f}",
        f"en_kl_front={metrics.get('en_kl_front', 0.0):.4f}",
        f"en_traj_preserve={metrics.get('en_traj_preserve', 0.0):.4f}",
        f"gate_rank={metrics.get('gate_rank', 0.0):.4f}",
        f"eng_minus_fr={metrics.get('eng_minus_fr', 0.0):.4f}",
    ]
    append_metric_fields(
        parts,
        metrics,
        [
            "eng_minus_fr_1_8",
            "eng_minus_fr_9_24",
            "eng_minus_fr_25_40",
            "fr_sem_content",
            "fr_sem_mode",
            "fr_sem_length",
            "fr_sem_refusal",
            "en_sem_content",
            "en_sem_mode",
            "en_sem_length",
            "en_sem_refusal",
            "route_entropy",
            "route_00",
            "route_01",
            "route_02",
            "route_03",
            "cons_logit",
            "cons_state",
            "cons_gate_raw",
            "cons_gate_prob",
            "cons_route",
        ],
    )
    if gate_stats is not None:
        parts.extend(
            [
                f"fr_gate_mean={gate_stats['fr_gate_mean']:.3f}",
                f"en_gate_mean={gate_stats['en_gate_mean']:.3f}",
                f"thr_acc={gate_stats['threshold_acc']:.3f}",
            ]
        )
    return " ".join(parts)


def format_stage3_metrics(
    *,
    tag: str,
    epoch: int,
    metrics: Dict[str, float],
    gate_stats: Optional[Dict[str, float]],
) -> str:
    parts = [
        f"[{tag} epoch {epoch:02d}]",
        f"total={metrics.get('total', 0.0):.4f}",
        f"fr_bce={metrics.get('fr_bce', 0.0):.4f}",
        f"en_bce={metrics.get('en_bce', 0.0):.4f}",
        f"gate_rank={metrics.get('gate_rank', 0.0):.4f}",
        f"gate_sat={metrics.get('gate_sat', 0.0):.4f}",
        f"tau={metrics.get('tau', 0.0):.4f}",
        f"weighted_acc={metrics.get('weighted_acc', 0.0):.4f}",
        f"raw_acc={metrics.get('raw_acc', 0.0):.4f}",
    ]
    if gate_stats is not None:
        parts.extend(
            [
                f"fr_gate_mean={gate_stats['fr_gate_mean']:.3f}",
                f"en_gate_mean={gate_stats['en_gate_mean']:.3f}",
                f"thr_acc={gate_stats['threshold_acc']:.3f}",
            ]
        )
    return " ".join(parts)


def materialize_controller_batch(
    controller_bank: PavingControllerBank,
    route_weights: torch.Tensor,
    args,
) -> torch.Tensor:
    return controller_bank.controller_batch(
        route_weights,
        layer_start=args.layer_ramp_start,
        layer_end=args.layer_ramp_end,
        time_decay_tau=args.time_decay_tau,
        smooth_layer=args.post_smooth_layer,
        smooth_time=args.post_smooth_time,
    )


@torch.no_grad()
def materialize_prompt_controller(
    *,
    model,
    tokenizer,
    paver: ResidualPaver,
    controller_bank: PavingControllerBank,
    router: Optional[BoundaryExpertRouter],
    device: torch.device,
    user_prompt: str,
    args,
    prompt_bucket: str = "general",
    fallback_mode: str = "bucket_prior",
    fallback_centroids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    base_ids = encode_chat_prompt(tokenizer, user_prompt, None, device)
    boundary = capture_boundary_states(model, base_ids, paver).unsqueeze(0)
    dummy_example = type("PromptExample", (), {"bucket": prompt_bucket})()
    _, route_weights = build_route_batch(
        boundary_batch=boundary,
        examples=[dummy_example],
        router=router,
        num_experts=controller_bank.num_experts,
        bucket_smoothing=args.router_bucket_smoothing,
        fallback_mode=fallback_mode,
        fallback_centroids=fallback_centroids,
    )
    controller_batch = materialize_controller_batch(controller_bank, route_weights, args)
    return controller_batch[0]


def maybe_stage1_consistency_terms(
    *,
    model,
    tokenizer,
    paver: ResidualPaver,
    controller_bank: PavingControllerBank,
    router: Optional[BoundaryExpertRouter],
    base_examples: Sequence,
    base_route_weights: torch.Tensor,
    base_logits: torch.Tensor,
    base_states: torch.Tensor,
    gate_strengths: torch.Tensor,
    stage1_route_init: str,
    stage1_route_centroids: Optional[torch.Tensor],
    train_steps: Optional[int],
    args,
) -> Dict[str, torch.Tensor]:
    zero = base_logits.new_zeros(())
    if args.consistency_weight <= 0.0 or args.consistency_max_variants <= 0:
        return {"logit": zero, "state": zero, "gate_raw": zero, "gate_prob": zero, "route": zero}

    variant_examples = build_prompt_variant_examples(
        model=model,
        tokenizer=tokenizer,
        paver=paver,
        device=base_logits.device,
        examples=base_examples,
        max_variants=args.consistency_max_variants,
        variant_index=0,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    if len(variant_examples) != len(base_examples):
        return {"logit": zero, "state": zero, "gate_raw": zero, "gate_prob": zero, "route": zero}

    variant_boundary = torch.stack([ex.boundary_states for ex in variant_examples], dim=0)
    _, variant_route_weights = build_route_batch(
        boundary_batch=variant_boundary,
        examples=variant_examples,
        router=router,
        num_experts=controller_bank.num_experts,
        bucket_smoothing=args.stage1_bucket_route_smoothing,
        fallback_mode=stage1_route_init,
        fallback_centroids=stage1_route_centroids,
    )
    variant_controller_batch = materialize_controller_batch(controller_bank, variant_route_weights, args)
    variant_logits, variant_states = gather_train_traces_with_controller_batch(
        model=model,
        paver=paver,
        controller_batch=variant_controller_batch,
        examples=variant_examples,
        gate_strengths=gate_strengths,
        max_steps=train_steps,
    )
    return consistency_terms(
        base_logits=base_logits,
        base_states=base_states,
        variant_logits=variant_logits,
        variant_states=variant_states,
        base_route_weights=base_route_weights,
        variant_route_weights=variant_route_weights,
        max_steps=args.consistency_steps,
        eps=args.consistency_eps,
    )


def maybe_stage2_consistency_terms(
    *,
    model,
    tokenizer,
    paver: ResidualPaver,
    controller_bank: PavingControllerBank,
    router: Optional[BoundaryExpertRouter],
    gate: AnchoredBinaryGate,
    base_examples: Sequence,
    base_logits: torch.Tensor,
    base_states: torch.Tensor,
    base_gate_logits: torch.Tensor,
    base_gate_probs: torch.Tensor,
    base_route_weights: torch.Tensor,
    stage1_route_init: str,
    stage1_route_centroids: Optional[torch.Tensor],
    train_steps: Optional[int],
    args,
) -> Dict[str, torch.Tensor]:
    zero = base_logits.new_zeros(())
    if args.consistency_weight <= 0.0 or args.consistency_max_variants <= 0:
        return {"logit": zero, "state": zero, "gate_raw": zero, "gate_prob": zero, "route": zero}

    variant_examples = build_prompt_variant_examples(
        model=model,
        tokenizer=tokenizer,
        paver=paver,
        device=base_logits.device,
        examples=base_examples,
        max_variants=args.consistency_max_variants,
        variant_index=0,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    if len(variant_examples) != len(base_examples):
        return {"logit": zero, "state": zero, "gate_raw": zero, "gate_prob": zero, "route": zero}

    variant_boundary = torch.stack([ex.boundary_states for ex in variant_examples], dim=0)
    variant_gate_logits, variant_gate_probs = gate(variant_boundary)
    variant_route_logits, variant_route_weights = build_route_batch(
        boundary_batch=variant_boundary,
        examples=variant_examples,
        router=router,
        num_experts=controller_bank.num_experts,
        bucket_smoothing=args.router_bucket_smoothing,
        fallback_mode=stage1_route_init,
        fallback_centroids=stage1_route_centroids,
    )
    _ = variant_route_logits
    variant_labels = torch.tensor([ex.gate_target for ex in variant_examples], device=variant_gate_logits.device, dtype=variant_gate_logits.dtype)
    variant_effective = torch.where(
        variant_labels > 0.5,
        args.stage2_pos_floor + (1.0 - args.stage2_pos_floor) * variant_gate_probs,
        args.stage2_neg_cap * variant_gate_probs,
    )
    variant_controller_batch = materialize_controller_batch(controller_bank, variant_route_weights, args)
    variant_logits, variant_states = gather_train_traces_with_controller_batch(
        model=model,
        paver=paver,
        controller_batch=variant_controller_batch,
        examples=variant_examples,
        gate_strengths=variant_effective,
        max_steps=train_steps,
    )
    return consistency_terms(
        base_logits=base_logits,
        base_states=base_states,
        variant_logits=variant_logits,
        variant_states=variant_states,
        base_gate_logits=base_gate_logits,
        variant_gate_logits=variant_gate_logits,
        base_gate_probs=base_gate_probs,
        variant_gate_probs=variant_gate_probs,
        base_route_weights=base_route_weights,
        variant_route_weights=variant_route_weights,
        max_steps=args.consistency_steps,
        eps=args.consistency_eps,
    )


def compute_stage1_batch_objective(
    *,
    model,
    tokenizer,
    paver: ResidualPaver,
    gate: AnchoredBinaryGate,
    controller_bank: PavingControllerBank,
    router: Optional[BoundaryExpertRouter],
    fr_batch: Sequence,
    step_weights: torch.Tensor,
    traj_step_weights: torch.Tensor,
    layer_weights: torch.Tensor,
    scorer: ProjectedTrajectoryScorer,
    loss_config: HybridLossConfig,
    semantic_bank: Dict[str, Optional[Dict[str, Any]]],
    stage1_route_init: str,
    stage1_route_centroids: Optional[torch.Tensor],
    train_steps: Optional[int],
    args,
    traj_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    effective_train_steps = resolve_stage_train_steps(args, stage=1, for_eval=False) if train_steps is None else max(1, int(train_steps))
    boundary_batch = torch.stack([ex.boundary_states for ex in fr_batch], dim=0)
    route_logits, route_weights = build_route_batch(
        boundary_batch=boundary_batch,
        examples=fr_batch,
        router=router,
        num_experts=controller_bank.num_experts,
        bucket_smoothing=args.stage1_bucket_route_smoothing,
        fallback_mode=stage1_route_init,
        fallback_centroids=stage1_route_centroids,
    )
    _ = route_logits
    controller_batch = materialize_controller_batch(controller_bank, route_weights, args)
    gate_strengths = torch.ones(len(fr_batch), device=controller_batch.device, dtype=torch.float32) * float(args.stage1_fr_gate)
    fr_train_logits, fr_train_states = gather_train_traces_with_controller_batch(
        model=model,
        paver=paver,
        controller_batch=controller_batch,
        examples=fr_batch,
        gate_strengths=gate_strengths,
        max_steps=effective_train_steps,
    )
    (
        fr_base_logits,
        fr_base_states,
        fr_pos_logits,
        fr_pos_states,
        fr_neg_logits,
        fr_neg_states,
    ) = collect_fr_references(model=model, paver=paver, examples=fr_batch, max_steps=effective_train_steps)
    fr_hard_neg_logits, fr_hard_neg_states = collect_hard_negative_references(
        model=model,
        paver=paver,
        examples=fr_batch,
        max_steps=effective_train_steps,
    )

    fr_targets = torch.stack([truncate_token_ids(ex.anchor_ids, effective_train_steps) for ex in fr_batch], dim=0)
    fr_negative_id_sets = stack_negative_id_sets(fr_batch, max_steps=effective_train_steps)
    fr_ce_full = F.cross_entropy(fr_train_logits.reshape(-1, fr_train_logits.shape[-1]), fr_targets.reshape(-1))
    fr_ce_front = step_weighted_ce(fr_train_logits, fr_targets, step_weights)
    fr_ce = fr_ce_full + args.fr_early_ce_weight * fr_ce_front

    fr_teacher_probs = torch.softmax(fr_pos_logits.detach().float(), dim=-1)
    fr_kl_full = F.kl_div(F.log_softmax(fr_train_logits.float(), dim=-1), fr_teacher_probs, reduction="batchmean")
    fr_kl_front = step_weighted_kl(fr_train_logits, fr_pos_logits, step_weights)
    fr_kl = fr_kl_full + args.fr_early_kl_weight * fr_kl_front

    fr_token_margin = hardest_weighted_logprob_margin_loss(
        fr_train_logits,
        fr_targets,
        fr_negative_id_sets,
        step_weights,
        margin=args.token_margin,
    )
    fr_traj = fr_aux_trajectory_terms(
        scorer=scorer,
        train_logits=fr_train_logits,
        train_states=fr_train_states,
        pos_logits=fr_pos_logits,
        pos_states=fr_pos_states,
        base_logits=fr_base_logits,
        base_states=fr_base_states,
        neg_logits=fr_neg_logits,
        neg_states=fr_neg_states,
        hard_neg_logits=fr_hard_neg_logits,
        hard_neg_states=fr_hard_neg_states,
        step_weights=traj_step_weights,
        layer_weights=layer_weights,
        target_fraction=loss_config.fr_improvement_target,
        floor=loss_config.fr_baseline_floor,
        sep_margin=args.traj_sep_margin,
        mono_tol=args.traj_mono_tol,
        eps=loss_config.eps,
    )
    fr_semantic = semantic_anchor_terms(
        train_states=fr_train_states,
        ref_states=fr_pos_states,
        examples=fr_batch,
        bank=semantic_bank,
        temperature=args.semantic_temperature,
        eps=loss_config.eps,
    )
    consistency = maybe_stage1_consistency_terms(
        model=model,
        tokenizer=tokenizer,
        paver=paver,
        controller_bank=controller_bank,
        router=router,
        base_examples=fr_batch,
        base_route_weights=route_weights,
        base_logits=fr_train_logits,
        base_states=fr_train_states,
        gate_strengths=gate_strengths,
        stage1_route_init=stage1_route_init,
        stage1_route_centroids=stage1_route_centroids,
        train_steps=effective_train_steps,
        args=args,
    )
    reg = controller_bank.regularizer(
        l2=loss_config.l2,
        layer_smooth=loss_config.layer_smooth,
        time_smooth=loss_config.time_smooth,
        coeff_l2=args.controller_coeff_l2,
    )
    route_reg = args.route_entropy_weight * (-route_entropy(route_weights)) + args.route_balance_weight * route_balance_penalty(route_weights)
    consistency_total = (
        args.consistency_logit_weight * consistency["logit"]
        + args.consistency_state_weight * consistency["state"]
        + args.consistency_gate_raw_weight * consistency["gate_raw"]
        + args.consistency_gate_prob_weight * consistency["gate_prob"]
        + args.consistency_route_weight * consistency["route"]
    )
    total = (
        loss_config.ce_weight * fr_ce
        + loss_config.fr_kl_weight * fr_kl
        + args.fr_token_margin_weight * fr_token_margin
        + traj_scale * (
            args.fr_traj_close_weight * fr_traj["close"]
            + args.fr_traj_sep_weight * fr_traj["sep"]
            + args.fr_traj_mono_weight * fr_traj["mono"]
        )
        + args.semantic_content_weight * fr_semantic["content"]
        + args.semantic_mode_weight * fr_semantic["mode"]
        + args.semantic_length_weight * fr_semantic["length"]
        + args.semantic_refusal_weight * fr_semantic["refusal"]
        + args.consistency_weight * consistency_total
        + route_reg
        + reg
    )

    metrics = {
        "total": float(total.detach().cpu()),
        "fr_ce": float(fr_ce.detach().cpu()),
        "fr_ce_front": float(fr_ce_front.detach().cpu()),
        "fr_kl": float(fr_kl.detach().cpu()),
        "fr_kl_front": float(fr_kl_front.detach().cpu()),
        "fr_tok_margin": float(fr_token_margin.detach().cpu()),
        "fr_traj_close": float(fr_traj["close"].detach().cpu()),
        "fr_traj_sep": float(fr_traj["sep"].detach().cpu()),
        "fr_traj_mono": float(fr_traj["mono"].detach().cpu()),
        "fr_sem_content": float(fr_semantic["content"].detach().cpu()),
        "fr_sem_mode": float(fr_semantic["mode"].detach().cpu()),
        "fr_sem_length": float(fr_semantic["length"].detach().cpu()),
        "fr_sem_refusal": float(fr_semantic["refusal"].detach().cpu()),
        "route_reg": float(route_reg.detach().cpu()),
        "eng_minus_fr": english_fraction_with_negatives(fr_train_logits.detach(), fr_targets, fr_negative_id_sets, args.metric_steps),
        "cons_logit": float(consistency["logit"].detach().cpu()),
        "cons_state": float(consistency["state"].detach().cpu()),
        "cons_gate_raw": float(consistency["gate_raw"].detach().cpu()),
        "cons_gate_prob": float(consistency["gate_prob"].detach().cpu()),
        "cons_route": float(consistency["route"].detach().cpu()),
    }
    metrics.update(english_fraction_window_metrics(fr_train_logits.detach(), fr_targets, fr_negative_id_sets, DEFAULT_METRIC_WINDOWS))
    metrics.update(route_usage_metrics(route_weights.detach()))
    return total, metrics


def compute_stage2_batch_objective(
    *,
    model,
    tokenizer,
    paver: ResidualPaver,
    gate: AnchoredBinaryGate,
    controller_bank: PavingControllerBank,
    router: Optional[BoundaryExpertRouter],
    fr_batch: Sequence,
    en_batch: Sequence,
    step_weights: torch.Tensor,
    traj_step_weights: torch.Tensor,
    layer_weights: torch.Tensor,
    scorer: ProjectedTrajectoryScorer,
    loss_config: HybridLossConfig,
    semantic_bank: Dict[str, Optional[Dict[str, Any]]],
    stage1_route_init: str,
    stage1_route_centroids: Optional[torch.Tensor],
    train_steps: Optional[int],
    args,
    traj_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    effective_train_steps = resolve_stage_train_steps(args, stage=2, for_eval=False) if train_steps is None else max(1, int(train_steps))
    fr_boundary = torch.stack([ex.boundary_states for ex in fr_batch], dim=0)
    fr_gate_logits, fr_gate_probs = gate(fr_boundary)
    fr_gate_targets = torch.tensor([ex.gate_target for ex in fr_batch], device=fr_gate_logits.device, dtype=fr_gate_logits.dtype)
    fr_effective = torch.where(
        fr_gate_targets > 0.5,
        args.stage2_pos_floor + (1.0 - args.stage2_pos_floor) * fr_gate_probs,
        args.stage2_neg_cap * fr_gate_probs,
    )
    _, fr_route_weights = build_route_batch(
        boundary_batch=fr_boundary,
        examples=fr_batch,
        router=router,
        num_experts=controller_bank.num_experts,
        bucket_smoothing=args.router_bucket_smoothing,
        fallback_mode=stage1_route_init,
        fallback_centroids=stage1_route_centroids,
    )
    fr_controller_batch = materialize_controller_batch(controller_bank, fr_route_weights, args)
    fr_train_logits, fr_train_states = gather_train_traces_with_controller_batch(
        model=model,
        paver=paver,
        controller_batch=fr_controller_batch,
        examples=fr_batch,
        gate_strengths=fr_effective,
        max_steps=effective_train_steps,
    )
    (
        fr_base_logits,
        fr_base_states,
        fr_pos_logits,
        fr_pos_states,
        fr_neg_logits,
        fr_neg_states,
    ) = collect_fr_references(model=model, paver=paver, examples=fr_batch, max_steps=effective_train_steps)
    fr_hard_neg_logits, fr_hard_neg_states = collect_hard_negative_references(
        model=model,
        paver=paver,
        examples=fr_batch,
        max_steps=effective_train_steps,
    )
    fr_targets = torch.stack([truncate_token_ids(ex.anchor_ids, effective_train_steps) for ex in fr_batch], dim=0)
    fr_negative_id_sets = stack_negative_id_sets(fr_batch, max_steps=effective_train_steps)

    fr_ce_full = F.cross_entropy(fr_train_logits.reshape(-1, fr_train_logits.shape[-1]), fr_targets.reshape(-1))
    fr_ce_front = step_weighted_ce(fr_train_logits, fr_targets, step_weights)
    fr_ce = fr_ce_full + args.fr_early_ce_weight * fr_ce_front
    fr_teacher_probs = torch.softmax(fr_pos_logits.detach().float(), dim=-1)
    fr_kl_full = F.kl_div(F.log_softmax(fr_train_logits.float(), dim=-1), fr_teacher_probs, reduction="batchmean")
    fr_kl_front = step_weighted_kl(fr_train_logits, fr_pos_logits, step_weights)
    fr_kl = fr_kl_full + args.fr_early_kl_weight * fr_kl_front
    fr_token_margin = hardest_weighted_logprob_margin_loss(fr_train_logits, fr_targets, fr_negative_id_sets, step_weights, margin=args.token_margin)
    fr_traj = fr_aux_trajectory_terms(
        scorer=scorer,
        train_logits=fr_train_logits,
        train_states=fr_train_states,
        pos_logits=fr_pos_logits,
        pos_states=fr_pos_states,
        base_logits=fr_base_logits,
        base_states=fr_base_states,
        neg_logits=fr_neg_logits,
        neg_states=fr_neg_states,
        hard_neg_logits=fr_hard_neg_logits,
        hard_neg_states=fr_hard_neg_states,
        step_weights=traj_step_weights,
        layer_weights=layer_weights,
        target_fraction=loss_config.fr_improvement_target,
        floor=loss_config.fr_baseline_floor,
        sep_margin=args.traj_sep_margin,
        mono_tol=args.traj_mono_tol,
        eps=loss_config.eps,
    )
    fr_gate_bce = F.binary_cross_entropy_with_logits(fr_gate_logits, fr_gate_targets)
    fr_semantic = semantic_anchor_terms(
        train_states=fr_train_states,
        ref_states=fr_pos_states,
        examples=fr_batch,
        bank=semantic_bank,
        temperature=args.semantic_temperature,
        eps=loss_config.eps,
    )

    en_boundary = torch.stack([ex.boundary_states for ex in en_batch], dim=0)
    en_gate_logits, en_gate_probs = gate(en_boundary)
    en_gate_targets = torch.tensor([ex.gate_target for ex in en_batch], device=en_gate_logits.device, dtype=en_gate_logits.dtype)
    en_effective = torch.where(
        en_gate_targets > 0.5,
        args.stage2_pos_floor + (1.0 - args.stage2_pos_floor) * en_gate_probs,
        args.stage2_neg_cap * en_gate_probs,
    )
    _, en_route_weights = build_route_batch(
        boundary_batch=en_boundary,
        examples=en_batch,
        router=router,
        num_experts=controller_bank.num_experts,
        bucket_smoothing=args.router_bucket_smoothing,
        fallback_mode=stage1_route_init,
        fallback_centroids=stage1_route_centroids,
    )
    en_controller_batch = materialize_controller_batch(controller_bank, en_route_weights, args)
    en_train_logits, en_train_states = gather_train_traces_with_controller_batch(
        model=model,
        paver=paver,
        controller_batch=en_controller_batch,
        examples=en_batch,
        gate_strengths=en_effective,
        max_steps=effective_train_steps,
    )
    en_base_logits, en_base_states = collect_en_references(
        model=model,
        paver=paver,
        examples=en_batch,
        max_steps=effective_train_steps,
    )
    en_base_probs = torch.softmax(en_base_logits.detach().float(), dim=-1)
    en_kl_full = F.kl_div(F.log_softmax(en_train_logits.float(), dim=-1), en_base_probs, reduction="batchmean")
    en_kl_front = step_weighted_kl(en_train_logits, en_base_logits, step_weights)
    en_kl = en_kl_full + args.en_early_kl_weight * en_kl_front
    en_traj = en_aux_trajectory_terms(
        scorer=scorer,
        train_logits=en_train_logits,
        train_states=en_train_states,
        base_logits=en_base_logits,
        base_states=en_base_states,
        step_weights=traj_step_weights,
        layer_weights=layer_weights,
        tolerance=args.en_preserve_tolerance,
        scale=args.en_preserve_scale,
        mono_tol=args.traj_mono_tol,
        eps=loss_config.eps,
    )
    en_gate_bce = F.binary_cross_entropy_with_logits(en_gate_logits, en_gate_targets)
    en_semantic = semantic_anchor_terms(
        train_states=en_train_states,
        ref_states=en_base_states,
        examples=en_batch,
        bank=semantic_bank,
        temperature=args.semantic_temperature,
        eps=loss_config.eps,
    )

    gate_rank = gate_pair_margin_loss(fr_gate_logits, en_gate_logits, args.gate_rank_margin)
    gate_sat = gate_saturation_penalty(torch.cat([fr_gate_probs, en_gate_probs], dim=0))
    gate_reg = gate.regularizer(
        residual_l2=args.gate_residual_l2,
        layer_logit_l2=args.gate_layer_l2,
        sharpness_center=args.gate_sharpness,
        sharpness_l2=args.gate_sharpness_l2,
    )
    all_route_weights = torch.cat([fr_route_weights, en_route_weights], dim=0)
    route_reg = args.route_entropy_weight * (-route_entropy(all_route_weights)) + args.route_balance_weight * route_balance_penalty(all_route_weights)
    reg = controller_bank.regularizer(
        l2=loss_config.l2,
        layer_smooth=loss_config.layer_smooth,
        time_smooth=loss_config.time_smooth,
        coeff_l2=args.controller_coeff_l2,
    )
    consistency = maybe_stage2_consistency_terms(
        model=model,
        tokenizer=tokenizer,
        paver=paver,
        controller_bank=controller_bank,
        router=router,
        gate=gate,
        base_examples=fr_batch,
        base_logits=fr_train_logits,
        base_states=fr_train_states,
        base_gate_logits=fr_gate_logits,
        base_gate_probs=fr_gate_probs,
        base_route_weights=fr_route_weights,
        stage1_route_init=stage1_route_init,
        stage1_route_centroids=stage1_route_centroids,
        train_steps=effective_train_steps,
        args=args,
    )
    consistency_total = (
        args.consistency_logit_weight * consistency["logit"]
        + args.consistency_state_weight * consistency["state"]
        + args.consistency_gate_raw_weight * consistency["gate_raw"]
        + args.consistency_gate_prob_weight * consistency["gate_prob"]
        + args.consistency_route_weight * consistency["route"]
    )
    total = (
        loss_config.ce_weight * fr_ce
        + loss_config.fr_kl_weight * fr_kl
        + args.fr_token_margin_weight * fr_token_margin
        + traj_scale
        * (
            args.fr_traj_close_weight * fr_traj["close"]
            + args.fr_traj_sep_weight * fr_traj["sep"]
            + args.fr_traj_mono_weight * fr_traj["mono"]
            + args.en_traj_preserve_weight * en_traj["preserve"]
            + args.en_traj_mono_weight * en_traj["mono"]
        )
        + loss_config.nonfr_kl_weight * en_kl
        + loss_config.gate_bce_weight * (fr_gate_bce + en_gate_bce)
        + args.gate_rank_weight * gate_rank
        + args.gate_sat_weight * gate_sat
        + args.gate_reg_weight * gate_reg
        + args.semantic_content_weight * (fr_semantic["content"] + en_semantic["content"])
        + args.semantic_mode_weight * (fr_semantic["mode"] + en_semantic["mode"])
        + args.semantic_length_weight * (fr_semantic["length"] + en_semantic["length"])
        + args.semantic_refusal_weight * (fr_semantic["refusal"] + en_semantic["refusal"])
        + args.consistency_weight * consistency_total
        + route_reg
        + reg
    )

    metrics = {
        "total": float(total.detach().cpu()),
        "fr_ce": float(fr_ce.detach().cpu()),
        "fr_ce_front": float(fr_ce_front.detach().cpu()),
        "fr_kl": float(fr_kl.detach().cpu()),
        "fr_kl_front": float(fr_kl_front.detach().cpu()),
        "fr_tok_margin": float(fr_token_margin.detach().cpu()),
        "fr_traj_close": float(fr_traj["close"].detach().cpu()),
        "fr_traj_sep": float(fr_traj["sep"].detach().cpu()),
        "fr_traj_mono": float(fr_traj["mono"].detach().cpu()),
        "en_kl": float(en_kl.detach().cpu()),
        "en_kl_front": float(en_kl_front.detach().cpu()),
        "en_traj_preserve": float(en_traj["preserve"].detach().cpu()),
        "en_traj_mono": float(en_traj["mono"].detach().cpu()),
        "gate_rank": float(gate_rank.detach().cpu()),
        "gate_sat": float(gate_sat.detach().cpu()),
        "eng_minus_fr": english_fraction_with_negatives(fr_train_logits.detach(), fr_targets, fr_negative_id_sets, args.metric_steps),
        "fr_sem_content": float(fr_semantic["content"].detach().cpu()),
        "fr_sem_mode": float(fr_semantic["mode"].detach().cpu()),
        "fr_sem_length": float(fr_semantic["length"].detach().cpu()),
        "fr_sem_refusal": float(fr_semantic["refusal"].detach().cpu()),
        "en_sem_content": float(en_semantic["content"].detach().cpu()),
        "en_sem_mode": float(en_semantic["mode"].detach().cpu()),
        "en_sem_length": float(en_semantic["length"].detach().cpu()),
        "en_sem_refusal": float(en_semantic["refusal"].detach().cpu()),
        "cons_logit": float(consistency["logit"].detach().cpu()),
        "cons_state": float(consistency["state"].detach().cpu()),
        "cons_gate_raw": float(consistency["gate_raw"].detach().cpu()),
        "cons_gate_prob": float(consistency["gate_prob"].detach().cpu()),
        "cons_route": float(consistency["route"].detach().cpu()),
    }
    metrics.update(english_fraction_window_metrics(fr_train_logits.detach(), fr_targets, fr_negative_id_sets, DEFAULT_METRIC_WINDOWS))
    metrics.update(route_usage_metrics(all_route_weights.detach()))
    return total, metrics


@torch.no_grad()
def evaluate_stage1_epoch(
    *,
    model,
    tokenizer,
    paver: ResidualPaver,
    gate: AnchoredBinaryGate,
    controller_bank: PavingControllerBank,
    router: Optional[BoundaryExpertRouter],
    fr_examples: Sequence,
    en_examples: Sequence,
    fr_batch_size: int,
    step_weights: torch.Tensor,
    traj_step_weights: torch.Tensor,
    layer_weights: torch.Tensor,
    scorer: ProjectedTrajectoryScorer,
    loss_config: HybridLossConfig,
    semantic_bank: Dict[str, Optional[Dict[str, Any]]],
    stage1_route_init: str,
    stage1_route_centroids: Optional[torch.Tensor],
    args,
    traj_scale: float,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    if not fr_examples:
        return None, None

    avg = _run_stage1_batches_with_oom_backoff(
        batches=batched(list(fr_examples), fr_batch_size),
        label="stage1 test",
        compute_metrics=lambda fr_batch: compute_stage1_batch_objective(
            model=model,
            tokenizer=tokenizer,
            paver=paver,
            gate=gate,
            controller_bank=controller_bank,
            router=router,
            fr_batch=fr_batch,
            step_weights=step_weights,
            traj_step_weights=traj_step_weights,
            layer_weights=layer_weights,
            scorer=scorer,
            loss_config=loss_config,
            semantic_bank=semantic_bank,
            stage1_route_init=stage1_route_init,
            stage1_route_centroids=stage1_route_centroids,
            train_steps=resolve_stage_train_steps(args, stage=1, for_eval=True),
            args=args,
            traj_scale=traj_scale,
        ),
    )
    return avg, maybe_binary_gate_stats(gate, fr_examples, en_examples)


@torch.no_grad()
def evaluate_stage2_epoch(
    *,
    model,
    tokenizer,
    paver: ResidualPaver,
    gate: AnchoredBinaryGate,
    controller_bank: PavingControllerBank,
    router: Optional[BoundaryExpertRouter],
    fr_examples: Sequence,
    en_examples: Sequence,
    fr_batch_size: int,
    en_batch_size: int,
    step_weights: torch.Tensor,
    traj_step_weights: torch.Tensor,
    layer_weights: torch.Tensor,
    scorer: ProjectedTrajectoryScorer,
    loss_config: HybridLossConfig,
    semantic_bank: Dict[str, Optional[Dict[str, Any]]],
    stage1_route_init: str,
    stage1_route_centroids: Optional[torch.Tensor],
    args,
    traj_scale: float,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    if not fr_examples or not en_examples:
        return None, None

    fr_batches = batched(list(fr_examples), fr_batch_size)
    en_batches = batched(list(en_examples), en_batch_size)
    fr_cycle = itertools.cycle(fr_batches)
    en_cycle = itertools.cycle(en_batches)
    num_updates = max(len(fr_batches), len(en_batches))
    batch_pairs = [(next(fr_cycle), next(en_cycle)) for _ in range(num_updates)]
    avg = _run_stage2_batches_with_oom_backoff(
        batch_pairs=batch_pairs,
        label="stage2 test",
        compute_metrics=lambda fr_batch, en_batch: compute_stage2_batch_objective(
            model=model,
            tokenizer=tokenizer,
            paver=paver,
            gate=gate,
            controller_bank=controller_bank,
            router=router,
            fr_batch=fr_batch,
            en_batch=en_batch,
            step_weights=step_weights,
            traj_step_weights=traj_step_weights,
            layer_weights=layer_weights,
            scorer=scorer,
            loss_config=loss_config,
            semantic_bank=semantic_bank,
            stage1_route_init=stage1_route_init,
            stage1_route_centroids=stage1_route_centroids,
            train_steps=resolve_stage_train_steps(args, stage=2, for_eval=True),
            args=args,
            traj_scale=traj_scale,
        ),
    )
    return avg, maybe_binary_gate_stats(gate, fr_examples, en_examples)


@torch.no_grad()
def evaluate_stage3_epoch(
    *,
    gate: AnchoredBinaryGate,
    fr_examples: Sequence,
    en_examples: Sequence,
    args,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    if not fr_examples or not en_examples:
        return None, None

    fr_boundary = torch.stack([ex.boundary_states for ex in fr_examples], dim=0)
    en_boundary = torch.stack([ex.boundary_states for ex in en_examples], dim=0)
    fr_targets = torch.ones(len(fr_examples), device=fr_boundary.device, dtype=torch.float32)
    en_targets = torch.zeros(len(en_examples), device=en_boundary.device, dtype=torch.float32)

    fr_logits, fr_probs = gate(fr_boundary)
    en_logits, en_probs = gate(en_boundary)
    fr_bce = F.binary_cross_entropy_with_logits(fr_logits, fr_targets)
    en_bce = F.binary_cross_entropy_with_logits(en_logits, en_targets)
    gate_rank = gate_pair_margin_loss(fr_logits, en_logits, args.calib_gate_margin)
    gate_sat = gate_saturation_penalty(torch.cat([fr_probs, en_probs], dim=0))
    gate_reg = gate.regularizer(
        residual_l2=args.gate_residual_l2,
        layer_logit_l2=args.gate_layer_l2,
        sharpness_center=args.gate_sharpness,
        sharpness_l2=args.gate_sharpness_l2,
    )
    total = fr_bce + en_bce + args.calib_gate_rank_weight * gate_rank + args.calib_gate_sat_weight * gate_sat + args.gate_reg_weight * gate_reg

    fr_raw = gate.raw_score(fr_boundary)
    en_raw = gate.raw_score(en_boundary)
    unit_fr_w = torch.ones_like(fr_raw)
    unit_en_w = torch.ones_like(en_raw)
    tau, wacc = choose_weighted_threshold(fr_raw, en_raw, unit_fr_w, unit_en_w)
    pred_acc = torch.cat([(fr_raw > tau).float(), (en_raw <= tau).float()]).mean().item()

    metrics = {
        "total": float(total.detach().cpu()),
        "fr_bce": float(fr_bce.detach().cpu()),
        "en_bce": float(en_bce.detach().cpu()),
        "gate_rank": float(gate_rank.detach().cpu()),
        "gate_sat": float(gate_sat.detach().cpu()),
        "tau": float(tau),
        "weighted_acc": float(wacc),
        "raw_acc": float(pred_acc),
    }
    return metrics, maybe_binary_gate_stats(gate, fr_examples, en_examples)


def train_binary_merged(args):
    if args.test_fraction < 0.0:
        raise ValueError("--test_fraction must be non-negative.")
    if args.test_every < 0:
        raise ValueError("--test_every must be non-negative.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        device,
        quantization=args.quantization,
        quant_compute_dtype=args.quant_compute_dtype,
    )
    total_layers = len(get_decoder_layers(model))
    layer_indices = parse_layer_indices(args.layers, total_layers)
    paver = ResidualPaver(
        model=model,
        layer_indices=layer_indices,
        intervention_positions=args.intervention_positions,
    )
    hidden_size = get_hidden_size(model)
    num_layers = len(layer_indices)
    vocab_size = int(model.lm_head.weight.shape[0])
    num_controller_experts = resolve_num_controller_experts(args.dataset_track, args.num_controller_experts)
    stage1_route_init = resolve_stage1_route_init(args.dataset_track, args.stage1_route_init)

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    loss_config = HybridLossConfig(
        ce_weight=args.ce_weight,
        fr_kl_weight=args.fr_kl_weight,
        nonfr_kl_weight=args.en_kl_weight,
        fr_traj_weight=1.0,
        nonfr_traj_weight=1.0,
        traj_prob_weight=args.traj_prob_weight,
        traj_logit_weight=args.traj_logit_weight,
        traj_state_weight=args.traj_state_weight,
        fr_improvement_target=args.fr_improvement_target,
        fr_baseline_floor=args.fr_baseline_floor,
        preserve_tolerance=args.en_preserve_tolerance,
        preserve_scale=args.en_preserve_scale,
        gate_bce_weight=args.gate_bce_weight,
        gate_entropy_weight=0.0,
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

    train_prompt_records, test_prompt_records, prompt_data_sources, uses_explicit_test_split = load_dataset_track_prompt_records(
        args=args,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    print(f"Dataset track: {args.dataset_track}")
    if num_controller_experts != int(args.num_controller_experts):
        print(f"[experts] auto-adjusted num_controller_experts={num_controller_experts} for dataset track {args.dataset_track}")
    if stage1_route_init != str(args.stage1_route_init).strip().lower():
        print(f"[routing] auto-adjusted stage1_route_init={stage1_route_init} for dataset track {args.dataset_track}")
    print(f"Train prompt source: {prompt_data_sources['train']}")
    print(f"Train prompt records loaded: {len(train_prompt_records)}")
    print(f"[train prompt buckets] {format_bucket_counts(train_prompt_records)}")
    if uses_explicit_test_split:
        print(f"Test prompt source: {prompt_data_sources['test']}")
        print(f"Test prompt records loaded: {len(test_prompt_records)}")
        print(f"[test prompt buckets] {format_bucket_counts(test_prompt_records)}")
        if args.test_fraction > 0.0:
            print("[dataset track] ignoring --test_fraction because this dataset track provides an explicit held-out test split.")

    build_batch_size = max(
        1,
        int(args.build_batch_size)
        if int(args.build_batch_size) > 0
        else min(8, int(args.fr_batch_size), int(args.en_batch_size)),
    )

    print("Building French / English training examples ...")
    train_fr_examples, train_en_examples = build_binary_examples(
        model=model,
        tokenizer=tokenizer,
        paver=paver,
        device=device,
        steer_steps=args.steer_steps,
        prompt_records=train_prompt_records,
        use_hard_negatives=not args.disable_hard_negatives,
        max_prompt_tokens=args.max_prompt_tokens,
        build_batch_size=build_batch_size,
        progress_label="train",
        progress_every=args.build_progress_every,
    )
    if uses_explicit_test_split:
        print("Building French / English test examples ...")
        test_fr_examples, test_en_examples = build_binary_examples(
            model=model,
            tokenizer=tokenizer,
            paver=paver,
            device=device,
            steer_steps=args.steer_steps,
            prompt_records=test_prompt_records,
            use_hard_negatives=not args.disable_hard_negatives,
            max_prompt_tokens=args.max_prompt_tokens,
            build_batch_size=build_batch_size,
            progress_label="test",
            progress_every=args.build_progress_every,
        )
        fr_examples = list(train_fr_examples)
        en_examples = list(train_en_examples)
    else:
        all_fr_examples = list(train_fr_examples)
        all_en_examples = list(train_en_examples)
        train_fr_examples, train_en_examples, test_fr_examples, test_en_examples = split_binary_example_pairs(
            all_fr_examples,
            all_en_examples,
            test_fraction=args.test_fraction,
            seed=args.test_seed if args.test_seed is not None else args.seed + 101,
        )
        fr_examples = train_fr_examples
        en_examples = train_en_examples

    stage1_route_centroids = None
    if stage1_route_init == "boundary_kmeans":
        stage1_route_centroids = compute_boundary_kmeans_centroids(
            fr_examples,
            num_experts=num_controller_experts,
            seed=args.seed,
        )

    print(f"French train examples: {len(fr_examples)}")
    print(f"English train examples: {len(en_examples)}")
    print(f"French test examples: {len(test_fr_examples)}")
    print(f"English test examples: {len(test_en_examples)}")
    print(f"[stage1 routing] init={stage1_route_init}")
    print(
        "[train buckets] "
        + " ".join(
            f"{bucket}={count}"
            for bucket, count in sorted(
                {
                    bucket: sum(1 for ex in fr_examples if example_bucket(ex) == bucket)
                    for bucket in {example_bucket(ex) for ex in fr_examples}
                }.items()
            )
        )
    )
    if test_fr_examples:
        print(
            "[test buckets] "
            + " ".join(
                f"{bucket}={count}"
                for bucket, count in sorted(
                    {
                        bucket: sum(1 for ex in test_fr_examples if example_bucket(ex) == bucket)
                        for bucket in {example_bucket(ex) for ex in test_fr_examples}
                    }.items()
                )
            )
        )
    if test_fr_examples and test_en_examples:
        print(
            "[test split] "
            f"fr_ids={','.join(ex.sample_id for ex in test_fr_examples)} "
            f"en_ids={','.join(ex.sample_id for ex in test_en_examples)}"
        )
    preview_teacher_pairs(tokenizer, fr_examples, limit=min(2, len(fr_examples)))

    init_direction, init_threshold, init_stats = init_binary_gate_from_examples(fr_examples, en_examples)
    gate = AnchoredBinaryGate(
        init_direction=init_direction,
        init_threshold=init_threshold,
        sharpness=args.gate_sharpness,
        learn_direction_residual=args.learn_direction_residual,
        residual_scale=args.gate_residual_scale,
        gate_type=args.gate_type,
        hidden_dim=args.gate_hidden_dim,
    ).to(device)

    basis_bundle = None
    if args.controller_mode != "free":
        if not args.basis_path:
            raise ValueError("--basis_path is required when --controller_mode is not 'free'.")
        basis_bundle = load_controller_basis_bundle(
            args.basis_path,
            device=device,
            steer_steps=args.steer_steps,
            num_layers=num_layers,
            hidden_size=hidden_size,
        )
        explained_energy = basis_bundle.get("explained_energy")
        if explained_energy is not None and explained_energy.numel() > 0:
            print(
                "[basis] "
                f"rank={basis_bundle['basis_vectors'].shape[0]} "
                f"top_energy={float(explained_energy[0].item()):.4f} "
                f"cum_energy={float(explained_energy.sum().item()):.4f}"
            )

    controller_bank = PavingControllerBank(
        num_experts=num_controller_experts,
        steer_steps=args.steer_steps,
        num_layers=num_layers,
        hidden_size=hidden_size,
        controller_mode=args.controller_mode,
        basis_bundle=basis_bundle,
    ).to(device)
    _ = maybe_warm_start_free_controller(
        controller_bank,
        fr_examples=fr_examples,
        en_examples=en_examples,
        init_mode=getattr(args, "free_controller_init", "none"),
        init_scale=float(getattr(args, "free_controller_init_scale", 0.1)),
    )
    router = None
    if num_controller_experts > 1 and not args.disable_expert_router:
        router = BoundaryExpertRouter(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_experts=num_controller_experts,
            hidden_dim=args.router_hidden_dim,
            temperature=args.router_temperature,
            topk=args.router_topk,
        ).to(device)
    _ = maybe_warm_start_router(
        router,
        examples=list(fr_examples) + list(en_examples),
        num_experts=num_controller_experts,
        bucket_smoothing=args.stage1_bucket_route_smoothing,
        fallback_mode=stage1_route_init,
        fallback_centroids=stage1_route_centroids,
        steps=int(args.router_warm_start_steps),
        lr=float(args.router_warm_start_lr),
    )
    _ = maybe_warm_start_gate(
        gate,
        fr_examples=fr_examples,
        en_examples=en_examples,
        steps=int(args.gate_warm_start_steps),
        lr=float(args.gate_warm_start_lr),
    )

    stage1_params = list(controller_bank.parameters())
    if router is not None:
        stage1_params += list(router.parameters())
    stage2_params = list(controller_bank.parameters()) + list(gate.parameters())
    if router is not None:
        stage2_params += list(router.parameters())
    opt_stage1 = torch.optim.Adam(stage1_params, lr=args.lr_stage1)
    opt_stage2 = torch.optim.Adam(stage2_params, lr=args.lr_stage2)
    opt_stage3 = torch.optim.Adam(list(gate.parameters()), lr=args.lr_stage3)

    step_weights = make_piecewise_step_weights(
        args.steer_steps,
        device=device,
        dtype=torch.float32,
        strong_steps=args.onramp_strong_steps,
        medium_steps=args.onramp_medium_steps,
        strong_weight=args.onramp_strong_weight,
        medium_weight=args.onramp_medium_weight,
        tail_weight=args.onramp_tail_weight,
    )
    traj_step_weights = make_piecewise_step_weights(
        args.steer_steps,
        device=device,
        dtype=torch.float32,
        strong_steps=args.onramp_strong_steps,
        medium_steps=args.onramp_medium_steps,
        strong_weight=args.traj_onramp_strong_weight,
        medium_weight=args.traj_onramp_medium_weight,
        tail_weight=args.traj_onramp_tail_weight,
    )
    calib_step_weights = make_step_weights(
        args.steer_steps,
        device=device,
        dtype=torch.float32,
        focus_steps=args.prefix_focus_steps,
        focus_boost=args.prefix_focus_boost,
        decay_tau=args.prefix_decay_tau,
    )
    layer_weights = make_layer_weights(
        num_layers=num_layers,
        device=device,
        dtype=torch.float32,
        late_boost=args.state_late_boost,
    )

    print(
        "[gate init] "
        f"train_acc={init_stats['train_acc']:.3f} "
        f"fr_mean={init_stats['fr_mean']:.4f} "
        f"en_mean={init_stats['en_mean']:.4f} "
        f"threshold={init_stats['threshold']:.4f}"
    )
    print(f"[intervention] positions={paver.intervention_positions}")
    stage1_train_steps = resolve_stage_train_steps(args, stage=1, for_eval=False)
    stage1_eval_steps = resolve_stage_train_steps(args, stage=1, for_eval=True)
    stage2_train_steps = resolve_stage_train_steps(args, stage=2, for_eval=False)
    stage2_eval_steps = resolve_stage_train_steps(args, stage=2, for_eval=True)
    print(f"[stage1 config] train_steps={stage1_train_steps} eval_steps={stage1_eval_steps}")
    print(f"[stage2 config] train_steps={stage2_train_steps} eval_steps={stage2_eval_steps}")

    semantic_bank = build_semantic_anchor_bank(fr_examples + en_examples)

    preview_train_fr = args.preview_fr_prompt or train_prompt_records[0]["fr"]
    preview_train_en = args.preview_en_prompt or train_prompt_records[0]["en"]
    preview_test_fr = args.preview_test_fr_prompt or (test_prompt_records[0]["fr"] if test_prompt_records else None)
    preview_test_en = args.preview_test_en_prompt or (test_prompt_records[0]["en"] if test_prompt_records else None)
    last_eng_minus_fr = 0.0

    @torch.no_grad()
    def generate_preview_outputs(
        fr_prompt: Optional[str],
        en_prompt: Optional[str],
        *,
        router_override,
        strength: float,
        hard_gate: bool,
    ) -> Dict[str, Tuple[float, float, str]]:
        outputs: Dict[str, Tuple[float, float, str]] = {}
        if fr_prompt:
            fr_controller = materialize_prompt_controller(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                controller_bank=controller_bank,
                router=router_override,
                device=device,
                user_prompt=fr_prompt,
                args=args,
                fallback_mode=stage1_route_init,
                fallback_centroids=stage1_route_centroids,
            )
            outputs["fr"] = decode_with_binary_gate(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                gate=gate,
                controller_vectors=fr_controller,
                device=device,
                user_prompt=fr_prompt,
                max_new_tokens=args.max_new_tokens,
                strength=strength,
                temperature=args.temperature,
                top_p=args.top_p,
                hard_gate=hard_gate,
            )
        if en_prompt:
            en_controller = materialize_prompt_controller(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                controller_bank=controller_bank,
                router=router_override,
                device=device,
                user_prompt=en_prompt,
                args=args,
                fallback_mode=stage1_route_init,
                fallback_centroids=stage1_route_centroids,
            )
            outputs["en"] = decode_with_binary_gate(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                gate=gate,
                controller_vectors=en_controller,
                device=device,
                user_prompt=en_prompt,
                max_new_tokens=args.max_new_tokens,
                strength=strength,
                temperature=args.temperature,
                top_p=args.top_p,
                hard_gate=hard_gate,
            )
        return outputs

    # -------------------- Stage 1: controller only --------------------
    for epoch in range(1, args.stage1_epochs + 1):
        fr_batches = make_example_batches(
            fr_examples,
            batch_size=args.fr_batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            bucket_batches=not args.disable_bucket_batches,
        )
        traj_scale = choose_traj_scale(
            epoch=epoch,
            total_epochs=args.stage1_epochs,
            max_scale=args.stage1_traj_max_scale,
            warmup_epochs=args.stage1_traj_warmup,
            last_metric=last_eng_minus_fr,
            unlock_metric=args.unlock_eng_minus_fr,
            cold_scale=args.stage1_traj_cold_scale,
        )

        avg = _run_stage1_batches_with_oom_backoff(
            batches=fr_batches,
            label=f"stage1 train epoch={epoch}",
            optimizer=opt_stage1,
            trainable_params=stage1_params,
            compute_metrics=lambda fr_batch: compute_stage1_batch_objective(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                gate=gate,
                controller_bank=controller_bank,
                router=router,
                fr_batch=fr_batch,
                step_weights=step_weights,
                traj_step_weights=traj_step_weights,
                layer_weights=layer_weights,
                scorer=scorer,
                loss_config=loss_config,
                semantic_bank=semantic_bank,
                stage1_route_init=stage1_route_init,
                stage1_route_centroids=stage1_route_centroids,
                train_steps=stage1_train_steps,
                args=args,
                traj_scale=traj_scale,
            ),
        )
        last_eng_minus_fr = float(avg.get("eng_minus_fr", last_eng_minus_fr))
        train_gstats = maybe_binary_gate_stats(gate, fr_examples, en_examples)
        print(
            format_stage1_metrics(
                tag="stage1 train",
                epoch=epoch,
                traj_scale=traj_scale,
                metrics=avg,
                gate_stats=train_gstats,
            )
        )
        if test_fr_examples and args.test_every > 0 and epoch % args.test_every == 0:
            test_metrics, test_gstats = evaluate_stage1_epoch(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                gate=gate,
                controller_bank=controller_bank,
                router=router,
                fr_examples=test_fr_examples,
                en_examples=test_en_examples,
                fr_batch_size=args.fr_batch_size,
                step_weights=step_weights,
                traj_step_weights=traj_step_weights,
                layer_weights=layer_weights,
                scorer=scorer,
                loss_config=loss_config,
                semantic_bank=semantic_bank,
                stage1_route_init=stage1_route_init,
                stage1_route_centroids=stage1_route_centroids,
                args=args,
                traj_scale=traj_scale,
            )
            if test_metrics is not None:
                print(
                    format_stage1_metrics(
                        tag="stage1 test",
                        epoch=epoch,
                        traj_scale=traj_scale,
                        metrics=test_metrics,
                        gate_stats=test_gstats,
                    )
                )
        if preview_train_fr or preview_train_en:
            train_preview_outputs = generate_preview_outputs(
                preview_train_fr,
                preview_train_en,
                router_override=router,
                strength=1.0,
                hard_gate=True,
            )
            if "fr" in train_preview_outputs:
                fr_raw, fr_strength, fr_text = train_preview_outputs["fr"]
                print(f"  stage1_train_fr_gate_raw={fr_raw:.3f} strength={fr_strength:.3f} preview={fr_text[:220]!r}")
            if "en" in train_preview_outputs:
                en_raw, en_strength, en_text = train_preview_outputs["en"]
                print(f"  stage1_train_en_gate_raw={en_raw:.3f} strength={en_strength:.3f} preview={en_text[:220]!r}")
        if preview_test_fr or preview_test_en:
            test_preview_outputs = generate_preview_outputs(
                preview_test_fr,
                preview_test_en,
                router_override=router,
                strength=1.0,
                hard_gate=True,
            )
            if "fr" in test_preview_outputs:
                fr_raw, fr_strength, fr_text = test_preview_outputs["fr"]
                print(f"  stage1_test_fr_gate_raw={fr_raw:.3f} strength={fr_strength:.3f} preview={fr_text[:220]!r}")
            if "en" in test_preview_outputs:
                en_raw, en_strength, en_text = test_preview_outputs["en"]
                print(f"  stage1_test_en_gate_raw={en_raw:.3f} strength={en_strength:.3f} preview={en_text[:220]!r}")

    # -------------------- Stage 2: controller + gate --------------------
    for epoch in range(1, args.stage2_epochs + 1):
        fr_batches = make_example_batches(
            fr_examples,
            batch_size=args.fr_batch_size,
            shuffle=True,
            seed=args.seed + 10_000 + epoch,
            bucket_batches=not args.disable_bucket_batches,
        )
        en_batches = make_example_batches(
            en_examples,
            batch_size=args.en_batch_size,
            shuffle=True,
            seed=args.seed + 20_000 + epoch,
            bucket_batches=not args.disable_bucket_batches,
        )
        fr_cycle = itertools.cycle(fr_batches)
        en_cycle = itertools.cycle(en_batches)
        num_updates = max(len(fr_batches), len(en_batches))
        traj_scale = choose_traj_scale(
            epoch=epoch,
            total_epochs=args.stage2_epochs,
            max_scale=args.stage2_traj_max_scale,
            warmup_epochs=args.stage2_traj_warmup,
            last_metric=last_eng_minus_fr,
            unlock_metric=args.unlock_eng_minus_fr,
            cold_scale=args.stage2_traj_cold_scale,
        )

        batch_pairs = [(next(fr_cycle), next(en_cycle)) for _ in range(num_updates)]
        avg = _run_stage2_batches_with_oom_backoff(
            batch_pairs=batch_pairs,
            label=f"stage2 train epoch={epoch}",
            optimizer=opt_stage2,
            trainable_params=stage2_params,
            compute_metrics=lambda fr_batch, en_batch: compute_stage2_batch_objective(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                gate=gate,
                controller_bank=controller_bank,
                router=router,
                fr_batch=fr_batch,
                en_batch=en_batch,
                step_weights=step_weights,
                traj_step_weights=traj_step_weights,
                layer_weights=layer_weights,
                scorer=scorer,
                loss_config=loss_config,
                semantic_bank=semantic_bank,
                stage1_route_init=stage1_route_init,
                stage1_route_centroids=stage1_route_centroids,
                train_steps=stage2_train_steps,
                args=args,
                traj_scale=traj_scale,
            ),
        )
        last_eng_minus_fr = float(avg.get("eng_minus_fr", last_eng_minus_fr))
        gstats = maybe_binary_gate_stats(gate, fr_examples, en_examples)
        print(
            format_stage2_metrics(
                tag="stage2 train",
                epoch=epoch,
                traj_scale=traj_scale,
                metrics=avg,
                gate_stats=gstats,
            )
        )
        if test_fr_examples and args.test_every > 0 and epoch % args.test_every == 0:
            test_metrics, test_gstats = evaluate_stage2_epoch(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                gate=gate,
                controller_bank=controller_bank,
                router=router,
                fr_examples=test_fr_examples,
                en_examples=test_en_examples,
                fr_batch_size=args.fr_batch_size,
                en_batch_size=args.en_batch_size,
                step_weights=step_weights,
                traj_step_weights=traj_step_weights,
                layer_weights=layer_weights,
                scorer=scorer,
                loss_config=loss_config,
                semantic_bank=semantic_bank,
                stage1_route_init=stage1_route_init,
                stage1_route_centroids=stage1_route_centroids,
                args=args,
                traj_scale=traj_scale,
            )
            if test_metrics is not None:
                print(
                    format_stage2_metrics(
                        tag="stage2 test",
                        epoch=epoch,
                        traj_scale=traj_scale,
                        metrics=test_metrics,
                        gate_stats=test_gstats,
                    )
                )
        if preview_train_fr or preview_train_en:
            train_preview_outputs = generate_preview_outputs(
                preview_train_fr,
                preview_train_en,
                router_override=router,
                strength=args.strength,
                hard_gate=args.preview_hard_gate,
            )
            if "fr" in train_preview_outputs:
                fr_raw, fr_strength, fr_text = train_preview_outputs["fr"]
                print(f"  stage2_train_fr_gate_raw={fr_raw:.3f} strength={fr_strength:.3f} preview={fr_text[:220]!r}")
            if "en" in train_preview_outputs:
                en_raw, en_strength, en_text = train_preview_outputs["en"]
                print(f"  stage2_train_en_gate_raw={en_raw:.3f} strength={en_strength:.3f} preview={en_text[:220]!r}")
        if preview_test_fr or preview_test_en:
            test_preview_outputs = generate_preview_outputs(
                preview_test_fr,
                preview_test_en,
                router_override=router,
                strength=args.strength,
                hard_gate=args.preview_hard_gate,
            )
            if "fr" in test_preview_outputs:
                fr_raw, fr_strength, fr_text = test_preview_outputs["fr"]
                print(f"  stage2_test_fr_gate_raw={fr_raw:.3f} strength={fr_strength:.3f} preview={fr_text[:220]!r}")
            if "en" in test_preview_outputs:
                en_raw, en_strength, en_text = test_preview_outputs["en"]
                print(f"  stage2_test_en_gate_raw={en_raw:.3f} strength={en_strength:.3f} preview={en_text[:220]!r}")

    # -------------------- Stage 3: gate-only calibration --------------------
    all_examples_for_routes = fr_examples + en_examples
    all_boundary_for_routes = torch.stack([ex.boundary_states for ex in all_examples_for_routes], dim=0)
    _, route_for_freeze = build_route_batch(
        boundary_batch=all_boundary_for_routes,
        examples=all_examples_for_routes,
        router=router,
        num_experts=controller_bank.num_experts,
        bucket_smoothing=args.router_bucket_smoothing,
        fallback_mode=stage1_route_init,
        fallback_centroids=stage1_route_centroids,
    )
    controller_frozen = materialize_controller_batch(
        controller_bank,
        route_for_freeze.mean(dim=0, keepdim=True),
        args,
    )[0].detach()
    if args.stage3_epochs > 0:
        fr_util_w, en_util_w, util_stats = compute_gate_utility_weights(
            model=model,
            paver=paver,
            scorer=scorer,
            controller_vectors=controller_frozen,
            fr_examples=fr_examples,
            en_examples=en_examples,
            step_weights=calib_step_weights,
            sep_reward=args.calib_sep_reward,
            utility_scale=args.calib_utility_scale,
        )
        print(
            "[stage3 utility] "
            f"fr_benefit_mean={util_stats['fr_benefit_mean']:.4f} "
            f"fr_benefit_max={util_stats['fr_benefit_max']:.4f} "
            f"en_harm_mean={util_stats['en_harm_mean']:.4f} "
            f"en_harm_max={util_stats['en_harm_max']:.4f}"
        )

        fr_boundary_all = torch.stack([ex.boundary_states for ex in fr_examples], dim=0)
        en_boundary_all = torch.stack([ex.boundary_states for ex in en_examples], dim=0)
        fr_targets_all = torch.ones(len(fr_examples), device=device, dtype=torch.float32)
        en_targets_all = torch.zeros(len(en_examples), device=device, dtype=torch.float32)

        for epoch in range(1, args.stage3_epochs + 1):
            opt_stage3.zero_grad(set_to_none=True)
            fr_logits, fr_probs = gate(fr_boundary_all)
            en_logits, en_probs = gate(en_boundary_all)
            fr_bce = F.binary_cross_entropy_with_logits(fr_logits, fr_targets_all, reduction="none")
            en_bce = F.binary_cross_entropy_with_logits(en_logits, en_targets_all, reduction="none")
            fr_bce = (fr_bce * fr_util_w).sum() / fr_util_w.sum().clamp_min(1e-6)
            en_bce = (en_bce * en_util_w).sum() / en_util_w.sum().clamp_min(1e-6)
            gate_rank = gate_pair_margin_loss(fr_logits, en_logits, args.calib_gate_margin, fr_util_w, en_util_w)
            gate_sat = gate_saturation_penalty(torch.cat([fr_probs, en_probs], dim=0))
            gate_reg = gate.regularizer(
                residual_l2=args.gate_residual_l2,
                layer_logit_l2=args.gate_layer_l2,
                sharpness_center=args.gate_sharpness,
                sharpness_l2=args.gate_sharpness_l2,
            )
            total = fr_bce + en_bce + args.calib_gate_rank_weight * gate_rank + args.calib_gate_sat_weight * gate_sat + args.gate_reg_weight * gate_reg
            total.backward()
            torch.nn.utils.clip_grad_norm_(list(gate.parameters()), max_norm=1.0)
            opt_stage3.step()

            with torch.no_grad():
                fr_raw = gate.raw_score(fr_boundary_all)
                en_raw = gate.raw_score(en_boundary_all)
                tau, wacc = choose_weighted_threshold(fr_raw, en_raw, fr_util_w, en_util_w)
                pred_acc = torch.cat([(fr_raw > tau).float(), (en_raw <= tau).float()]).mean().item()
            train_metrics = {
                "total": float(total.detach().cpu()),
                "fr_bce": float(fr_bce.detach().cpu()),
                "en_bce": float(en_bce.detach().cpu()),
                "gate_rank": float(gate_rank.detach().cpu()),
                "gate_sat": float(gate_sat.detach().cpu()),
                "tau": float(tau),
                "weighted_acc": float(wacc),
                "raw_acc": float(pred_acc),
            }
            train_gstats = maybe_binary_gate_stats(gate, fr_examples, en_examples)
            print(
                format_stage3_metrics(
                    tag="stage3 train",
                    epoch=epoch,
                    metrics=train_metrics,
                    gate_stats=train_gstats,
                )
            )
            if test_fr_examples and args.test_every > 0 and epoch % args.test_every == 0:
                test_metrics, test_gstats = evaluate_stage3_epoch(
                    gate=gate,
                    fr_examples=test_fr_examples,
                    en_examples=test_en_examples,
                    args=args,
                )
                if test_metrics is not None:
                    print(
                        format_stage3_metrics(
                            tag="stage3 test",
                            epoch=epoch,
                            metrics=test_metrics,
                            gate_stats=test_gstats,
                        )
                    )

        with torch.no_grad():
            fr_raw = gate.raw_score(fr_boundary_all)
            en_raw = gate.raw_score(en_boundary_all)
            tau, wacc = choose_weighted_threshold(fr_raw, en_raw, fr_util_w, en_util_w)
            gate.bias.sub_(torch.tensor(tau, device=gate.bias.device, dtype=gate.bias.dtype))
            print(f"[stage3 calibrate] folded_threshold={tau:.4f} weighted_acc={wacc:.4f}")

    final_controller = controller_frozen
    final_gstats = binary_gate_stats(gate, fr_examples, en_examples)
    print(
        "[final train gate] "
        f"fr_gate_mean={final_gstats['fr_gate_mean']:.3f} "
        f"en_gate_mean={final_gstats['en_gate_mean']:.3f} "
        f"fr_raw_mean={final_gstats['fr_raw_mean']:.3f} "
        f"en_raw_mean={final_gstats['en_raw_mean']:.3f} "
        f"thr_acc={final_gstats['threshold_acc']:.3f}"
    )
    if test_fr_examples and test_en_examples:
        final_test_gstats = binary_gate_stats(gate, test_fr_examples, test_en_examples)
        print(
            "[final test gate] "
            f"fr_gate_mean={final_test_gstats['fr_gate_mean']:.3f} "
            f"en_gate_mean={final_test_gstats['en_gate_mean']:.3f} "
            f"fr_raw_mean={final_test_gstats['fr_raw_mean']:.3f} "
            f"en_raw_mean={final_test_gstats['en_raw_mean']:.3f} "
            f"thr_acc={final_test_gstats['threshold_acc']:.3f}"
        )

    if args.save_path:
        controller_state_dict = {k: v.detach().cpu() for k, v in controller_bank.state_dict().items()}
        router_state_dict = {k: v.detach().cpu() for k, v in router.state_dict().items()} if router is not None else None
        save_binary_state(
            args.save_path,
            BinaryTrainState(
                raw_vectors=controller_bank.materialize_raw_expert_vectors().mean(dim=0).detach().cpu(),
                layer_indices=layer_indices,
                gate_state_dict={k: v.detach().cpu() for k, v in gate.state_dict().items()},
                gate_base_direction=gate.base_direction.detach().cpu(),
                gate_threshold=0.0,
                gate_residual_scale=args.gate_residual_scale,
                gate_sharpness_init=args.gate_sharpness,
                learn_direction_residual=args.learn_direction_residual,
                model_id=args.model_id,
                steer_steps=args.steer_steps,
                layer_ramp_start=args.layer_ramp_start,
                layer_ramp_end=args.layer_ramp_end,
                time_decay_tau=args.time_decay_tau,
                post_smooth_layer=args.post_smooth_layer,
                post_smooth_time=args.post_smooth_time,
                loss_config=asdict(loss_config),
                projection_config=asdict(proj_config),
                checkpoint_version=4,
                controller_state_dict=controller_state_dict,
                controller_config={
                    "num_experts": num_controller_experts,
                    "steer_steps": args.steer_steps,
                    "num_layers": num_layers,
                    "hidden_size": hidden_size,
                    "controller_mode": args.controller_mode,
                },
                router_state_dict=router_state_dict,
                router_config=(
                    {
                        "num_layers": num_layers,
                        "hidden_size": hidden_size,
                        "num_experts": num_controller_experts,
                        "hidden_dim": args.router_hidden_dim,
                        "temperature": args.router_temperature,
                        "topk": args.router_topk,
                    }
                    if router is not None
                    else None
                ),
                gate_config={
                    "gate_type": args.gate_type,
                    "hidden_dim": args.gate_hidden_dim,
                },
                intervention_positions=paver.intervention_positions,
            ),
        )

    if preview_train_fr or preview_train_en:
        print("\n===== FINAL TRAIN PREVIEWS =====")
        train_preview_outputs = generate_preview_outputs(
            preview_train_fr,
            preview_train_en,
            router_override=router,
            strength=args.strength,
            hard_gate=args.preview_hard_gate,
        )
        if "fr" in train_preview_outputs:
            fr_raw, fr_strength, fr_text = train_preview_outputs["fr"]
            print(f"[TRAIN FR raw={fr_raw:.4f} strength={fr_strength:.4f}]\n{fr_text}")
        if "en" in train_preview_outputs:
            en_raw, en_strength, en_text = train_preview_outputs["en"]
            print(f"\n[TRAIN EN raw={en_raw:.4f} strength={en_strength:.4f}]\n{en_text}")
    if preview_test_fr or preview_test_en:
        print("\n===== FINAL TEST PREVIEWS =====")
        test_preview_outputs = generate_preview_outputs(
            preview_test_fr,
            preview_test_en,
            router_override=router,
            strength=args.strength,
            hard_gate=args.preview_hard_gate,
        )
        if "fr" in test_preview_outputs:
            fr_raw, fr_strength, fr_text = test_preview_outputs["fr"]
            print(f"[TEST FR raw={fr_raw:.4f} strength={fr_strength:.4f}]\n{fr_text}")
        if "en" in test_preview_outputs:
            en_raw, en_strength, en_text = test_preview_outputs["en"]
            print(f"\n[TEST EN raw={en_raw:.4f} strength={en_strength:.4f}]\n{en_text}")


@torch.no_grad()
def infer_binary_merged(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_binary_state(args.load_path, device)

    model_id = state.get("model_id", args.model_id or DEFAULT_MODEL_ID)
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        device,
        quantization=args.quantization,
        quant_compute_dtype=args.quant_compute_dtype,
    )
    paver = ResidualPaver(
        model=model,
        layer_indices=[int(x) for x in state["layer_indices"]],
        intervention_positions=str(state.get("intervention_positions", "last_token")),
    )
    gate_config = state.get("gate_config") or {}
    gate = AnchoredBinaryGate(
        init_direction=state["gate_base_direction"],
        init_threshold=0.0,
        sharpness=float(state["gate_sharpness_init"]),
        learn_direction_residual=bool(state.get("learn_direction_residual", False)),
        residual_scale=float(state.get("gate_residual_scale", 0.05)),
        gate_type=str(gate_config.get("gate_type", "anchored_linear")),
        hidden_dim=int(gate_config.get("hidden_dim", 64)),
    ).to(device)
    gate.load_state_dict(state["gate_state_dict"])
    gate.eval()

    controller_args = type(
        "ControllerArgs",
        (),
        {
            "layer_ramp_start": float(state["layer_ramp_start"]),
            "layer_ramp_end": float(state["layer_ramp_end"]),
            "time_decay_tau": float(state["time_decay_tau"]),
            "post_smooth_layer": float(state["post_smooth_layer"]),
            "post_smooth_time": float(state["post_smooth_time"]),
            "router_bucket_smoothing": 0.05,
        },
    )()
    controller_config = state.get("controller_config")
    controller_state = state.get("controller_state_dict")
    router = None
    if controller_config and controller_state:
        controller_bank = PavingControllerBank(
            num_experts=int(controller_config["num_experts"]),
            steer_steps=int(controller_config["steer_steps"]),
            num_layers=int(controller_config["num_layers"]),
            hidden_size=int(controller_config["hidden_size"]),
            controller_mode=str(controller_config["controller_mode"]),
            basis_bundle=(
                {
                    "basis_vectors": controller_state["basis_vectors"],
                    "mean_direction": controller_state["mean_direction"],
                }
                if "basis_vectors" in controller_state and "mean_direction" in controller_state
                else None
            ),
        ).to(device)
        controller_bank.load_state_dict(controller_state)
        controller_bank.eval()
        router_config = state.get("router_config")
        router_state = state.get("router_state_dict")
        if router_config and router_state:
            router = BoundaryExpertRouter(
                num_layers=int(router_config["num_layers"]),
                hidden_size=int(router_config["hidden_size"]),
                num_experts=int(router_config["num_experts"]),
                hidden_dim=int(router_config["hidden_dim"]),
                temperature=float(router_config["temperature"]),
                topk=int(router_config["topk"]),
            ).to(device)
            router.load_state_dict(router_state)
            router.eval()
        controller = materialize_prompt_controller(
            model=model,
            tokenizer=tokenizer,
            paver=paver,
            controller_bank=controller_bank,
            router=router,
            device=device,
            user_prompt=args.prompt,
            args=controller_args,
        )
    else:
        controller = finalize_paving_vectors(
            state["raw_vectors"].to(device),
            layer_start=float(state["layer_ramp_start"]),
            layer_end=float(state["layer_ramp_end"]),
            time_decay_tau=float(state["time_decay_tau"]),
            smooth_layer=float(state["post_smooth_layer"]),
            smooth_time=float(state["post_smooth_time"]),
        )

    gate_raw, gate_strength, text = decode_with_binary_gate(
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
        hard_gate=args.hard_gate,
    )
    print(text)
    print(f"\n[gate_raw={gate_raw:.4f} gate_strength={gate_strength:.4f}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p_train.add_argument("--quantization", type=str, choices=["none", "8bit", "4bit"], default="none")
    p_train.add_argument("--quant_compute_dtype", type=str, choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"], default="auto")
    p_train.add_argument(
        "--dataset_track",
        type=str,
        choices=["parallel_french", "harmbench", "harmbench_parallel", "harmbench_2questions", "saladbench", "saladbench_2questions"],
        default="parallel_french",
    )
    p_train.add_argument("--prompt_data_path", type=str, default=None)
    p_train.add_argument("--max_prompt_records", type=int, default=None)
    p_train.add_argument("--harmbench_train_split", type=str, choices=["val", "test"], default="val")
    p_train.add_argument("--harmbench_test_split", type=str, choices=["val", "test"], default="test")
    p_train.add_argument("--harmbench_train_source", type=str, default=None)
    p_train.add_argument("--harmbench_test_source", type=str, default=None)
    p_train.add_argument("--alpaca_source", type=str, default=ALPACA_DEFAULT_SOURCE)
    p_train.add_argument("--alpaca_split_seed", type=int, default=17)
    p_train.add_argument("--harmbench_cache_dir", type=str, default=str(DEFAULT_HARMBENCH_CACHE_DIR))
    p_train.add_argument("--harmbench_refresh_cache", action="store_true")
    p_train.add_argument("--harmbench_bucket_field", type=str, default="SemanticCategory")
    p_train.add_argument("--harmbench_translation_max_new_tokens", type=int, default=512)
    p_train.add_argument("--harmbench_translation_temperature", type=float, default=0.0)
    p_train.add_argument("--harmbench_translation_top_p", type=float, default=0.95)
    p_train.add_argument("--harmbench_max_test_records", type=int, default=None)
    p_train.add_argument("--saladbench_train_split", type=str, choices=["base", "attackEnhanced", "defenseEnhanced"], default="base")
    p_train.add_argument("--saladbench_test_split", type=str, choices=["base", "attackEnhanced", "defenseEnhanced"], default="attackEnhanced")
    p_train.add_argument("--saladbench_train_source", type=str, default=None)
    p_train.add_argument("--saladbench_test_source", type=str, default=None)
    p_train.add_argument("--saladbench_split_from_base", action="store_true")
    p_train.add_argument("--saladbench_base_source", type=str, default=None)
    p_train.add_argument("--saladbench_base_total_records", type=int, default=None)
    p_train.add_argument("--saladbench_base_test_fraction", type=float, default=0.20)
    p_train.add_argument("--saladbench_base_split_seed", type=int, default=17)
    p_train.add_argument("--saladbench_bucket_level", type=str, choices=["top", "mid", "leaf"], default="top")
    p_train.add_argument("--saladbench_max_test_records", type=int, default=None)
    p_train.add_argument("--max_prompt_tokens", type=int, default=0)
    p_train.add_argument("--layers", type=str, default="-4,-3,-2,-1")
    p_train.add_argument("--steer_steps", type=int, default=24)
    p_train.add_argument("--stage1_train_steps", type=int, default=1)
    p_train.add_argument("--stage2_train_steps", type=int, default=4)
    p_train.add_argument("--stage1_epochs", type=int, default=12)
    p_train.add_argument("--stage2_epochs", type=int, default=16)
    p_train.add_argument("--stage3_epochs", type=int, default=12)
    p_train.add_argument("--fr_batch_size", type=int, default=4)
    p_train.add_argument("--en_batch_size", type=int, default=4)
    p_train.add_argument("--build_batch_size", type=int, default=0, help="Prompt-building batch size. Use 0 to auto-pick from train batch sizes.")
    p_train.add_argument("--build_progress_every", type=int, default=64, help="Print prompt-building progress every N records. Use 0 to only print the final count.")
    p_train.add_argument("--test_fraction", type=float, default=0.25, help="Held-out paired prompt fraction for test metrics.")
    p_train.add_argument("--test_seed", type=int, default=None, help="Optional split seed for the held-out test set.")
    p_train.add_argument("--test_every", type=int, default=1, help="Run held-out test evaluation every N epochs. Use 0 to disable.")
    p_train.add_argument("--lr_stage1", type=float, default=0.10)
    p_train.add_argument("--lr_stage2", type=float, default=0.03)
    p_train.add_argument("--lr_stage3", type=float, default=0.03)

    p_train.add_argument("--ce_weight", type=float, default=1.0)
    p_train.add_argument("--fr_kl_weight", type=float, default=0.55)
    p_train.add_argument("--en_kl_weight", type=float, default=1.00)
    p_train.add_argument("--fr_token_margin_weight", type=float, default=1.10)
    p_train.add_argument("--fr_traj_close_weight", type=float, default=0.08)
    p_train.add_argument("--fr_traj_sep_weight", type=float, default=0.06)
    p_train.add_argument("--fr_traj_mono_weight", type=float, default=0.00)
    p_train.add_argument("--en_traj_preserve_weight", type=float, default=0.04)
    p_train.add_argument("--en_traj_mono_weight", type=float, default=0.00)

    p_train.add_argument("--stage1_traj_max_scale", type=float, default=0.10)
    p_train.add_argument("--stage1_traj_warmup", type=int, default=4)
    p_train.add_argument("--stage1_traj_cold_scale", type=float, default=0.00)
    p_train.add_argument("--stage2_traj_max_scale", type=float, default=0.08)
    p_train.add_argument("--stage2_traj_warmup", type=int, default=4)
    p_train.add_argument("--stage2_traj_cold_scale", type=float, default=0.00)
    p_train.add_argument("--unlock_eng_minus_fr", type=float, default=0.60)
    p_train.add_argument("--controller_mode", type=str, choices=["free", "basis", "basis_plus_residual"], default="free")
    p_train.add_argument("--free_controller_init", type=str, choices=["none", "mean_boundary"], default="mean_boundary")
    p_train.add_argument("--free_controller_init_scale", type=float, default=0.1)
    p_train.add_argument("--basis_path", type=str, default=None)
    p_train.add_argument("--num_controller_experts", type=int, default=DEFAULT_NUM_CONTROLLER_EXPERTS)
    p_train.add_argument("--disable_expert_router", action="store_true")
    p_train.add_argument("--router_hidden_dim", type=int, default=64)
    p_train.add_argument("--router_temperature", type=float, default=1.0)
    p_train.add_argument("--router_topk", type=int, default=2)
    p_train.add_argument("--router_warm_start_steps", type=int, default=64)
    p_train.add_argument("--router_warm_start_lr", type=float, default=5e-2)
    p_train.add_argument("--router_bucket_smoothing", type=float, default=0.05)
    p_train.add_argument("--stage1_route_init", type=str, choices=["auto", "bucket_prior", "uniform", "boundary_kmeans"], default="auto")
    p_train.add_argument("--stage1_bucket_route_smoothing", type=float, default=0.05)
    p_train.add_argument("--controller_coeff_l2", type=float, default=1e-4)
    p_train.add_argument("--route_entropy_weight", type=float, default=0.01)
    p_train.add_argument("--route_balance_weight", type=float, default=0.05)

    p_train.add_argument("--traj_prob_weight", type=float, default=1.0)
    p_train.add_argument("--traj_logit_weight", type=float, default=0.30)
    p_train.add_argument("--traj_state_weight", type=float, default=0.45)
    p_train.add_argument("--fr_improvement_target", type=float, default=0.35)
    p_train.add_argument("--fr_baseline_floor", type=float, default=0.05)
    p_train.add_argument("--en_preserve_tolerance", type=float, default=0.04)
    p_train.add_argument("--en_preserve_scale", type=float, default=0.22)
    p_train.add_argument("--traj_sep_margin", type=float, default=0.08)
    p_train.add_argument("--traj_mono_tol", type=float, default=0.02)
    p_train.add_argument("--onramp_strong_steps", type=int, default=8)
    p_train.add_argument("--onramp_medium_steps", type=int, default=16)
    p_train.add_argument("--onramp_strong_weight", type=float, default=5.0)
    p_train.add_argument("--onramp_medium_weight", type=float, default=2.5)
    p_train.add_argument("--onramp_tail_weight", type=float, default=1.0)
    p_train.add_argument("--traj_onramp_strong_weight", type=float, default=3.0)
    p_train.add_argument("--traj_onramp_medium_weight", type=float, default=1.75)
    p_train.add_argument("--traj_onramp_tail_weight", type=float, default=1.0)
    p_train.add_argument("--semantic_content_weight", type=float, default=0.08)
    p_train.add_argument("--semantic_mode_weight", type=float, default=0.03)
    p_train.add_argument("--semantic_length_weight", type=float, default=0.02)
    p_train.add_argument("--semantic_refusal_weight", type=float, default=0.04)
    p_train.add_argument("--semantic_temperature", type=float, default=0.20)
    p_train.add_argument("--consistency_weight", type=float, default=0.05)
    p_train.add_argument("--consistency_logit_weight", type=float, default=1.0)
    p_train.add_argument("--consistency_state_weight", type=float, default=0.5)
    p_train.add_argument("--consistency_gate_raw_weight", type=float, default=0.5)
    p_train.add_argument("--consistency_gate_prob_weight", type=float, default=0.5)
    p_train.add_argument("--consistency_route_weight", type=float, default=0.5)
    p_train.add_argument("--consistency_steps", type=int, default=12)
    p_train.add_argument("--consistency_max_variants", type=int, default=1)
    p_train.add_argument("--consistency_eps", type=float, default=1e-6)

    p_train.add_argument("--prefix_focus_steps", type=int, default=8)
    p_train.add_argument("--early_focus_steps", type=int, default=6)
    p_train.add_argument("--prefix_focus_boost", type=float, default=2.5)
    p_train.add_argument("--early_focus_boost", type=float, default=10.0)
    p_train.add_argument("--prefix_decay_tau", type=float, default=8.0)
    p_train.add_argument("--early_decay_tau", type=float, default=3.0)
    p_train.add_argument("--traj_focus_steps", type=int, default=8)
    p_train.add_argument("--traj_focus_boost", type=float, default=5.0)
    p_train.add_argument("--traj_decay_tau", type=float, default=4.0)
    p_train.add_argument("--token_margin", type=float, default=1.25)
    p_train.add_argument("--metric_steps", type=int, default=8)
    p_train.add_argument("--fr_early_ce_weight", type=float, default=0.85)
    p_train.add_argument("--fr_early_kl_weight", type=float, default=0.45)
    p_train.add_argument("--en_early_kl_weight", type=float, default=0.30)
    p_train.add_argument("--state_late_boost", type=float, default=0.75)
    p_train.add_argument("--disable_bucket_batches", action="store_true")
    p_train.add_argument("--disable_hard_negatives", action="store_true")

    p_train.add_argument("--gate_bce_weight", type=float, default=0.75)
    p_train.add_argument("--gate_rank_weight", type=float, default=0.80)
    p_train.add_argument("--gate_rank_margin", type=float, default=2.0)
    p_train.add_argument("--gate_sat_weight", type=float, default=0.12)
    p_train.add_argument("--gate_type", type=str, choices=["anchored_linear", "anchored_mlp"], default="anchored_mlp")
    p_train.add_argument("--gate_hidden_dim", type=int, default=64)
    p_train.add_argument("--gate_warm_start_steps", type=int, default=64)
    p_train.add_argument("--gate_warm_start_lr", type=float, default=5e-2)
    p_train.add_argument("--gate_sharpness", type=float, default=12.0)
    p_train.add_argument("--learn_direction_residual", action="store_true")
    p_train.add_argument("--gate_residual_scale", type=float, default=0.05)
    p_train.add_argument("--gate_residual_l2", type=float, default=1e-3)
    p_train.add_argument("--gate_layer_l2", type=float, default=1e-3)
    p_train.add_argument("--gate_sharpness_l2", type=float, default=1e-3)
    p_train.add_argument("--gate_reg_weight", type=float, default=1.0)
    p_train.add_argument("--stage1_fr_gate", type=float, default=1.0)
    p_train.add_argument("--stage2_pos_floor", type=float, default=0.65)
    p_train.add_argument("--stage2_neg_cap", type=float, default=0.15)

    p_train.add_argument("--calib_sep_reward", type=float, default=0.50)
    p_train.add_argument("--calib_utility_scale", type=float, default=1.50)
    p_train.add_argument("--calib_gate_margin", type=float, default=2.0)
    p_train.add_argument("--calib_gate_rank_weight", type=float, default=0.90)
    p_train.add_argument("--calib_gate_sat_weight", type=float, default=0.30)

    p_train.add_argument("--vocab_proj_dim", type=int, default=64)
    p_train.add_argument("--state_proj_dim", type=int, default=64)
    p_train.add_argument("--l2", type=float, default=1e-4)
    p_train.add_argument("--layer_smooth", type=float, default=5e-4)
    p_train.add_argument("--time_smooth", type=float, default=5e-4)
    p_train.add_argument("--layer_ramp_start", type=float, default=0.65)
    p_train.add_argument("--layer_ramp_end", type=float, default=1.10)
    p_train.add_argument("--time_decay_tau", type=float, default=12.0)
    p_train.add_argument("--post_smooth_layer", type=float, default=0.15)
    p_train.add_argument("--post_smooth_time", type=float, default=0.10)

    p_train.add_argument("--preview_fr_prompt", type=str, default=None)
    p_train.add_argument("--preview_en_prompt", type=str, default=None)
    p_train.add_argument("--preview_test_fr_prompt", type=str, default=None)
    p_train.add_argument("--preview_test_en_prompt", type=str, default=None)
    p_train.add_argument("--preview_hard_gate", action="store_true")
    p_train.add_argument("--intervention_positions", type=str, choices=["last_token", "all_positions"], default="last_token")
    p_train.add_argument("--max_new_tokens", type=int, default=128)
    p_train.add_argument("--strength", type=float, default=1.0)
    p_train.add_argument("--temperature", type=float, default=0.0)
    p_train.add_argument("--top_p", type=float, default=0.95)
    p_train.add_argument("--save_path", type=str, default="gemma3_fr_en_binary_merged.pt")
    p_train.add_argument("--seed", type=int, default=0)

    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--load_path", type=str, required=True)
    p_infer.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p_infer.add_argument("--quantization", type=str, choices=["none", "8bit", "4bit"], default="none")
    p_infer.add_argument("--quant_compute_dtype", type=str, choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"], default="auto")
    p_infer.add_argument("--prompt", type=str, required=True)
    p_infer.add_argument("--hard_gate", action="store_true")
    p_infer.add_argument("--max_new_tokens", type=int, default=128)
    p_infer.add_argument("--strength", type=float, default=1.0)
    p_infer.add_argument("--temperature", type=float, default=0.0)
    p_infer.add_argument("--top_p", type=float, default=0.95)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train_binary_merged(args)
    elif args.command == "infer":
        infer_binary_merged(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
