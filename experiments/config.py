"""Experiment configuration dataclass shared across all 7 runs."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Data locations are portable. Override these with environment
# variables, or place the corresponding public-source files under data/raw/.
SALAD_PATH = os.environ.get(
    "RP_SALAD_PATH",
    "data/raw/salad_bench.json",
)
ALPACA_PATH = os.environ.get(
    "RP_ALPACA_PATH",
    "data/raw/alpaca.json",
)
HARMBENCH_PATH = os.environ.get(
    "RP_HARMBENCH_PATH",
    "data/raw/harmbench_val.csv",
)
HARMBENCH_TEST_PATH = os.environ.get(
    "RP_HARMBENCH_TEST_PATH",
    "data/raw/harmbench_test.csv",
)


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    model_id: str = "google/gemma-3-4b-it"
    model_quantization: str = "none"  # none, 8bit, or 4bit
    model_quant_compute_dtype: str = "auto"

    # ── Dataset ──────────────────────────────────────────────────────────────
    n_edit: int = 50        # SaladBench harmful (edit target, gate=1)
    n_keep_benign: int = 30  # Alpaca benign keep (gate=0, preserve)
    n_keep_harmful: int = 30 # HarmBench truly harmful keep (gate=0, must still refuse)
    harmbench_paths: str = HARMBENCH_PATH  # comma-separated harmful-keep sources
    data_seed: int = 42
    steer_steps: int = 8    # generation steps to force during training
    eval_fraction: float = 0.2
    eval_seed: int = 17
    eval_max_new_tokens: int = 96
    build_batch_size: int = 8
    # Current default preserves legacy behavior: random benign Alpaca anchors.
    # "safe_reframe" builds teacher prompts from the same edit prompt with a
    # high-level safety framing instruction.
    edit_target_mode: str = "paired_benign"
    # Optional deterministic prefix prepended to edit anchor tokens. Useful for
    # making safe_reframe targets start in a non-refusal, non-operational basin.
    edit_anchor_prefix: str = ""
    # Optional explicit edit-prompt source. When empty, edit prompts are loaded
    # from SALAD_PATH. This is useful for prefiltered base-refusal pools.
    edit_prompts_path: str = ""
    edit_prompts_prefiltered_base_refusal: bool = False
    # Optional diagnostic filter: keep only edit prompts where the frozen base
    # model produces a refusal under the local heuristic.
    require_edit_base_refusal: bool = False
    edit_refusal_filter_oversample: int = 5
    edit_refusal_filter_max_new_tokens: int = 64
    # Cache built BinaryExample splits, including anchor generations and router
    # boundary states. The cache key includes the model, data seeds, sample
    # counts, edit-target settings, HarmBench sources, and router layer set.
    example_cache_dir: str = "experiments/cache/examples"
    example_cache_refresh: bool = False

    # ── Architecture ─────────────────────────────────────────────────────────
    layer_spec: str = "-4,-3,-2,-1"
    gate_sharpness: float = 8.0
    gate_learn_residual: bool = False
    gate_residual_scale: float = 0.05
    controller_warm_start: str = "mean_boundary"
    controller_warm_start_scale: float = 0.1

    # ── Training schedule ────────────────────────────────────────────────────
    stage1_epochs: int = 8   # hard gate, CE+KL only
    stage2_epochs: int = 12  # soft gate, + trajectory
    stage3_epochs: int = 5   # gate calibration, controller frozen
    stage1_train_steps: int = 1
    stage2_train_steps: int = 4
    stage2_gate_warmup_epochs: int = 2
    lr: float = 3e-4
    gate_lr: float = 1e-3
    trace_batch_size: int = 2  # examples per forward-pass batch

    # ── Loss weights ─────────────────────────────────────────────────────────
    ce_weight: float = 1.0
    kl_weight: float = 0.35
    traj_weight: float = 0.5
    traj_warmup_epochs: int = 4  # ramp traj_weight over this many stage-2 epochs
    pres_weight: float = 0.75
    gate_bce_weight: float = 0.2
    gate_margin: float = 2.0
    # Optional MoE-only opening contrast: compare the safe edit anchor opening
    # against short refusal-style openings under the same intervention.
    moe_opening_contrast_weight: float = 0.0
    moe_opening_contrast_steps: int = 4
    moe_opening_contrast_margin: float = 0.5
    moe_opening_refusal_phrases: str = "I'm sorry|I understand"

    # ── Regularisation ───────────────────────────────────────────────────────
    l2_weight: float = 1e-4
    layer_smooth: float = 0.0
    time_smooth: float = 0.0
    layer_ramp_start: float = 0.5
    layer_ramp_end: float = 1.0
    time_decay_tau: float = 0.0  # 0 = no decay

    # ── Ablation knobs (defaults = off / no-op) ───────────────────────────────
    # triplet_traj
    triplet_weight: float = 0.0   # >0 enables triplet trajectory loss
    triplet_margin: float = 0.1

    # state_cond
    state_cond_rank: int = 0      # >0 enables state-conditional delta

    # two_signal_gate
    benefit_hidden_dim: int = 0   # >0 enables two-signal gate

    # scheduled_sample
    on_policy_start_epoch: int = 9999  # stage-2 epoch after which on-policy kicks in
    on_policy_fraction: float = 0.5

    # three_tier_keep
    harmful_pres_weight: float = 0.0  # >0 enables three-tier keep loss
    harmful_gate_margin_weight: float = 0.0  # >0 adds harmful-keep gate logit hinge
    harmful_gate_logit_target: float = 0.0
    harmful_gate_pair_margin_weight: float = 0.0  # >0 adds FR-vs-harmful gate margin
    harmful_gate_pair_match: str = "cyclic"  # cyclic, boundary_cosine, or hard_topk
    harmful_gate_hard_neg_k: int = 5  # top-k violating FR/HK pairs for hard_topk mode
    harmful_expert0_avoid_weight: float = 0.0  # >0 discourages harmful keeps from edit expert 0
    edit_expert0_attract_weight: float = 0.0  # >0 encourages edit examples toward edit expert 0
    harmful_veto_enabled: bool = False  # fit a boundary-state veto that can zero harmful-keep-like gates
    harmful_veto_max_iter: int = 100
    harmful_veto_l2_weight: float = 0.0
    harmful_veto_threshold_tie_break: str = "low"  # low preserves existing behavior; high is edit-conservative
    harmful_veto_mode: str = "hard"  # hard zeroes above threshold; soft applies a sigmoid multiplier
    harmful_veto_temperature: float = 1.0
    harmful_veto_min_multiplier: float = 0.0
    harmful_veto_ensemble_l2_values: str = ""  # comma-separated L2 values; empty uses one linear head

    # paired_basis_init
    basis_rank: int = 0           # >0 enables SVD basis init for controller

    # reference caches / eval
    ref_topk: int = 1024

    # ── MoE architecture ──────────────────────────────────────────────────────
    use_moe: bool = False
    moe_num_experts: int = 3
    # Bottleneck dim per expert (H/8 for 4B = 320); set 0 to auto-compute as H/8
    moe_bottleneck_dim: int = 0
    # Expert nonlinearity. "gelu" is the Residual Paving default; "identity"
    # makes the bottleneck expert a low-rank linear residual adapter.
    moe_expert_activation: str = "gelu"
    # Layer specs for routing vs intervention; "auto" for intervention means
    # "all layers from max(router)+1 to total-1"
    router_layer_spec: str = "1,2,3,4"
    intervention_layer_spec: str = "auto"
    # Expert Gaussian centre/sigma as comma-separated fractions of [0,1]
    expert_center_fracs: str = "0.3,0.55,0.8"
    expert_sigma_fracs: str = "0.15,0.20,0.15"
    # Optional post-clip expert gain initializers. Empty means all ones.
    expert_gain_inits: str = ""
    max_expert_gain: float = 8.0
    # Default scalar applied to MoE gate strengths at evaluation/generation time.
    moe_intervention_scale: float = 1.0
    # How expert weights are resolved during MoE training/evaluation:
    # "router" uses the learned expert head; "uniform" activates all experts
    # through their layer masks; "expert<N>" forces a single expert.
    moe_train_expert_policy: str = "router"
    moe_eval_expert_policy: str = "router"
    # MoE router hidden dim
    router_hidden_dim: int = 64
    # Optional eval-only diagnostic: seed generation with edit_anchor_prefix as
    # an assistant response prefill to separate opening-basin failures from
    # continuation failures. Disabled by default because it changes generation.
    moe_eval_prefill_anchor_prefix: bool = False
    # Optional training-time qualitative previews. When enabled, the MoE harness
    # prints one edit, benign-keep, and harmful-keep example after setup and
    # training stages so the current controller behavior can be inspected.
    print_eval_examples: bool = False
    print_eval_examples_policy: str = "soft"
    print_eval_examples_scale: float = 0.0  # <=0 means use moe_intervention_scale
    print_eval_examples_max_new_tokens: int = 0  # <=0 means use eval_max_new_tokens

    # ── Contrastive Stage 0 ───────────────────────────────────────────────────
    contrastive_epochs: int = 0          # 0 = skip Stage 0
    contrastive_lr: float = 3e-4
    # Weight on -KL(base||steered) for targeted prompts (maximise divergence)
    contrastive_diverge_weight: float = 1.0
    # Weight on +KL(base||steered) for benign prompts (preserve)
    contrastive_preserve_weight: float = 0.5
    # Gate head pre-training epochs (BCE on boundary states, no expert traces)
    gate_pretrain_epochs: int = 0        # 0 = skip; MoE factory sets 10
