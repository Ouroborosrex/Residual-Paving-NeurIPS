"""Run one tiny MoE training job and evaluate gate/expert diagnostic policies."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Match the main experiment runner's CUDA allocator workaround without
# changing PyTorch's process-global allocator configuration after import.
if "torch" not in sys.modules:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
for _d in [str(_EXP_DIR), str(_ROOT / "src" / "trainer"), str(_ROOT / "src" / "optimized")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from moe_harness import make_moe_experiment, moe_evaluate


POLICIES = ("soft", "thresholded_soft", "hard", "oracle")
EXPERT_PATHS = (
    ("oracle_router", "oracle", "router", 1.0),
    ("oracle_uniform", "oracle", "uniform", 1.0),
    ("oracle_expert0", "oracle", "expert0", 1.0),
    ("oracle_expert0_x4", "oracle", "expert0", 4.0),
)
SCALE_SWEEP_PATHS = (
    ("soft_router_x0.5", "soft", "router", 0.5),
    ("soft_router_x1", "soft", "router", 1.0),
    ("soft_router_x2", "soft", "router", 2.0),
    ("oracle_router_x0.5", "oracle", "router", 0.5),
    ("oracle_router_x1", "oracle", "router", 1.0),
    ("oracle_router_x2", "oracle", "router", 2.0),
    ("oracle_expert0_x0.5", "oracle", "expert0", 0.5),
    ("oracle_expert0_x1", "oracle", "expert0", 1.0),
    ("oracle_expert0_x2", "oracle", "expert0", 2.0),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="",
        help="Optional Hugging Face model id override. Empty uses the experiment default.",
    )
    parser.add_argument(
        "--model-quantization",
        choices=("none", "8bit", "4bit"),
        default="none",
        help="Optional quantized model loading mode for larger backbones.",
    )
    parser.add_argument(
        "--model-quant-compute-dtype",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
        default="auto",
        help="Compute dtype used by quantized model loading.",
    )
    parser.add_argument("--n", type=int, default=4, help="Default count for edit, benign_keep, and harmful_keep.")
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Optional data_seed override for multi-seed robustness runs. Defaults to config.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Optional eval_seed override for multi-seed robustness runs. Defaults to config.",
    )
    parser.add_argument(
        "--n-edit",
        type=int,
        default=None,
        help="Optional edit count override. Defaults to --n.",
    )
    parser.add_argument(
        "--n-keep-benign",
        type=int,
        default=None,
        help="Optional benign_keep count override. Defaults to --n.",
    )
    parser.add_argument(
        "--n-keep-harmful",
        type=int,
        default=None,
        help="Optional harmful_keep count override. Defaults to --n.",
    )
    parser.add_argument("--stage1-epochs", type=int, default=1)
    parser.add_argument("--stage2-epochs", type=int, default=2)
    parser.add_argument("--stage3-epochs", type=int, default=1)
    parser.add_argument("--contrastive-epochs", type=int, default=1)
    parser.add_argument("--gate-pretrain-epochs", type=int, default=10)
    parser.add_argument("--steer-steps", type=int, default=5)
    parser.add_argument("--stage1-train-steps", type=int, default=1)
    parser.add_argument("--stage2-train-steps", type=int, default=4)
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=None,
        help="Optional internal train/eval split fraction override.",
    )
    parser.add_argument("--eval-max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--skip-post-eval",
        action="store_true",
        help=(
            "Save the trained checkpoint and skip the built-in policy evaluation. "
            "Use this when a separate held-out export/judge pipeline will evaluate the checkpoint."
        ),
    )
    parser.add_argument("--trace-batch-size", type=int, default=1)
    parser.add_argument("--build-batch-size", type=int, default=2)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.35)
    parser.add_argument("--traj-weight", type=float, default=0.5)
    parser.add_argument("--pres-weight", type=float, default=0.75)
    parser.add_argument("--harmful-pres-weight", type=float, default=0.0)
    parser.add_argument("--harmful-gate-margin-weight", type=float, default=0.0)
    parser.add_argument("--harmful-gate-logit-target", type=float, default=0.0)
    parser.add_argument("--harmful-gate-pair-margin-weight", type=float, default=0.0)
    parser.add_argument(
        "--harmful-gate-pair-match",
        choices=("cyclic", "boundary_cosine", "hard_topk"),
        default="cyclic",
        help="How to pair FR logits against harmful-keep logits for harmful gate margins.",
    )
    parser.add_argument(
        "--harmful-gate-hard-neg-k",
        type=int,
        default=5,
        help="Top-k violating FR/HK pairs to use when --harmful-gate-pair-match=hard_topk.",
    )
    parser.add_argument(
        "--harmful-expert0-avoid-weight",
        type=float,
        default=0.0,
        help="Weight for discouraging harmful-keep examples from using high-gain edit expert 0.",
    )
    parser.add_argument(
        "--edit-expert0-attract-weight",
        type=float,
        default=0.0,
        help="Weight for encouraging edit examples to use high-gain edit expert 0.",
    )
    parser.add_argument(
        "--harmful-veto",
        action="store_true",
        help="Fit and apply a cached-boundary harmful-keep veto to learned gate strengths.",
    )
    parser.add_argument(
        "--harmful-veto-l2-weight",
        type=float,
        default=0.0,
        help="L2 weight for the harmful-veto linear head.",
    )
    parser.add_argument(
        "--harmful-veto-threshold-tie-break",
        choices=("low", "high"),
        default="low",
        help=(
            "Tie-break for equal-accuracy harmful-veto thresholds. "
            "'low' preserves prior behavior; 'high' is more conservative about edit false positives."
        ),
    )
    parser.add_argument(
        "--harmful-veto-mode",
        choices=("hard", "soft"),
        default="hard",
        help=(
            "How the harmful-veto logit changes learned gate strengths. "
            "'hard' zeroes gates above threshold; 'soft' uses a sigmoid multiplier."
        ),
    )
    parser.add_argument("--harmful-veto-temperature", type=float, default=1.0)
    parser.add_argument("--harmful-veto-min-multiplier", type=float, default=0.0)
    parser.add_argument(
        "--harmful-veto-ensemble-l2-values",
        default="",
        help=(
            "Optional comma-separated L2 weights for an ensemble of harmful-veto "
            "linear heads. Empty uses the single-head path."
        ),
    )
    parser.add_argument("--gate-bce-weight", type=float, default=0.2)
    parser.add_argument("--moe-opening-contrast-weight", type=float, default=0.0)
    parser.add_argument("--moe-opening-contrast-steps", type=int, default=4)
    parser.add_argument("--moe-opening-contrast-margin", type=float, default=0.5)
    parser.add_argument(
        "--moe-opening-refusal-phrases",
        default="I'm sorry|I understand",
        help="Pipe-separated refusal-style openings for the optional opening contrast loss.",
    )
    parser.add_argument(
        "--edit-target-mode",
        choices=("paired_benign", "safe_reframe"),
        default="paired_benign",
    )
    parser.add_argument(
        "--edit-anchor-prefix",
        default="",
        help="Optional deterministic prefix prepended to edit target anchor tokens.",
    )
    parser.add_argument(
        "--eval-prefill-anchor-prefix",
        action="store_true",
        help="Also evaluate edit generations with edit_anchor_prefix prefilled as assistant text.",
    )
    parser.add_argument(
        "--edit-prompts-path",
        default="",
        help=(
            "Optional explicit edit-prompt JSON/JSONL source. "
            "Rows may use instruction/prompt/question/Behavior fields."
        ),
    )
    parser.add_argument(
        "--edit-prompts-prefiltered-base-refusal",
        action="store_true",
        help=(
            "Treat --edit-prompts-path as already filtered to base-refusal prompts "
            "and skip the runtime base-refusal filter."
        ),
    )
    parser.add_argument(
        "--require-edit-base-refusal",
        action="store_true",
        help="Filter edit prompts to cases where the base model refuses before train/eval split.",
    )
    parser.add_argument("--edit-refusal-filter-oversample", type=int, default=5)
    parser.add_argument("--edit-refusal-filter-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--example-cache-dir",
        default=None,
        help=(
            "Optional override for cached BinaryExample split directory. "
            "Use an empty string to disable example caching."
        ),
    )
    parser.add_argument(
        "--refresh-example-cache",
        action="store_true",
        help="Rebuild and overwrite the keyed BinaryExample split cache.",
    )
    parser.add_argument(
        "--harmbench-paths",
        default="",
        help="Optional comma-separated harmful-keep source paths.",
    )
    parser.add_argument(
        "--expert-gain-inits",
        default="",
        help="Comma-separated post-clip expert gain initializers; empty uses config defaults.",
    )
    parser.add_argument("--max-expert-gain", type=float, default=8.0)
    parser.add_argument(
        "--moe-num-experts",
        type=int,
        default=None,
        help="Optional override for number of residual experts.",
    )
    parser.add_argument(
        "--moe-bottleneck-dim",
        type=int,
        default=None,
        help="Optional override for expert bottleneck dimension; 0 keeps the config auto value.",
    )
    parser.add_argument(
        "--moe-expert-activation",
        choices=("gelu", "identity"),
        default=None,
        help="Expert activation. identity gives a low-rank linear residual adapter.",
    )
    parser.add_argument(
        "--router-hidden-dim",
        type=int,
        default=None,
        help="Optional override for the learned router hidden dimension.",
    )
    parser.add_argument(
        "--expert-center-fracs",
        default="",
        help="Optional comma-separated expert layer-mask centers.",
    )
    parser.add_argument(
        "--expert-sigma-fracs",
        default="",
        help="Optional comma-separated expert layer-mask sigmas.",
    )
    parser.add_argument(
        "--moe-intervention-scale",
        type=float,
        default=1.0,
        help="Default MoE gate-strength multiplier used by regular policy evaluation.",
    )
    parser.add_argument(
        "--train-expert-policy",
        default="router",
        help="Expert policy used during MoE training: router, uniform, or expert<N>.",
    )
    parser.add_argument(
        "--eval-expert-policy",
        default="router",
        help="Expert policy used for regular policy evaluation and scale calibration.",
    )
    parser.add_argument(
        "--router-layer-spec",
        default="",
        help="Optional override for the MoE router layer spec, e.g. '8,9,10,11'.",
    )
    parser.add_argument(
        "--intervention-layer-spec",
        default="",
        help=(
            "Optional override for the MoE intervention layer spec, e.g. "
            "'16,20,24,28,31'. Empty preserves the config default."
        ),
    )
    parser.add_argument(
        "--include-expert-paths",
        action="store_true",
        help="Also evaluate oracle-gated expert routing/scale diagnostic paths.",
    )
    parser.add_argument(
        "--include-scale-sweep",
        action="store_true",
        help="Evaluate a fixed small gate-scale sweep for soft/router, oracle/router, and oracle/expert0.",
    )
    parser.add_argument(
        "--calibrate-scale",
        action="store_true",
        help="Pick the regular soft/router intervention scale on a calibration slice before final eval.",
    )
    parser.add_argument(
        "--scale-candidates",
        default="0.5,1.0,2.0",
        help="Comma-separated MoE intervention scales to consider when --calibrate-scale is set.",
    )
    parser.add_argument(
        "--scale-calibration-fraction",
        type=float,
        default=0.5,
        help="Fraction of available eval examples per split to reserve for scale calibration.",
    )
    parser.add_argument(
        "--print-eval-examples",
        action="store_true",
        help=(
            "During training, print one edit, benign-keep, and harmful-keep "
            "generation preview after setup and each training stage/epoch."
        ),
    )
    parser.add_argument(
        "--print-eval-examples-policy",
        choices=POLICIES,
        default="soft",
        help="Gate policy used for training-time qualitative eval previews.",
    )
    parser.add_argument(
        "--print-eval-examples-scale",
        type=float,
        default=0.0,
        help=(
            "Gate scale used for training-time qualitative eval previews. "
            "Values <=0 use --moe-intervention-scale."
        ),
    )
    parser.add_argument(
        "--print-eval-examples-max-new-tokens",
        type=int,
        default=0,
        help=(
            "Max generated tokens per preview example. Values <=0 use "
            "--eval-max-new-tokens."
        ),
    )
    return parser.parse_args()


def _smoke_overrides(args: argparse.Namespace) -> dict:
    n_edit = int(args.n if args.n_edit is None else args.n_edit)
    n_keep_benign = int(args.n if args.n_keep_benign is None else args.n_keep_benign)
    n_keep_harmful = int(args.n if args.n_keep_harmful is None else args.n_keep_harmful)
    if min(n_edit, n_keep_benign, n_keep_harmful) < 0:
        raise ValueError(
            "Example counts must be non-negative: "
            f"n_edit={n_edit}, n_keep_benign={n_keep_benign}, "
            f"n_keep_harmful={n_keep_harmful}"
        )
    overrides = {
        "name": "moe_gate_policy_smoke",
        "n_edit": n_edit,
        "n_keep_benign": n_keep_benign,
        "n_keep_harmful": n_keep_harmful,
        "stage1_epochs": int(args.stage1_epochs),
        "stage2_epochs": int(args.stage2_epochs),
        "stage3_epochs": int(args.stage3_epochs),
        "steer_steps": int(args.steer_steps),
        "stage1_train_steps": int(args.stage1_train_steps),
        "stage2_train_steps": int(args.stage2_train_steps),
        "trace_batch_size": int(args.trace_batch_size),
        "build_batch_size": int(args.build_batch_size),
        "eval_max_new_tokens": int(args.eval_max_new_tokens),
        "contrastive_epochs": int(args.contrastive_epochs),
        "gate_pretrain_epochs": int(args.gate_pretrain_epochs),
        "ce_weight": float(args.ce_weight),
        "kl_weight": float(args.kl_weight),
        "traj_weight": float(args.traj_weight),
        "pres_weight": float(args.pres_weight),
        "harmful_pres_weight": float(args.harmful_pres_weight),
        "harmful_gate_margin_weight": float(args.harmful_gate_margin_weight),
        "harmful_gate_logit_target": float(args.harmful_gate_logit_target),
        "harmful_gate_pair_margin_weight": float(args.harmful_gate_pair_margin_weight),
        "harmful_gate_pair_match": str(args.harmful_gate_pair_match),
        "harmful_gate_hard_neg_k": int(args.harmful_gate_hard_neg_k),
        "harmful_expert0_avoid_weight": float(args.harmful_expert0_avoid_weight),
        "edit_expert0_attract_weight": float(args.edit_expert0_attract_weight),
        "harmful_veto_enabled": bool(args.harmful_veto),
        "harmful_veto_l2_weight": float(args.harmful_veto_l2_weight),
        "harmful_veto_threshold_tie_break": str(args.harmful_veto_threshold_tie_break),
        "harmful_veto_mode": str(args.harmful_veto_mode),
        "harmful_veto_temperature": float(args.harmful_veto_temperature),
        "harmful_veto_min_multiplier": float(args.harmful_veto_min_multiplier),
        "harmful_veto_ensemble_l2_values": str(args.harmful_veto_ensemble_l2_values),
        "gate_bce_weight": float(args.gate_bce_weight),
        "moe_opening_contrast_weight": float(args.moe_opening_contrast_weight),
        "moe_opening_contrast_steps": int(args.moe_opening_contrast_steps),
        "moe_opening_contrast_margin": float(args.moe_opening_contrast_margin),
        "moe_opening_refusal_phrases": str(args.moe_opening_refusal_phrases),
        "edit_target_mode": str(args.edit_target_mode),
        "edit_anchor_prefix": str(args.edit_anchor_prefix),
        "moe_eval_prefill_anchor_prefix": bool(args.eval_prefill_anchor_prefix),
        "edit_prompts_path": str(args.edit_prompts_path).strip(),
        "edit_prompts_prefiltered_base_refusal": bool(
            args.edit_prompts_prefiltered_base_refusal
        ),
        "require_edit_base_refusal": bool(args.require_edit_base_refusal),
        "edit_refusal_filter_oversample": int(args.edit_refusal_filter_oversample),
        "edit_refusal_filter_max_new_tokens": int(args.edit_refusal_filter_max_new_tokens),
        "example_cache_refresh": bool(args.refresh_example_cache),
        "harmbench_paths": str(args.harmbench_paths).strip() or None,
        "expert_gain_inits": str(args.expert_gain_inits),
        "max_expert_gain": float(args.max_expert_gain),
        "moe_intervention_scale": float(args.moe_intervention_scale),
        "moe_train_expert_policy": str(args.train_expert_policy),
        "moe_eval_expert_policy": str(args.eval_expert_policy),
        "print_eval_examples": bool(args.print_eval_examples),
        "print_eval_examples_policy": str(args.print_eval_examples_policy),
        "print_eval_examples_scale": float(args.print_eval_examples_scale),
        "print_eval_examples_max_new_tokens": int(
            args.print_eval_examples_max_new_tokens
        ),
        "model_quantization": str(args.model_quantization),
        "model_quant_compute_dtype": str(args.model_quant_compute_dtype),
    }
    if str(args.model_id).strip():
        overrides["model_id"] = str(args.model_id).strip()
    if args.eval_fraction is not None:
        overrides["eval_fraction"] = float(args.eval_fraction)
    if str(args.router_layer_spec).strip():
        overrides["router_layer_spec"] = str(args.router_layer_spec).strip()
    if str(args.intervention_layer_spec).strip():
        overrides["intervention_layer_spec"] = str(args.intervention_layer_spec).strip()
    if args.moe_num_experts is not None:
        overrides["moe_num_experts"] = int(args.moe_num_experts)
    if args.moe_bottleneck_dim is not None:
        overrides["moe_bottleneck_dim"] = int(args.moe_bottleneck_dim)
    if args.moe_expert_activation is not None:
        overrides["moe_expert_activation"] = str(args.moe_expert_activation)
    if args.router_hidden_dim is not None:
        overrides["router_hidden_dim"] = int(args.router_hidden_dim)
    if str(args.expert_center_fracs).strip():
        overrides["expert_center_fracs"] = str(args.expert_center_fracs).strip()
    if str(args.expert_sigma_fracs).strip():
        overrides["expert_sigma_fracs"] = str(args.expert_sigma_fracs).strip()
    if args.example_cache_dir is not None:
        overrides["example_cache_dir"] = str(args.example_cache_dir)
    if getattr(args, "data_seed", None) is not None:
        overrides["data_seed"] = int(args.data_seed)
    if getattr(args, "eval_seed", None) is not None:
        overrides["eval_seed"] = int(args.eval_seed)
    return overrides


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _write_run_note(run_root: Path, summary: dict, overrides: dict) -> None:
    lines = [
        "# MoE Gate Policy Smoke",
        "",
        f"- command: `{summary['command']}`",
        f"- timestamp: `{summary['timestamp']}`",
        f"- git_commit: `{summary['git_commit']}`",
        f"- status: `{summary['status']}`",
        f"- policies: `{', '.join(POLICIES)}`",
        f"- overrides: `{json.dumps(overrides, sort_keys=True)}`",
    ]
    if summary.get("seeds"):
        lines.append(f"- seeds: `{json.dumps(summary['seeds'], sort_keys=True)}`")
    if summary.get("expert_paths"):
        lines.append(f"- expert_paths: `{', '.join(summary['expert_paths'])}`")
    if summary.get("scale_sweep"):
        lines.append(f"- scale_sweep: `{', '.join(summary['scale_sweep'])}`")
    if summary.get("scale_calibration"):
        calibration = summary["scale_calibration"]
        lines.append(
            "- scale_calibration: "
            f"`selected={calibration.get('selected_scale')} "
            f"score={calibration.get('selected_score')}`"
        )
    if summary.get("anomalies"):
        lines.append(f"- anomalies: `{'; '.join(summary['anomalies'])}`")
    if "elapsed_s" in summary:
        lines.append(f"- elapsed_s: `{summary['elapsed_s']:.2f}`")
    run_root.joinpath("run_note.md").write_text("\n".join(lines) + "\n")


def _format_scale(scale: float) -> str:
    return f"{float(scale):g}"


def _parse_scale_candidates(raw: str) -> Tuple[float, ...]:
    candidates = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not candidates:
        raise ValueError("scale_candidates must contain at least one numeric value")
    if any(scale < 0.0 for scale in candidates):
        raise ValueError(f"scale_candidates must be non-negative, got {candidates}")
    return candidates


def _split_examples_for_calibration(examples: List, fraction: float) -> Tuple[List, List, bool]:
    examples = list(examples)
    if len(examples) <= 1:
        return examples, examples, bool(examples)
    frac = min(max(float(fraction), 0.05), 0.95)
    n_calib = int(round(len(examples) * frac))
    n_calib = max(1, min(n_calib, len(examples) - 1))
    return examples[:n_calib], examples[n_calib:], False


def _make_scale_calibration_splits(exp, fraction: float) -> Tuple[Dict, Dict, List[str]]:
    anomalies: List[str] = []
    source_fallback = False
    fr_examples = exp.eval_fr_examples or exp.fr_examples
    benign_examples = exp.eval_keep_benign or exp.keep_benign
    harmful_examples = exp.eval_keep_harmful or exp.keep_harmful
    if not exp.eval_fr_examples or not exp.eval_keep_benign or not exp.eval_keep_harmful:
        source_fallback = True

    calib_fr, test_fr, reused_fr = _split_examples_for_calibration(fr_examples, fraction)
    calib_benign, test_benign, reused_benign = _split_examples_for_calibration(
        benign_examples, fraction
    )
    calib_harmful, test_harmful, reused_harmful = _split_examples_for_calibration(
        harmful_examples, fraction
    )
    if source_fallback:
        anomalies.append(
            "scale calibration used train examples for at least one split because eval examples were unavailable"
        )
    if reused_fr or reused_benign or reused_harmful:
        anomalies.append(
            "scale calibration reused calibration examples as test examples for at least one tiny split"
        )
    if not calib_fr:
        anomalies.append("scale calibration has no edit calibration examples")

    calibration_kwargs = {
        "eval_fr_examples": calib_fr,
        "eval_keep_benign": calib_benign,
        "eval_keep_harmful": calib_harmful,
    }
    test_kwargs = {
        "eval_fr_examples": test_fr,
        "eval_keep_benign": test_benign,
        "eval_keep_harmful": test_harmful,
    }
    return calibration_kwargs, test_kwargs, anomalies


def _scale_calibration_score(result: Dict) -> float:
    edit_success = 1.0 - float(result.get("edit_refusal_rate", 1.0))
    operational_compliance = float(
        result.get("edit_operational_compliance_heuristic_rate", 0.0)
    )
    safe_nonrefusal = max(
        0.0,
        float(result.get("edit_nonrefusal_nonoperational_rate", edit_success - operational_compliance)),
    )
    target_alignment = float(result.get("edit_target_alignment", 0.0))
    preservation = float(result.get("keep_preservation", 0.0))
    if result.get("harmful_keep_refusal_delta_vs_base") is not None:
        harmful_preservation = max(
            0.0,
            min(1.0, 1.0 + float(result.get("harmful_keep_refusal_delta_vs_base", -1.0))),
        )
    else:
        harmful_preservation = float(result.get("harmful_keep_refusal_rate", 0.0))
    return (
        safe_nonrefusal
        + 0.10 * target_alignment
        + 0.10 * preservation
        + 0.10 * harmful_preservation
    )


def _select_scale_candidate(candidate_results: Dict[str, Dict]) -> Tuple[float, float]:
    if not candidate_results:
        raise ValueError("Cannot select a scale without candidate results")
    best_key, best_result = max(
        candidate_results.items(),
        key=lambda item: (_scale_calibration_score(item[1]), -float(item[0])),
    )
    return float(best_key), _scale_calibration_score(best_result)


def _any_prefill_enabled(results: Dict[str, Dict]) -> bool:
    return any(
        bool(result.get("edit_prefill_anchor_prefix_enabled", False))
        for result in results.values()
    )


def _print_prefill_summary(title: str, results: Dict[str, Dict]) -> None:
    if not _any_prefill_enabled(results):
        return
    print(f"\n{title}:")
    print(
        f"  {'Path':<20} {'EditRef':>8} {'PrefRef':>8} {'PrefSafe':>8} "
        f"{'PrefOper':>8} {'PrefTok':>8}"
    )
    print(f"  {'-' * 69}")
    for name, result in results.items():
        print(
            f"  {name:<20} "
            f"{result.get('edit_refusal_rate', float('nan')):>8.3f} "
            f"{result.get('edit_prefill_refusal_rate', float('nan')):>8.3f} "
            f"{result.get('edit_prefill_nonrefusal_nonoperational_rate', float('nan')):>8.3f} "
            f"{result.get('edit_prefill_operational_compliance_heuristic_rate', float('nan')):>8.3f} "
            f"{result.get('edit_prefill_anchor_prefix_token_count', float('nan')):>8}"
        )


def _any_opening_rank_metrics(results: Dict[str, Dict]) -> bool:
    return any("edit_anchor_first_token_rank_mean" in result for result in results.values())


def _print_opening_rank_summary(title: str, results: Dict[str, Dict]) -> None:
    if not _any_opening_rank_metrics(results):
        return
    print(f"\n{title}:")
    print(f"  {'Path':<20} {'FirstProb':>10} {'FirstRank':>10} {'Top10':>7}")
    print(f"  {'-' * 53}")
    for name, result in results.items():
        print(
            f"  {name:<20} "
            f"{result.get('edit_anchor_first_token_prob_mean', float('nan')):>10.6f} "
            f"{result.get('edit_anchor_first_token_rank_mean', float('nan')):>10.1f} "
            f"{result.get('edit_anchor_first_token_top10_rate', float('nan')):>7.3f}"
        )


def main() -> None:
    args = _parse_args()
    scale_candidates = (
        _parse_scale_candidates(args.scale_candidates) if args.calibrate_scale else ()
    )
    overrides = _smoke_overrides(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_gate_policy_smoke")
    run_root = _ROOT / "experiments" / "runs" / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    variant_dir = run_root / "moe_gate_policy_smoke"
    command = " ".join([sys.executable] + sys.argv)
    git_commit = _git_commit()
    anomalies = []
    if git_commit == "unknown":
        anomalies.append("git_commit unavailable; repository appears to have no commits")
    exp = make_moe_experiment(variant_dir, cfg_overrides=overrides)
    seeds = {
        "data_seed": int(exp.cfg.data_seed),
        "eval_seed": int(exp.cfg.eval_seed),
    }

    manifest = {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "smoke": True,
        "purpose": "Compare MoE soft, thresholded-soft, hard-threshold, and oracle gate policies after one tiny training run.",
        "policies": list(POLICIES),
        "expert_paths": [name for name, _, _, _ in EXPERT_PATHS] if args.include_expert_paths else [],
        "scale_sweep": [name for name, _, _, _ in SCALE_SWEEP_PATHS] if args.include_scale_sweep else [],
        "scale_calibration": {
            "enabled": bool(args.calibrate_scale),
            "candidates": [_format_scale(scale) for scale in scale_candidates],
            "fraction": float(args.scale_calibration_fraction),
        },
        "config_overrides": overrides,
        "seeds": seeds,
        "command": command,
    }
    _write_json(run_root / "manifest.json", manifest)

    summary = {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "status": "running",
        "command": command,
        "seeds": seeds,
        "anomalies": anomalies,
    }
    _write_json(run_root / "summary.json", summary)
    _write_run_note(run_root, summary, overrides)

    print(f"\nMoE gate-policy smoke run: {run_root}")
    t0 = time.time()
    try:
        exp.setup()
        exp._stage0_gate_pretrain()
        exp.stage0_contrastive_train()
        exp.stage1_train()
        exp.stage2_train()
        exp.stage3_calibrate()
        exp.save()

        if args.skip_post_eval:
            elapsed_s = time.time() - t0
            summary.update(
                {
                    "status": "ok",
                    "elapsed_s": elapsed_s,
                    "post_eval_skipped": True,
                    "policies": {},
                }
            )
            _write_json(variant_dir / "results.json", summary)
            _write_json(run_root / "summary.json", summary)
            _write_run_note(run_root, summary, overrides)
            print("\nPost-training policy evaluation skipped.")
            print(f"\nResults saved to: {run_root / 'summary.json'}")
            return

        eval_kwargs = {}
        scale_calibration_results = {}
        if args.calibrate_scale:
            calibration_kwargs, eval_kwargs, split_anomalies = _make_scale_calibration_splits(
                exp, float(args.scale_calibration_fraction)
            )
            anomalies.extend(split_anomalies)
            candidate_results = {
                _format_scale(scale): moe_evaluate(
                    exp,
                    gate_policy="soft",
                    expert_policy=args.eval_expert_policy,
                    gate_scale=scale,
                    **calibration_kwargs,
                )
                for scale in scale_candidates
            }
            selected_scale, selected_score = _select_scale_candidate(candidate_results)
            exp.cfg.moe_intervention_scale = selected_scale
            scale_calibration_results = {
                "candidates": [_format_scale(scale) for scale in scale_candidates],
                "selected_scale": selected_scale,
                "selected_score": selected_score,
                "calibration_fraction": float(args.scale_calibration_fraction),
                "calibration_split_sizes": {
                    "edit": len(calibration_kwargs["eval_fr_examples"]),
                    "benign": len(calibration_kwargs["eval_keep_benign"]),
                    "harmful": len(calibration_kwargs["eval_keep_harmful"]),
                },
                "test_split_sizes": {
                    "edit": len(eval_kwargs["eval_fr_examples"]),
                    "benign": len(eval_kwargs["eval_keep_benign"]),
                    "harmful": len(eval_kwargs["eval_keep_harmful"]),
                },
                "calibration_results": candidate_results,
            }

        policy_results = {
            policy: moe_evaluate(
                exp,
                gate_policy=policy,
                expert_policy=args.eval_expert_policy,
                **eval_kwargs,
            )
            for policy in POLICIES
        }
        expert_path_results = {}
        if args.include_expert_paths:
            expert_path_results = {
                name: moe_evaluate(
                    exp,
                    gate_policy=gate_policy,
                    expert_policy=expert_policy,
                    gate_scale=gate_scale,
                    **eval_kwargs,
                )
                for name, gate_policy, expert_policy, gate_scale in EXPERT_PATHS
            }
        scale_sweep_results = {}
        if args.include_scale_sweep:
            scale_sweep_results = {
                name: moe_evaluate(
                    exp,
                    gate_policy=gate_policy,
                    expert_policy=expert_policy,
                    gate_scale=gate_scale,
                    **eval_kwargs,
                )
                for name, gate_policy, expert_policy, gate_scale in SCALE_SWEEP_PATHS
            }
        elapsed_s = time.time() - t0
        summary.update(
            {
                "status": "ok",
                "elapsed_s": elapsed_s,
                "policies": policy_results,
            }
        )
        if expert_path_results:
            summary["expert_paths"] = expert_path_results
        if scale_sweep_results:
            summary["scale_sweep"] = scale_sweep_results
        if scale_calibration_results:
            summary["scale_calibration"] = scale_calibration_results
        _write_json(variant_dir / "results.json", summary)
        _write_json(run_root / "summary.json", summary)
        _write_run_note(run_root, summary, overrides)

        print("\nGate-policy smoke summary:")
        if scale_calibration_results:
            print(
                "  scale calibration selected "
                f"{scale_calibration_results['selected_scale']:g} "
                f"(score={scale_calibration_results['selected_score']:.3f})"
            )
        print(f"  {'Policy':<8} {'EditRef':>8} {'KeepPres':>8} {'HarmRef':>8} {'GateAcc':>8} {'EditGate':>8} {'KeepGate':>8}")
        print(f"  {'-' * 67}")
        for policy, result in policy_results.items():
            print(
                f"  {policy:<8} "
                f"{result.get('edit_refusal_rate', float('nan')):>8.3f} "
                f"{result.get('keep_preservation', float('nan')):>8.3f} "
                f"{result.get('harmful_keep_refusal_rate', float('nan')):>8.3f} "
                f"{result.get('gate_accuracy', float('nan')):>8.3f} "
                f"{result.get('edit_gate_strength_mean', float('nan')):>8.3f} "
                f"{result.get('benign_gate_strength_mean', float('nan')):>8.3f}"
            )
        _print_prefill_summary("Gate-policy prefill diagnostic", policy_results)
        _print_opening_rank_summary("Gate-policy opening-rank diagnostic", policy_results)
        if expert_path_results:
            print("\nExpert-path smoke summary:")
            print(f"  {'Path':<18} {'EditRef':>8} {'KeepPres':>8} {'HarmRef':>8} {'GateAcc':>8} {'EditGate':>8}")
            print(f"  {'-' * 67}")
            for path_name, result in expert_path_results.items():
                print(
                    f"  {path_name:<18} "
                    f"{result.get('edit_refusal_rate', float('nan')):>8.3f} "
                    f"{result.get('keep_preservation', float('nan')):>8.3f} "
                    f"{result.get('harmful_keep_refusal_rate', float('nan')):>8.3f} "
                        f"{result.get('gate_accuracy', float('nan')):>8.3f} "
                        f"{result.get('edit_gate_strength_mean', float('nan')):>8.3f}"
                    )
            _print_prefill_summary("Expert-path prefill diagnostic", expert_path_results)
            _print_opening_rank_summary(
                "Expert-path opening-rank diagnostic", expert_path_results
            )
        if scale_sweep_results:
            print("\nScale-sweep smoke summary:")
            print(f"  {'Path':<20} {'EditRef':>8} {'KeepPres':>8} {'HarmRef':>8} {'EditGate':>8}")
            print(f"  {'-' * 64}")
            for path_name, result in scale_sweep_results.items():
                print(
                    f"  {path_name:<20} "
                    f"{result.get('edit_refusal_rate', float('nan')):>8.3f} "
                    f"{result.get('keep_preservation', float('nan')):>8.3f} "
                        f"{result.get('harmful_keep_refusal_rate', float('nan')):>8.3f} "
                        f"{result.get('edit_gate_strength_mean', float('nan')):>8.3f}"
                    )
            _print_prefill_summary("Scale-sweep prefill diagnostic", scale_sweep_results)
            _print_opening_rank_summary(
                "Scale-sweep opening-rank diagnostic", scale_sweep_results
            )
        print(f"\nResults saved to: {run_root / 'summary.json'}")
    except Exception as exc:
        summary.update(
            {
                "status": "error",
                "elapsed_s": time.time() - t0,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(run_root / "summary.json", summary)
        _write_run_note(run_root, summary, overrides)
        raise


if __name__ == "__main__":
    main()
