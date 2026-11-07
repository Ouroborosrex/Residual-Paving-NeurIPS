"""Re-evaluate a saved MoE checkpoint with target-control diagnostics."""
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
from typing import Dict, Iterable, List, Tuple

# Match the other experiment runners' CUDA allocator workaround without
# changing PyTorch's process-global allocator configuration after import.
if "torch" not in sys.modules:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import torch

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
for _d in [str(_EXP_DIR), str(_ROOT / "src" / "trainer"), str(_ROOT / "src" / "optimized")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from moe_gate_policy_smoke import (  # noqa: E402
    _format_scale,
    _make_scale_calibration_splits,
    _parse_scale_candidates,
    _scale_calibration_score,
    _select_scale_candidate,
)
from moe_harness import make_moe_experiment, moe_evaluate  # noqa: E402


DEFAULT_EVAL_PATHS = (
    ("thresholded_soft_router_calibrated", "thresholded_soft", "router", "selected", None),
    ("soft_router_calibrated", "soft", "router", "selected", None),
    ("oracle_router_calibrated", "oracle", "router", "selected", None),
    ("oracle_expert0_x4", "oracle", "expert0", "4", None),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a saved MoE checkpoint, rebuild deterministic eval/reference "
            "caches, calibrate intervention scale, and score diagnostic paths."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a MoE checkpoint.pt file.")
    parser.add_argument(
        "--model-quantization",
        choices=("none", "8bit", "4bit"),
        default=None,
        help="Optional override for quantized model loading during re-evaluation.",
    )
    parser.add_argument(
        "--model-quant-compute-dtype",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
        default=None,
        help="Optional override for quantized model compute dtype during re-evaluation.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Optional override for n_edit, n_keep_benign, and n_keep_harmful.",
    )
    parser.add_argument(
        "--n-edit",
        type=int,
        default=None,
        help="Optional override for n_edit after applying --n.",
    )
    parser.add_argument(
        "--n-keep-benign",
        type=int,
        default=None,
        help="Optional override for n_keep_benign after applying --n.",
    )
    parser.add_argument(
        "--n-keep-harmful",
        type=int,
        default=None,
        help="Optional override for n_keep_harmful after applying --n.",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=None,
        help="Optional override for the raw train/eval split fraction.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Optional override for prompt sampling seed.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Optional override for train/eval split seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional run directory. Defaults to experiments/runs/<timestamp>_moe_checkpoint_reeval.",
    )
    parser.add_argument("--name", default="moe_checkpoint_reeval")
    parser.add_argument("--scale-candidates", default="0.5,1,2,4")
    parser.add_argument("--scale-calibration-fraction", type=float, default=0.5)
    parser.add_argument(
        "--scale-calibration-expert-policy",
        default="router",
        help="Expert policy used while selecting the calibrated scale.",
    )
    parser.add_argument(
        "--skip-scale-calibration",
        action="store_true",
        help="Evaluate with the checkpoint/config scale instead of selecting on a calibration slice.",
    )
    parser.add_argument(
        "--eval-path",
        action="append",
        default=[],
        metavar="NAME:GATE:EXPERT:SCALE",
        help=(
            "Diagnostic path to evaluate. GATE may be soft, thresholded_soft, hard, "
            "or oracle. SCALE may be a float or 'selected'. "
            "Optionally append :THRESHOLD to override hard-gate threshold for that path. "
            "May be passed multiple times. Defaults to four target-control paths."
        ),
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=None,
        help="Override eval_max_new_tokens from the checkpoint config.",
    )
    parser.add_argument(
        "--trace-batch-size",
        type=int,
        default=None,
        help="Override trace_batch_size from the checkpoint config.",
    )
    parser.add_argument(
        "--build-batch-size",
        type=int,
        default=None,
        help="Override build_batch_size from the checkpoint config.",
    )
    parser.add_argument(
        "--edit-refusal-filter-max-new-tokens",
        type=int,
        default=None,
        help="Override edit_refusal_filter_max_new_tokens from the checkpoint config.",
    )
    parser.add_argument(
        "--edit-prompts-path",
        default=None,
        help="Optional explicit edit-prompt source for rebuilt evaluation data.",
    )
    parser.add_argument(
        "--edit-prompts-prefiltered-base-refusal",
        action="store_true",
        help="Treat --edit-prompts-path as already filtered to frozen-base refusals.",
    )
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
        "--eval-prefill-anchor-prefix",
        action="store_true",
        help="Also evaluate edit generations with edit_anchor_prefix prefilled as assistant text.",
    )
    parser.add_argument(
        "--harmful-veto",
        action="store_true",
        help="Fit and apply a cached-boundary harmful-keep veto during learned-gate re-eval.",
    )
    parser.add_argument(
        "--harmful-veto-l2-weight",
        type=float,
        default=None,
        help="Optional L2 weight for the harmful-veto linear head.",
    )
    parser.add_argument(
        "--harmful-veto-threshold-tie-break",
        choices=("low", "high"),
        default=None,
        help=(
            "Tie-break for equal-accuracy harmful-veto thresholds. "
            "'low' preserves prior behavior; 'high' is more conservative about edit false positives."
        ),
    )
    parser.add_argument(
        "--harmful-veto-mode",
        choices=("hard", "soft"),
        default=None,
        help=(
            "How the harmful-veto logit changes learned gate strengths. "
            "'hard' zeroes gates above threshold; 'soft' uses a sigmoid multiplier."
        ),
    )
    parser.add_argument(
        "--harmful-veto-temperature",
        type=float,
        default=None,
        help="Temperature for --harmful-veto-mode soft.",
    )
    parser.add_argument(
        "--harmful-veto-min-multiplier",
        type=float,
        default=None,
        help="Floor multiplier for --harmful-veto-mode soft.",
    )
    parser.add_argument(
        "--harmful-veto-ensemble-l2-values",
        default=None,
        help=(
            "Optional comma-separated L2 weights for an ensemble of harmful-veto "
            "linear heads. Empty uses the single-head path."
        ),
    )
    parser.add_argument(
        "--harmbench-paths",
        default="",
        help="Optional comma-separated harmful-keep source paths for rebuilt eval data.",
    )
    return parser.parse_args()


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


def _write_run_note(run_root: Path, summary: dict) -> None:
    lines = [
        "# MoE Checkpoint Re-Eval",
        "",
        f"- command: `{summary['command']}`",
        f"- timestamp: `{summary['timestamp']}`",
        f"- git_commit: `{summary['git_commit']}`",
        f"- status: `{summary['status']}`",
        f"- source_checkpoint: `{summary['source_checkpoint']}`",
    ]
    if "selected_scale" in summary:
        lines.append(
            f"- selected_scale: `{summary.get('selected_scale')}` "
            f"score=`{summary.get('selected_score')}`"
        )
    if summary.get("paths"):
        lines.append(f"- paths: `{', '.join(summary['paths'])}`")
    if summary.get("anomalies"):
        lines.append(f"- anomalies: `{'; '.join(summary['anomalies'])}`")
    if "elapsed_s" in summary:
        lines.append(f"- elapsed_s: `{summary['elapsed_s']:.2f}`")
    run_root.joinpath("run_note.md").write_text("\n".join(lines) + "\n")


def _resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _load_checkpoint(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ("moe_router_state_dict", "moe_controller_state_dict", "config")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Checkpoint is missing required MoE keys: {missing}")
    return payload


def _checkpoint_overrides(payload: Dict, args: argparse.Namespace) -> Dict:
    overrides = dict(payload.get("config", {}))
    overrides["name"] = str(args.name)
    overrides["use_moe"] = True
    if args.model_quantization is not None:
        overrides["model_quantization"] = str(args.model_quantization)
    if args.model_quant_compute_dtype is not None:
        overrides["model_quant_compute_dtype"] = str(args.model_quant_compute_dtype)
    if args.n is not None:
        overrides["n_edit"] = int(args.n)
        overrides["n_keep_benign"] = int(args.n)
        overrides["n_keep_harmful"] = int(args.n)
    if args.n_edit is not None:
        overrides["n_edit"] = int(args.n_edit)
    if args.n_keep_benign is not None:
        overrides["n_keep_benign"] = int(args.n_keep_benign)
    if args.n_keep_harmful is not None:
        overrides["n_keep_harmful"] = int(args.n_keep_harmful)
    if args.eval_fraction is not None:
        overrides["eval_fraction"] = float(args.eval_fraction)
    if args.data_seed is not None:
        overrides["data_seed"] = int(args.data_seed)
    if args.eval_seed is not None:
        overrides["eval_seed"] = int(args.eval_seed)
    if args.eval_max_new_tokens is not None:
        overrides["eval_max_new_tokens"] = int(args.eval_max_new_tokens)
    if args.trace_batch_size is not None:
        overrides["trace_batch_size"] = int(args.trace_batch_size)
    if args.build_batch_size is not None:
        overrides["build_batch_size"] = int(args.build_batch_size)
    if args.edit_refusal_filter_max_new_tokens is not None:
        overrides["edit_refusal_filter_max_new_tokens"] = int(
            args.edit_refusal_filter_max_new_tokens
        )
    if args.edit_prompts_path is not None:
        overrides["edit_prompts_path"] = str(args.edit_prompts_path)
    if bool(args.edit_prompts_prefiltered_base_refusal):
        overrides["edit_prompts_prefiltered_base_refusal"] = True
    if args.example_cache_dir is not None:
        overrides["example_cache_dir"] = str(args.example_cache_dir)
    if bool(args.refresh_example_cache):
        overrides["example_cache_refresh"] = True
    overrides["moe_eval_prefill_anchor_prefix"] = bool(args.eval_prefill_anchor_prefix)
    if bool(args.harmful_veto):
        overrides["harmful_veto_enabled"] = True
    if args.harmful_veto_l2_weight is not None:
        overrides["harmful_veto_l2_weight"] = float(args.harmful_veto_l2_weight)
    if args.harmful_veto_threshold_tie_break is not None:
        overrides["harmful_veto_threshold_tie_break"] = str(
            args.harmful_veto_threshold_tie_break
        )
    if args.harmful_veto_mode is not None:
        overrides["harmful_veto_mode"] = str(args.harmful_veto_mode)
    if args.harmful_veto_temperature is not None:
        overrides["harmful_veto_temperature"] = float(args.harmful_veto_temperature)
    if args.harmful_veto_min_multiplier is not None:
        overrides["harmful_veto_min_multiplier"] = float(args.harmful_veto_min_multiplier)
    if args.harmful_veto_ensemble_l2_values is not None:
        overrides["harmful_veto_ensemble_l2_values"] = str(
            args.harmful_veto_ensemble_l2_values
        )
    if str(args.harmbench_paths).strip():
        overrides["harmbench_paths"] = str(args.harmbench_paths).strip()
    return overrides


def _load_moe_state(exp, payload: Dict) -> List[str]:
    anomalies: List[str] = []
    exp.moe_router.load_state_dict(payload["moe_router_state_dict"])
    exp.moe_controller.load_state_dict(payload["moe_controller_state_dict"])
    exp.gate = exp.moe_router
    exp.gate_threshold = float(payload.get("gate_threshold", 0.0))
    exp.gate_threshold_search_acc = float(payload.get("gate_threshold_search_acc", 0.0))
    exp.gate_threshold_train_acc = float(payload.get("gate_threshold_train_acc", 0.0))

    router_layers = payload.get("router_layer_indices")
    if router_layers is not None and list(router_layers) != list(getattr(exp, "_router_indices", [])):
        anomalies.append(
            "checkpoint router_layer_indices differ from rebuilt experiment config"
        )
    intervention_layers = payload.get("intervention_layer_indices")
    if intervention_layers is not None and list(intervention_layers) != list(
        getattr(exp, "_interv_indices", [])
    ):
        anomalies.append(
            "checkpoint intervention_layer_indices differ from rebuilt experiment config"
        )
    return anomalies


def _parse_eval_path_spec(raw: str) -> Tuple[str, str, str, str, str | None]:
    parts = [part.strip() for part in str(raw).split(":")]
    if len(parts) not in {4, 5} or any(not part for part in parts):
        raise ValueError(
            f"Invalid eval path {raw!r}. Expected NAME:GATE:EXPERT:SCALE[:THRESHOLD]."
        )
    name, gate_policy, expert_policy, scale = parts[:4]
    gate_policy = gate_policy.lower()
    if gate_policy not in {"soft", "thresholded_soft", "hard", "oracle"}:
        raise ValueError(f"Invalid gate policy in eval path {raw!r}: {gate_policy!r}")
    scale = scale.lower()
    if scale != "selected":
        float(scale)
    threshold = None
    if len(parts) == 5:
        threshold = parts[4].lower()
        if threshold not in {"checkpoint", "default"}:
            float(threshold)
    return name, gate_policy, expert_policy.lower(), scale, threshold


def _resolve_eval_paths(
    raw_paths: Iterable[str],
    *,
    selected_scale: float,
) -> List[Tuple[str, str, str, float, float | None]]:
    raw_paths = list(raw_paths)
    specs = [_parse_eval_path_spec(raw) for raw in raw_paths] if raw_paths else list(DEFAULT_EVAL_PATHS)
    resolved = []
    for name, gate_policy, expert_policy, scale, threshold in specs:
        resolved_scale = float(selected_scale) if scale == "selected" else float(scale)
        resolved_threshold = (
            None if threshold in {None, "checkpoint", "default"} else float(threshold)
        )
        resolved.append((name, gate_policy, expert_policy, resolved_scale, resolved_threshold))
    return resolved


def _attach_target_control_score(result: Dict) -> Dict:
    scored = dict(result)
    scored["target_control_score"] = _scale_calibration_score(scored)
    return scored


def _any_prefill_enabled(results: Dict[str, Dict]) -> bool:
    return any(
        bool(result.get("edit_prefill_anchor_prefix_enabled", False))
        for result in results.values()
    )


def _any_opening_rank_metrics(results: Dict[str, Dict]) -> bool:
    return any("edit_anchor_first_token_rank_mean" in result for result in results.values())


def _print_opening_rank_summary(results: Dict[str, Dict]) -> None:
    if not _any_opening_rank_metrics(results):
        return
    print("\nOpening-rank diagnostic:")
    print(f"  {'Path':<28} {'FirstProb':>10} {'FirstRank':>10} {'Top10':>7}")
    print(f"  {'-' * 61}")
    for name, result in results.items():
        print(
            "  "
            f"{name:<28} "
            f"{result.get('edit_anchor_first_token_prob_mean', float('nan')):>10.6f} "
            f"{result.get('edit_anchor_first_token_rank_mean', float('nan')):>10.1f} "
            f"{result.get('edit_anchor_first_token_top10_rate', float('nan')):>7.3f}"
        )


def main() -> None:
    args = _parse_args()
    checkpoint_path = _resolve_path(args.checkpoint)
    payload = _load_checkpoint(checkpoint_path)
    scale_candidates = _parse_scale_candidates(args.scale_candidates)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_moe_checkpoint_reeval")
    run_root = _resolve_path(args.output_dir) if args.output_dir else _ROOT / "experiments" / "runs" / timestamp
    run_root.mkdir(parents=True, exist_ok=False)

    command = " ".join([sys.executable] + sys.argv)
    git_commit = _git_commit()
    anomalies: List[str] = []
    if git_commit == "unknown":
        anomalies.append("git_commit unavailable; repository appears to have no commits")

    cfg_overrides = _checkpoint_overrides(payload, args)
    manifest = {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "source_checkpoint": str(checkpoint_path),
        "scale_candidates": [_format_scale(scale) for scale in scale_candidates],
        "scale_calibration_fraction": float(args.scale_calibration_fraction),
        "scale_calibration_expert_policy": str(args.scale_calibration_expert_policy),
        "scale_calibration_enabled": not bool(args.skip_scale_calibration),
        "default_paths": [
            ":".join(part for part in path if part is not None)
            for path in DEFAULT_EVAL_PATHS
        ],
        "eval_paths": list(args.eval_path),
        "config_overrides": cfg_overrides,
        "command": command,
    }
    _write_json(run_root / "manifest.json", manifest)

    summary = {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "status": "running",
        "command": command,
        "source_checkpoint": str(checkpoint_path),
        "anomalies": anomalies,
    }
    _write_json(run_root / "summary.json", summary)
    _write_run_note(run_root, summary)

    print(f"\nMoE checkpoint re-eval run: {run_root}")
    print(f"Source checkpoint: {checkpoint_path}")
    t0 = time.time()
    try:
        exp = make_moe_experiment(run_root / "moe_checkpoint_reeval", cfg_overrides=cfg_overrides)
        exp.setup()
        anomalies.extend(_load_moe_state(exp, payload))
        if bool(getattr(exp.cfg, "harmful_veto_enabled", False)):
            exp._fit_harmful_veto_head()

        eval_kwargs = {}
        calibration_results = {}
        selected_scale = float(getattr(exp.cfg, "moe_intervention_scale", 1.0))
        selected_score = None
        if not args.skip_scale_calibration:
            calibration_kwargs, eval_kwargs, split_anomalies = _make_scale_calibration_splits(
                exp, float(args.scale_calibration_fraction)
            )
            anomalies.extend(split_anomalies)
            calibration_results = {
                _format_scale(scale): _attach_target_control_score(
                    moe_evaluate(
                        exp,
                        gate_policy="soft",
                        expert_policy=str(args.scale_calibration_expert_policy),
                        gate_scale=scale,
                        **calibration_kwargs,
                    )
                )
                for scale in scale_candidates
            }
            selected_scale, selected_score = _select_scale_candidate(calibration_results)
            exp.cfg.moe_intervention_scale = selected_scale
            calibration_split_sizes = {
                "edit": len(calibration_kwargs["eval_fr_examples"]),
                "benign": len(calibration_kwargs["eval_keep_benign"]),
                "harmful": len(calibration_kwargs["eval_keep_harmful"]),
            }
            test_split_sizes = {
                "edit": len(eval_kwargs["eval_fr_examples"]),
                "benign": len(eval_kwargs["eval_keep_benign"]),
                "harmful": len(eval_kwargs["eval_keep_harmful"]),
            }
        else:
            calibration_split_sizes = {}
            test_split_sizes = {}

        eval_paths = _resolve_eval_paths(args.eval_path, selected_scale=selected_scale)
        policy_results = {
            name: _attach_target_control_score(
                moe_evaluate(
                    exp,
                    gate_policy=gate_policy,
                    expert_policy=expert_policy,
                    gate_scale=gate_scale,
                    gate_threshold_override=gate_threshold,
                    **eval_kwargs,
                )
            )
            for name, gate_policy, expert_policy, gate_scale, gate_threshold in eval_paths
        }

        elapsed_s = time.time() - t0
        summary.update(
            {
                "status": "ok",
                "elapsed_s": elapsed_s,
                "selected_scale": selected_scale,
                "selected_score": selected_score,
                "scale_candidates": [_format_scale(scale) for scale in scale_candidates],
                "scale_calibration_fraction": float(args.scale_calibration_fraction),
                "scale_calibration_expert_policy": str(args.scale_calibration_expert_policy),
                "scale_calibration_enabled": not bool(args.skip_scale_calibration),
                "calibration_split_sizes": calibration_split_sizes,
                "test_split_sizes": test_split_sizes,
                "calibration_results": calibration_results,
                "paths": [name for name, _, _, _, _ in eval_paths],
                "policy_results": policy_results,
                "anomalies": anomalies,
            }
        )
        _write_json(run_root / "summary.json", summary)
        _write_json(run_root / "moe_checkpoint_reeval" / "results.json", summary)
        _write_run_note(run_root, summary)

        print("\nCheckpoint re-eval summary:")
        print(f"  selected_scale={selected_scale:g} selected_score={selected_score}")
        if _any_prefill_enabled(policy_results):
            print(
                "  "
                f"{'Path':<28} {'EditRef':>7} {'SafeNR':>7} {'Oper':>7} "
                f"{'PrefRef':>7} {'PrefSafe':>8} {'PrefOp':>7} {'Align':>8} "
                f"{'Keep':>7} {'HarmRef':>7} {'Score':>7}"
            )
            print(f"  {'-' * 117}")
            for name, result in policy_results.items():
                print(
                    "  "
                    f"{name:<28} "
                    f"{result.get('edit_refusal_rate', float('nan')):>7.3f} "
                    f"{result.get('edit_nonrefusal_nonoperational_rate', float('nan')):>7.3f} "
                    f"{result.get('edit_operational_compliance_heuristic_rate', float('nan')):>7.3f} "
                    f"{result.get('edit_prefill_refusal_rate', float('nan')):>7.3f} "
                    f"{result.get('edit_prefill_nonrefusal_nonoperational_rate', float('nan')):>8.3f} "
                    f"{result.get('edit_prefill_operational_compliance_heuristic_rate', float('nan')):>7.3f} "
                    f"{result.get('edit_target_alignment', float('nan')):>8.4f} "
                    f"{result.get('keep_preservation', float('nan')):>7.3f} "
                    f"{result.get('harmful_keep_refusal_rate', float('nan')):>7.3f} "
                    f"{result.get('target_control_score', float('nan')):>7.3f}"
                )
        else:
            print(
                "  "
                f"{'Path':<28} {'EditRef':>7} {'SafeNR':>7} {'Oper':>7} "
                f"{'Align':>8} {'Keep':>7} {'HarmRef':>7} {'Score':>7}"
            )
            print(f"  {'-' * 87}")
            for name, result in policy_results.items():
                print(
                    "  "
                    f"{name:<28} "
                    f"{result.get('edit_refusal_rate', float('nan')):>7.3f} "
                    f"{result.get('edit_nonrefusal_nonoperational_rate', float('nan')):>7.3f} "
                    f"{result.get('edit_operational_compliance_heuristic_rate', float('nan')):>7.3f} "
                    f"{result.get('edit_target_alignment', float('nan')):>8.4f} "
                    f"{result.get('keep_preservation', float('nan')):>7.3f} "
                    f"{result.get('harmful_keep_refusal_rate', float('nan')):>7.3f} "
                    f"{result.get('target_control_score', float('nan')):>7.3f}"
                )
        _print_opening_rank_summary(policy_results)
        print(f"\nResults saved to: {run_root / 'summary.json'}")
    except Exception as exc:
        summary.update(
            {
                "status": "error",
                "elapsed_s": time.time() - t0,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "anomalies": anomalies,
            }
        )
        _write_json(run_root / "summary.json", summary)
        _write_run_note(run_root, summary)
        raise


if __name__ == "__main__":
    main()
