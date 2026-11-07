"""Export MoE checkpoint completions for external judge scoring.

The standard checkpoint re-evaluator stores aggregate metrics and short previews.
This script saves full base and steered completions for each held-out bucket so
stronger judges can be run without regenerating.
"""
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

if "torch" not in sys.modules:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import torch

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
for _d in [str(_EXP_DIR), str(_ROOT / "src" / "trainer"), str(_ROOT / "src" / "optimized")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from moe_checkpoint_reeval import (  # noqa: E402
    _checkpoint_overrides,
    _load_checkpoint,
    _load_moe_state,
    _resolve_path,
)
from moe_harness import (  # noqa: E402
    _encode_prefill_prefix_ids,
    _is_operational_compliance_moe,
    _is_refusal_moe,
    apply_harmful_veto_to_gate_strengths,
    make_moe_experiment,
    moe_expert_weights_for_policy,
    moe_gate_strengths_for_policy,
)
from moe_residual_paver import moe_generate_batch  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--name", default="moe_export_completions")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--n-edit", type=int, default=None)
    parser.add_argument("--n-keep-benign", type=int, default=None)
    parser.add_argument("--n-keep-harmful", type=int, default=None)
    parser.add_argument("--eval-fraction", type=float, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--eval-max-new-tokens", type=int, default=None)
    parser.add_argument("--trace-batch-size", type=int, default=None)
    parser.add_argument("--build-batch-size", type=int, default=None)
    parser.add_argument("--edit-refusal-filter-max-new-tokens", type=int, default=None)
    parser.add_argument("--edit-prompts-path", default=None)
    parser.add_argument("--edit-prompts-prefiltered-base-refusal", action="store_true")
    parser.add_argument("--example-cache-dir", default=None)
    parser.add_argument("--refresh-example-cache", action="store_true")
    parser.add_argument("--eval-prefill-anchor-prefix", action="store_true")
    parser.add_argument(
        "--conditional-prefill-anchor-prefix",
        action="store_true",
        help=(
            "Export an algorithmic variant that prepends edit_anchor_prefix only "
            "for examples whose post-veto gate strength is nonzero."
        ),
    )
    parser.add_argument(
        "--conditional-prefill-veto-max",
        type=float,
        default=None,
        help=(
            "Optional max harmful-veto logit allowed for conditional prefixing. "
            "This can keep the prefix limited to high-confidence edit-like routes."
        ),
    )
    parser.add_argument(
        "--conditional-prefill-text",
        default="",
        help=(
            "Optional text to prepend when --conditional-prefill-anchor-prefix is enabled. "
            "Defaults to cfg.edit_anchor_prefix when empty."
        ),
    )
    parser.add_argument("--model-quantization", choices=("none", "8bit", "4bit"), default=None)
    parser.add_argument(
        "--model-quant-compute-dtype",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
        default=None,
    )
    parser.add_argument("--harmful-veto", action="store_true")
    parser.add_argument("--harmful-veto-l2-weight", type=float, default=None)
    parser.add_argument("--harmful-veto-threshold-tie-break", choices=("low", "high"), default=None)
    parser.add_argument("--harmful-veto-mode", choices=("hard", "soft"), default=None)
    parser.add_argument("--harmful-veto-temperature", type=float, default=None)
    parser.add_argument("--harmful-veto-min-multiplier", type=float, default=None)
    parser.add_argument("--harmful-veto-ensemble-l2-values", default=None)
    parser.add_argument("--harmbench-paths", default="")
    parser.add_argument(
        "--eval-path",
        action="append",
        default=[],
        metavar="NAME:GATE:EXPERT:SCALE[:THRESHOLD]",
        help="Path to export. Gate is soft, thresholded_soft, hard, or oracle.",
    )
    parser.add_argument(
        "--buckets",
        default="edit,benign_keep,harmful_keep",
        help="Comma-separated subset of edit, benign_keep, harmful_keep.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-ref-caches",
        action="store_true",
        help="Skip frozen reference-cache construction; completion export does not use it.",
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


def _write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _parse_eval_path(raw: str) -> Tuple[str, str, str, float, float | None]:
    parts = [part.strip() for part in str(raw).split(":")]
    if len(parts) not in {4, 5}:
        raise ValueError(f"Invalid eval path {raw!r}; expected NAME:GATE:EXPERT:SCALE[:THRESHOLD].")
    name, gate_policy, expert_policy, scale = parts[:4]
    threshold = None if len(parts) == 4 or parts[4] in {"", "default", "checkpoint"} else float(parts[4])
    return name, gate_policy.lower(), expert_policy.lower(), float(scale), threshold


def _resolve_eval_paths(raw_paths: Iterable[str]) -> List[Tuple[str, str, str, float, float | None]]:
    paths = list(raw_paths)
    if not paths:
        paths = ["thresholded_soft_router_x8_veto:thresholded_soft:router:8"]
    return [_parse_eval_path(path) for path in paths]


def _decode(tokenizer, tokens_list: List[torch.Tensor]) -> List[str]:
    return [
        tokenizer.decode(tokens.detach().cpu().tolist(), skip_special_tokens=True).strip()
        for tokens in tokens_list
    ]


def _bucket_examples(exp, bucket: str):
    if bucket == "edit":
        return exp.eval_fr_examples or exp.fr_examples, 1.0
    if bucket == "benign_keep":
        return exp.eval_keep_benign or exp.keep_benign, 0.0
    if bucket == "harmful_keep":
        return exp.eval_keep_harmful or exp.keep_harmful, 0.0
    raise ValueError(f"Unknown bucket {bucket!r}")


@torch.no_grad()
def _export_path(
    exp,
    *,
    path_name: str,
    gate_policy: str,
    expert_policy: str,
    gate_scale: float,
    gate_threshold_override: float | None,
    conditional_prefill_anchor_prefix: bool,
    conditional_prefill_veto_max: float | None,
    conditional_prefill_text: str,
    buckets: List[str],
    limit: int | None,
    run_root: Path,
) -> Dict:
    device = exp.device
    eos_id = getattr(exp.tokenizer, "eos_token_id", None)
    eval_tokens = int(max(exp.cfg.eval_max_new_tokens, exp.cfg.steer_steps))
    gate_threshold = (
        float(gate_threshold_override)
        if gate_threshold_override is not None
        else float(getattr(exp, "gate_threshold", 0.0))
    )
    prefill_prefix_ids = None
    if bool(conditional_prefill_anchor_prefix):
        prefix_text = str(conditional_prefill_text).strip()
        if not prefix_text:
            prefix_text = str(getattr(exp.cfg, "edit_anchor_prefix", "")).strip()
        if prefix_text:
            prefill_prefix_ids = _encode_prefill_prefix_ids(
                exp.tokenizer,
                prefix_text,
                device=device,
            )
    summary_rows = []
    all_rows = []

    for bucket in buckets:
        examples, oracle_target = _bucket_examples(exp, bucket)
        examples = list(examples)
        if limit is not None:
            examples = examples[: int(limit)]
        rows = []
        for start in range(0, len(examples), int(exp.cfg.trace_batch_size)):
            chunk = examples[start : start + int(exp.cfg.trace_batch_size)]
            bd_batch = torch.stack([ex.boundary_states.to(device) for ex in chunk])
            gate_logit, expert_weights = exp.moe_router(bd_batch)
            expert_weights = moe_expert_weights_for_policy(expert_weights, expert_policy)
            gate_strengths = moe_gate_strengths_for_policy(
                gate_logit,
                gate_threshold,
                gate_policy,
                oracle_target=oracle_target,
            ) * float(gate_scale)
            gate_strengths, veto_logits = apply_harmful_veto_to_gate_strengths(
                exp,
                bd_batch,
                gate_strengths,
                gate_policy,
            )
            base_generated = moe_generate_batch(
                model=exp.model,
                moe_paver=exp.moe_paver,
                controller=exp.moe_controller,
                base_ids_batch=[ex.base_ids for ex in chunk],
                gate_strengths=torch.zeros_like(gate_strengths),
                expert_weights=expert_weights,
                max_new_tokens=eval_tokens,
                eos_token_id=eos_id,
            )
            steered_generated = moe_generate_batch(
                model=exp.model,
                moe_paver=exp.moe_paver,
                controller=exp.moe_controller,
                base_ids_batch=[ex.base_ids for ex in chunk],
                gate_strengths=gate_strengths,
                expert_weights=expert_weights,
                max_new_tokens=eval_tokens,
                eos_token_id=eos_id,
            )
            if prefill_prefix_ids is not None:
                conditional_base_ids = []
                conditional_gate_strengths = []
                conditional_expert_weights = []
                conditional_indices = []
                gate_values = gate_strengths.detach().view(-1)
                veto_values_for_prefill = (
                    [None] * len(chunk)
                    if veto_logits is None
                    else veto_logits.detach().view(-1).float().tolist()
                )
                for idx, ex in enumerate(chunk):
                    if float(gate_values[idx].item()) <= 0.0:
                        continue
                    if (
                        conditional_prefill_veto_max is not None
                        and veto_values_for_prefill[idx] is not None
                        and float(veto_values_for_prefill[idx])
                        > float(conditional_prefill_veto_max)
                    ):
                        continue
                    conditional_indices.append(idx)
                    conditional_base_ids.append(
                        torch.cat(
                            [
                                ex.base_ids.view(-1),
                                prefill_prefix_ids.to(ex.base_ids.device),
                            ],
                            dim=0,
                        )
                    )
                    conditional_gate_strengths.append(gate_strengths[idx : idx + 1])
                    conditional_expert_weights.append(expert_weights[idx : idx + 1])
                if conditional_indices:
                    conditional_generated = moe_generate_batch(
                        model=exp.model,
                        moe_paver=exp.moe_paver,
                        controller=exp.moe_controller,
                        base_ids_batch=conditional_base_ids,
                        gate_strengths=torch.cat(conditional_gate_strengths, dim=0),
                        expert_weights=torch.cat(conditional_expert_weights, dim=0),
                        max_new_tokens=max(
                            1,
                            eval_tokens - int(prefill_prefix_ids.numel()),
                        ),
                        eos_token_id=eos_id,
                    )
                    for idx, generated in zip(conditional_indices, conditional_generated):
                        steered_generated[idx] = torch.cat(
                            [
                                prefill_prefix_ids.to(generated.device),
                                generated,
                            ],
                            dim=0,
                        )
            base_texts = _decode(exp.tokenizer, base_generated)
            steered_texts = _decode(exp.tokenizer, steered_generated)
            gate_logits = gate_logit.detach().cpu().view(-1).float().tolist()
            strengths = gate_strengths.detach().cpu().view(-1).float().tolist()
            veto_values = (
                [None] * len(chunk)
                if veto_logits is None
                else veto_logits.detach().cpu().view(-1).float().tolist()
            )
            for idx, ex in enumerate(chunk):
                row = {
                    "path": path_name,
                    "bucket": bucket,
                    "sample_id": str(getattr(ex, "sample_id", start + idx)),
                    "category": str(getattr(ex, "bucket", bucket)),
                    "prompt": str(ex.prompt),
                    "base_response": base_texts[idx],
                    "response": steered_texts[idx],
                    "base_refusal": bool(_is_refusal_moe(base_texts[idx])),
                    "refusal": bool(_is_refusal_moe(steered_texts[idx])),
                    "operational_compliance_heuristic": bool(
                        _is_operational_compliance_moe(steered_texts[idx])
                    ),
                    "conditional_prefill_anchor_prefix": bool(
                        prefill_prefix_ids is not None
                        and float(gate_strengths.detach().view(-1)[idx].item()) > 0.0
                        and (
                            conditional_prefill_veto_max is None
                            or veto_values[idx] is None
                            or float(veto_values[idx]) <= float(conditional_prefill_veto_max)
                        )
                    ),
                    "gate_logit": float(gate_logits[idx]),
                    "gate_strength": float(strengths[idx]),
                    "veto_logit": None if veto_values[idx] is None else float(veto_values[idx]),
                }
                rows.append(row)
                all_rows.append(row)

        out_path = run_root / "completions" / f"{path_name}_{bucket}.json"
        _write_json(out_path, {"rows": rows})
        refusal_rate = sum(row["refusal"] for row in rows) / max(len(rows), 1)
        base_refusal_rate = sum(row["base_refusal"] for row in rows) / max(len(rows), 1)
        op_rate = sum(row["operational_compliance_heuristic"] for row in rows) / max(len(rows), 1)
        summary_rows.append(
            {
                "path": path_name,
                "bucket": bucket,
                "n": len(rows),
                "refusal_rate": float(refusal_rate),
                "base_refusal_rate": float(base_refusal_rate),
                "operational_compliance_heuristic_rate": float(op_rate),
                "nonrefusal_nonoperational_rate": float(max(0.0, 1.0 - refusal_rate - op_rate)),
                "completions_path": str(out_path),
            }
        )

    _write_json(run_root / "completions" / f"{path_name}_all.json", {"rows": all_rows})
    return {
        "path": path_name,
        "gate_policy": gate_policy,
        "expert_policy": expert_policy,
        "gate_scale": float(gate_scale),
        "gate_threshold": gate_threshold,
        "rows": summary_rows,
    }


def main() -> None:
    args = _parse_args()
    checkpoint_path = _resolve_path(args.checkpoint)
    payload = _load_checkpoint(checkpoint_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_moe_export_completions")
    run_root = _resolve_path(args.output_dir) if args.output_dir else _ROOT / "experiments" / "runs" / timestamp
    run_root.mkdir(parents=True, exist_ok=False)

    summary = {
        "timestamp": timestamp,
        "status": "running",
        "command": " ".join([sys.executable] + sys.argv),
        "git_commit": _git_commit(),
        "source_checkpoint": str(checkpoint_path),
    }
    _write_json(run_root / "summary.json", summary)
    t0 = time.time()
    try:
        cfg_overrides = _checkpoint_overrides(payload, args)
        exp = make_moe_experiment(run_root / "moe_export_completions", cfg_overrides=cfg_overrides)
        if bool(args.skip_ref_caches):
            exp._build_ref_caches = lambda: None  # type: ignore[method-assign]
        exp.setup()
        anomalies = _load_moe_state(exp, payload)
        if bool(getattr(exp.cfg, "harmful_veto_enabled", False)):
            exp._fit_harmful_veto_head()

        buckets = [part.strip() for part in str(args.buckets).split(",") if part.strip()]
        path_summaries = [
            _export_path(
                exp,
                path_name=name,
                gate_policy=gate_policy,
                expert_policy=expert_policy,
                gate_scale=scale,
                gate_threshold_override=threshold,
                conditional_prefill_anchor_prefix=bool(args.conditional_prefill_anchor_prefix),
                conditional_prefill_veto_max=args.conditional_prefill_veto_max,
                conditional_prefill_text=str(args.conditional_prefill_text),
                buckets=buckets,
                limit=args.limit,
                run_root=run_root,
            )
            for name, gate_policy, expert_policy, scale, threshold in _resolve_eval_paths(args.eval_path)
        ]
        summary.update(
            {
                "status": "ok",
                "elapsed_s": float(time.time() - t0),
                "anomalies": anomalies,
                "paths": [row["path"] for row in path_summaries],
                "buckets": buckets,
                "path_summaries": path_summaries,
            }
        )
        _write_json(run_root / "summary.json", summary)
        (run_root / "run_note.md").write_text(
            "\n".join(
                [
                    "# MoE Completion Export",
                    "",
                    f"- command: `{summary['command']}`",
                    f"- status: `{summary['status']}`",
                    f"- source_checkpoint: `{summary['source_checkpoint']}`",
                    f"- elapsed_s: `{summary['elapsed_s']:.2f}`",
                    f"- paths: `{', '.join(summary['paths'])}`",
                    f"- buckets: `{', '.join(summary['buckets'])}`",
                ]
            )
            + "\n"
        )
        print(run_root)
    except Exception as exc:
        summary.update(
            {
                "status": "error",
                "elapsed_s": float(time.time() - t0),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(run_root / "summary.json", summary)
        raise


if __name__ == "__main__":
    main()
