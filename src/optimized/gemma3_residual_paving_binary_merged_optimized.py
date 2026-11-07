#!/usr/bin/env python3
"""
Optimized launcher for the merged binary residual-paving trainer.

Intent
------
Keep the original method intact by default, but replace the most expensive
serial trace paths with batched/chunked equivalents. Optional optimization
profiles expose a few deliberate recipe cuts for cheaper exploratory runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


CURRENT_DIR = Path(__file__).resolve().parent       # src/optimized/
_TRAINER_DIR = CURRENT_DIR.parent / "trainer"       # src/trainer/
for _d in [str(CURRENT_DIR), str(_TRAINER_DIR)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import gemma3_residual_paving_binary_merged as base

from optimized_binary_runtime import (
    build_prompt_variant_examples_optimized,
    collect_en_references_optimized,
    collect_fr_references_optimized,
    collect_hard_negative_references_optimized,
    compute_gate_utility_weights_optimized,
    configure_optimized_runtime,
    gather_train_traces_with_controller_batch_optimized,
)
from training_runtime_analysis import TrainingRuntimeAnalyzer, set_active_runtime_analyzer


def _normalize_argv(argv: List[str]) -> List[str]:
    normalized: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--layers" and i + 1 < len(argv):
            value = argv[i + 1]
            if value.startswith("-") and "," in value:
                normalized.append(f"--layers={value}")
                i += 2
                continue
        normalized.append(token)
        i += 1
    return normalized


def _find_subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            if name in action.choices:
                return action.choices[name]
    raise RuntimeError(f"Could not find subparser {name!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    train_parser = _find_subparser(parser, "train")
    train_parser.add_argument(
        "--trace_batch_size",
        type=int,
        default=0,
        help="Chunk size for batched trajectory tracing. Use 0 to auto-cap at 8 examples per trace chunk.",
    )
    train_parser.add_argument(
        "--trace_batch_cap",
        type=int,
        default=8,
        help="Auto-mode upper bound for trace_batch_size when --trace_batch_size=0.",
    )
    train_parser.add_argument(
        "--optimization_profile",
        choices=["faithful", "balanced", "fast"],
        default="faithful",
        help="faithful keeps the full recipe; balanced trims the highest-overhead auxiliaries; fast is for exploratory runs.",
    )
    train_parser.add_argument(
        "--frozen_reference_mode",
        choices=["exact", "compressed_cache"],
        default="exact",
        help="Use compressed cached frozen references for KL/trajectory/semantic terms instead of retracing them every epoch.",
    )
    train_parser.add_argument(
        "--frozen_reference_topk",
        type=int,
        default=64,
        help="Teacher top-k size to cache per frozen reference step when --frozen_reference_mode=compressed_cache.",
    )
    train_parser.add_argument(
        "--frozen_reference_student_topk",
        type=int,
        default=32,
        help="Student top-k size added to the union-support KL approximation when --frozen_reference_mode=compressed_cache.",
    )
    train_parser.add_argument(
        "--frozen_reference_cache_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Storage dtype for compressed frozen-reference caches.",
    )
    train_parser.add_argument(
        "--analyze_runtime",
        action="store_true",
        help="Collect wall-time and estimated FLOP breakdowns for setup, build, and training subsections.",
    )
    train_parser.add_argument(
        "--analysis_output_json",
        type=str,
        default=None,
        help="Optional JSON path for the full runtime analysis payload.",
    )
    train_parser.add_argument(
        "--analysis_top_k",
        type=int,
        default=25,
        help="How many sections to include in the printed runtime-analysis summary.",
    )
    train_parser.add_argument(
        "--analysis_no_cuda_sync",
        action="store_true",
        help="Disable CUDA synchronization around timed sections. Faster, but timing is less accurate on GPU.",
    )
    train_parser.add_argument(
        "--analysis_forward_flops_multiplier",
        type=float,
        default=2.0,
        help="Forward-only FLOP multiplier used for dense-token model-call estimates.",
    )
    train_parser.add_argument(
        "--analysis_train_flops_multiplier",
        type=float,
        default=4.0,
        help="Training-path FLOP multiplier used for dense-token model-call estimates.",
    )
    return parser


def _flag_present(argv: List[str], flag: str) -> bool:
    return any(item == flag or item.startswith(flag + "=") for item in argv)


def _apply_profile_defaults(args, argv: List[str]) -> List[str]:
    if getattr(args, "command", None) != "train":
        return []

    changes: List[str] = []
    profile = str(getattr(args, "optimization_profile", "faithful")).strip().lower()
    if profile == "faithful":
        return changes

    if profile == "balanced":
        if not _flag_present(argv, "--test_every"):
            args.test_every = 4
            changes.append("test_every=4")
        if not _flag_present(argv, "--consistency_max_variants") and not _flag_present(argv, "--consistency_weight"):
            args.consistency_max_variants = 0
            changes.append("consistency_max_variants=0")
        if not _flag_present(argv, "--stage3_epochs"):
            args.stage3_epochs = min(int(args.stage3_epochs), 6)
            changes.append(f"stage3_epochs={args.stage3_epochs}")
        return changes

    if profile == "fast":
        if not _flag_present(argv, "--test_every"):
            args.test_every = 0
            changes.append("test_every=0")
        if not _flag_present(argv, "--consistency_max_variants") and not _flag_present(argv, "--consistency_weight"):
            args.consistency_max_variants = 0
            changes.append("consistency_max_variants=0")
        if not _flag_present(argv, "--stage3_epochs"):
            args.stage3_epochs = 0
            changes.append("stage3_epochs=0")
        if not _flag_present(argv, "--disable_hard_negatives"):
            args.disable_hard_negatives = True
            changes.append("disable_hard_negatives=True")
        return changes

    raise ValueError(f"Unsupported optimization profile: {profile!r}")


def _install_runtime_patches() -> None:
    base.build_prompt_variant_examples = build_prompt_variant_examples_optimized
    base.collect_fr_references = collect_fr_references_optimized
    base.collect_en_references = collect_en_references_optimized
    base.collect_hard_negative_references = collect_hard_negative_references_optimized
    base.gather_train_traces_with_controller_batch = gather_train_traces_with_controller_batch_optimized
    base.compute_gate_utility_weights = compute_gate_utility_weights_optimized


def main(argv: List[str] | None = None) -> None:
    raw_argv = _normalize_argv(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    args = parser.parse_args(raw_argv)

    profile_changes = _apply_profile_defaults(args, raw_argv)
    configure_optimized_runtime(args)
    _install_runtime_patches()

    if args.command == "train":
        trace_mode = "auto" if int(args.trace_batch_size) <= 0 else str(int(args.trace_batch_size))
        print(
            "[optimized runtime] "
            f"profile={args.optimization_profile} "
            f"frozen_reference_mode={args.frozen_reference_mode} "
            f"trace_batch_size={trace_mode} "
            f"trace_batch_cap={int(args.trace_batch_cap)}"
        )
        if profile_changes:
            print("[optimized runtime] applied " + " ".join(profile_changes))
        analyzer = None
        if args.analyze_runtime:
            analyzer = TrainingRuntimeAnalyzer(
                enabled=True,
                sync_cuda=not bool(args.analysis_no_cuda_sync),
                forward_flops_multiplier=float(args.analysis_forward_flops_multiplier),
                train_total_flops_multiplier=float(args.analysis_train_flops_multiplier),
            )
            set_active_runtime_analyzer(analyzer)
            print(
                "[runtime analysis] "
                f"sync_cuda={'off' if args.analysis_no_cuda_sync else 'on'} "
                f"forward_mult={float(args.analysis_forward_flops_multiplier):.2f} "
                f"train_mult={float(args.analysis_train_flops_multiplier):.2f}"
            )
        try:
            base.train_binary_merged(args)
        finally:
            if analyzer is not None:
                try:
                    print(analyzer.format_summary(top_k=max(int(args.analysis_top_k), 1)))
                    if args.analysis_output_json:
                        analyzer.write_json(
                            args.analysis_output_json,
                            top_k=max(int(args.analysis_top_k), 1),
                            extra={
                                "optimization_profile": str(args.optimization_profile),
                                "trace_batch_size": int(args.trace_batch_size),
                                "trace_batch_cap": int(args.trace_batch_cap),
                            },
                        )
                        print(f"[runtime analysis] wrote {args.analysis_output_json}")
                finally:
                    set_active_runtime_analyzer(None)
        return

    if args.command == "infer":
        base.infer_binary_merged(args)
        return

    raise ValueError(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":
    main()
