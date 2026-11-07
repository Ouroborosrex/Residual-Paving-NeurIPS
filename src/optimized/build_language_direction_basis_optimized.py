#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch


CURRENT_DIR = Path(__file__).resolve().parent
TRAINER_DIR = CURRENT_DIR.parent / "trainer"
for _d in [str(CURRENT_DIR), str(TRAINER_DIR)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from gemma3_residual_paving_binary_merged import (
    ALPACA_DEFAULT_SOURCE,
    compute_language_direction_basis,
    load_dataset_track_prompt_records,
)
from gemma3_residual_paving_binary_refined import build_binary_examples
from gemma3_residual_paving_hybrid import (
    DEFAULT_MODEL_ID,
    ResidualPaver,
    batched,
    get_decoder_layers,
    load_model_and_tokenizer,
    parse_layer_indices,
    set_seed,
)
from optimized_binary_runtime import (
    configure_optimized_runtime,
    trace_forced_trajectory_batch_optimized,
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--quantization", type=str, choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument(
        "--quant_compute_dtype",
        type=str,
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        default="auto",
    )
    parser.add_argument(
        "--dataset_track",
        type=str,
        choices=[
            "parallel_french",
            "harmbench",
            "harmbench_parallel",
            "harmbench_2questions",
            "saladbench",
            "saladbench_2questions",
        ],
        default="parallel_french",
    )
    parser.add_argument("--prompt_data_path", type=str, default=None)
    parser.add_argument("--max_prompt_records", type=int, default=80)
    parser.add_argument("--harmbench_train_split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--harmbench_test_split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--harmbench_train_source", type=str, default=None)
    parser.add_argument("--harmbench_test_source", type=str, default=None)
    parser.add_argument("--alpaca_source", type=str, default=ALPACA_DEFAULT_SOURCE)
    parser.add_argument("--alpaca_split_seed", type=int, default=17)
    parser.add_argument("--harmbench_cache_dir", type=str, default=None)
    parser.add_argument("--harmbench_refresh_cache", action="store_true")
    parser.add_argument("--harmbench_bucket_field", type=str, default="SemanticCategory")
    parser.add_argument("--harmbench_translation_max_new_tokens", type=int, default=512)
    parser.add_argument("--harmbench_translation_temperature", type=float, default=0.0)
    parser.add_argument("--harmbench_translation_top_p", type=float, default=0.95)
    parser.add_argument("--harmbench_max_test_records", type=int, default=None)
    parser.add_argument("--saladbench_train_split", type=str, choices=["base", "attackEnhanced", "defenseEnhanced"], default="base")
    parser.add_argument("--saladbench_test_split", type=str, choices=["base", "attackEnhanced", "defenseEnhanced"], default="attackEnhanced")
    parser.add_argument("--saladbench_train_source", type=str, default=None)
    parser.add_argument("--saladbench_test_source", type=str, default=None)
    parser.add_argument("--saladbench_split_from_base", action="store_true")
    parser.add_argument("--saladbench_base_source", type=str, default=None)
    parser.add_argument("--saladbench_base_total_records", type=int, default=None)
    parser.add_argument("--saladbench_base_test_fraction", type=float, default=0.20)
    parser.add_argument("--saladbench_base_split_seed", type=int, default=17)
    parser.add_argument("--saladbench_bucket_level", type=str, choices=["top", "mid", "leaf"], default="top")
    parser.add_argument("--saladbench_max_test_records", type=int, default=None)
    parser.add_argument("--layers", type=str, default="-4,-3,-2,-1")
    parser.add_argument("--steer_steps", type=int, default=24)
    parser.add_argument("--max_prompt_tokens", type=int, default=0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--build_batch_size", type=int, default=8)
    parser.add_argument("--build_progress_every", type=int, default=64)
    parser.add_argument("--trace_batch_size", type=int, default=8)
    parser.add_argument("--trace_batch_cap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="data/fr_en_language_direction_basis.pt")
    return parser


@torch.no_grad()
def main() -> None:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(sys.argv[1:]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    configure_optimized_runtime(args)

    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        device,
        quantization=args.quantization,
        quant_compute_dtype=args.quant_compute_dtype,
    )
    total_layers = len(get_decoder_layers(model))
    layer_indices = parse_layer_indices(args.layers, total_layers)
    paver = ResidualPaver(model=model, layer_indices=layer_indices)

    prompt_records, _, sources, _ = load_dataset_track_prompt_records(
        args=args,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    print(f"Dataset track: {args.dataset_track}")
    print(f"Prompt source: {sources['train']}")
    print(f"Prompt records: {len(prompt_records)}")

    fr_examples, en_examples = build_binary_examples(
        model=model,
        tokenizer=tokenizer,
        paver=paver,
        device=device,
        steer_steps=args.steer_steps,
        prompt_records=prompt_records,
        use_hard_negatives=False,
        max_prompt_tokens=args.max_prompt_tokens,
        build_batch_size=args.build_batch_size,
        progress_label="basis-build",
        progress_every=args.build_progress_every,
    )

    diffs: List[torch.Tensor] = []
    chunk_size = max(1, int(args.trace_batch_size) if int(args.trace_batch_size) > 0 else int(args.trace_batch_cap))
    total_pairs = len(fr_examples)
    processed = 0
    for fr_chunk, en_chunk in zip(
        batched(list(fr_examples), chunk_size),
        batched(list(en_examples), chunk_size),
    ):
        en_logits, en_states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in en_chunk],
            anchor_ids_batch=[ex.anchor_ids for ex in en_chunk],
            controller_vectors=None,
            gate_strengths=0.0,
        )
        fr_logits, fr_states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in fr_chunk],
            anchor_ids_batch=[ex.negative_ids for ex in fr_chunk],
            controller_vectors=None,
            gate_strengths=0.0,
        )
        _ = en_logits, fr_logits
        diffs.append((en_states - fr_states).float().cpu())
        processed += len(fr_chunk)
        print(f"[basis traces] processed={processed}/{total_pairs}")

    diff_tensor = torch.cat(diffs, dim=0)
    bundle = compute_language_direction_basis(diff_tensor, rank=args.rank)
    bundle["layer_indices"] = torch.tensor(layer_indices, dtype=torch.long)
    bundle["steer_steps"] = torch.tensor(int(args.steer_steps), dtype=torch.long)
    bundle["source_size"] = torch.tensor(len(prompt_records), dtype=torch.long)
    bundle["model_id"] = args.model_id

    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    print(
        f"Saved basis to {output_path} "
        f"(rank={bundle['basis_vectors'].shape[0]}, cumulative_energy={float(bundle['explained_energy'].sum().item()):.4f})"
    )


if __name__ == "__main__":
    main()
