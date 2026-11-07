"""Compare safe-reframe edit target anchors before MoE training."""
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

# Configure the allocator only for a fresh CLI process. Importing this module
# after Torch must not rewrite PyTorch's already-parsed process-global backend.
if "torch" not in sys.modules:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import torch

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
for _d in [str(_EXP_DIR), str(_ROOT / "src" / "trainer"), str(_ROOT / "src" / "optimized")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dataset import (  # noqa: E402
    apply_anchor_prefix,
    encode_anchor_prefix_ids,
    filter_base_refusal_prompts,
    generate_ids_batch,
    load_salad_prompts,
    make_safe_reframe_prompts,
)
from gemma3_residual_paving_hybrid import encode_chat_prompt, load_model_and_tokenizer  # noqa: E402
from moe_harness import _is_operational_compliance_moe, _is_refusal_moe  # noqa: E402


DEFAULT_PREFIXES = (
    ("none", ""),
    ("at_high_level", "At a high level,"),
    ("safety_framing", "High-level safety framing:"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--model-id", default="google/gemma-3-4b-it")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--steer-steps", type=int, default=12)
    parser.add_argument("--build-batch-size", type=int, default=2)
    parser.add_argument("--require-edit-base-refusal", action="store_true")
    parser.add_argument("--edit-refusal-filter-oversample", type=int, default=20)
    parser.add_argument("--edit-refusal-filter-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        metavar="NAME=TEXT",
        help="Prefix candidate. Pass multiple times; default compares three candidates.",
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


def _parse_prefix_spec(raw: str) -> Tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid prefix spec {raw!r}. Expected NAME=TEXT.")
    name, text = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid prefix spec {raw!r}: NAME cannot be empty.")
    return name, text


def _decode(tokenizer, rows: List[torch.Tensor]) -> List[str]:
    return [
        tokenizer.decode(row.detach().cpu().tolist(), skip_special_tokens=True).strip()
        for row in rows
    ]


def _build_anchor_batch(
    model,
    tokenizer,
    device: torch.device,
    teacher_prompts: List[str],
    *,
    anchor_prefix: str,
    steer_steps: int,
    batch_size: int,
) -> List[torch.Tensor]:
    prefix_ids = encode_anchor_prefix_ids(tokenizer, anchor_prefix, device)
    prefix_steps = min(int(prefix_ids.numel()), int(steer_steps))
    suffix_steps = max(0, int(steer_steps) - prefix_steps)
    anchors: List[torch.Tensor] = []
    for start in range(0, len(teacher_prompts), max(1, int(batch_size))):
        chunk = teacher_prompts[start : start + batch_size]
        teacher_ids_batch = [
            encode_chat_prompt(tokenizer, prompt, None, device) for prompt in chunk
        ]
        suffix_base_ids_batch = teacher_ids_batch
        if prefix_steps > 0:
            prefix_for_prompt = prefix_ids[:prefix_steps]
            suffix_base_ids_batch = [
                torch.cat([teacher_ids.view(-1), prefix_for_prompt], dim=0)
                for teacher_ids in teacher_ids_batch
            ]
        if suffix_steps > 0:
            suffix_batch = generate_ids_batch(
                model,
                tokenizer,
                suffix_base_ids_batch,
                max_new_tokens=suffix_steps,
                stop_on_eos=False,
            )
        else:
            suffix_batch = [
                torch.empty(0, dtype=torch.long, device=device) for _ in teacher_ids_batch
            ]
        anchors.extend(
            apply_anchor_prefix(suffix_batch, prefix_ids, steer_steps=steer_steps)
        )
    return anchors


def _score_anchor_texts(texts: List[str], *, prefix_token_count: int) -> Dict:
    refusal_flags = [_is_refusal_moe(text) for text in texts]
    operational_flags = [_is_operational_compliance_moe(text) for text in texts]
    refusal_rate = sum(refusal_flags) / max(len(refusal_flags), 1)
    operational_rate = sum(operational_flags) / max(len(operational_flags), 1)
    safe_nonoperational_rate = max(0.0, 1.0 - refusal_rate - operational_rate)
    return {
        "examples": len(texts),
        "prefix_token_count": int(prefix_token_count),
        "anchor_refusal_rate": float(refusal_rate),
        "anchor_operational_compliance_rate": float(operational_rate),
        "anchor_nonrefusal_nonoperational_rate": float(safe_nonoperational_rate),
        "target_anchor_score": float(safe_nonoperational_rate - operational_rate),
        "anchor_previews": texts[:3],
    }


def _write_run_note(run_root: Path, summary: Dict) -> None:
    lines = [
        "# Target Anchor Smoke",
        "",
        f"- command: `{summary['command']}`",
        f"- timestamp: `{summary['timestamp']}`",
        f"- git_commit: `{summary['git_commit']}`",
        f"- status: `{summary['status']}`",
        f"- prefixes: `{', '.join(summary.get('prefixes', []))}`",
    ]
    if summary.get("anomalies"):
        lines.append(f"- anomalies: `{'; '.join(summary['anomalies'])}`")
    if "elapsed_s" in summary:
        lines.append(f"- elapsed_s: `{summary['elapsed_s']:.2f}`")
    run_root.joinpath("run_note.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    prefix_specs = [_parse_prefix_spec(raw) for raw in args.prefix] if args.prefix else list(DEFAULT_PREFIXES)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_target_anchor_smoke")
    run_root = _ROOT / "experiments" / "runs" / timestamp
    run_root.mkdir(parents=True, exist_ok=False)
    command = " ".join([sys.executable] + sys.argv)
    git_commit = _git_commit()
    anomalies = []
    if git_commit == "unknown":
        anomalies.append("git_commit unavailable; repository appears to have no commits")

    summary = {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "status": "running",
        "command": command,
        "prefixes": [name for name, _ in prefix_specs],
        "anomalies": anomalies,
    }
    _write_json(run_root / "summary.json", summary)
    _write_run_note(run_root, summary)

    print(f"\nTarget-anchor smoke run: {run_root}")
    t0 = time.time()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_model_and_tokenizer(str(args.model_id), device=device)
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        n = int(args.n)
        candidate_count = n
        if args.require_edit_base_refusal:
            candidate_count = max(n, n * max(1, int(args.edit_refusal_filter_oversample)))
        edit_prompts = load_salad_prompts(candidate_count, seed=int(args.data_seed))
        if args.require_edit_base_refusal:
            print(
                "[data] filtering edit prompts to base refusals "
                f"(candidates={len(edit_prompts)} target={n}) ..."
            )
            edit_prompts = filter_base_refusal_prompts(
                model,
                tokenizer,
                device,
                edit_prompts,
                max_new_tokens=int(args.edit_refusal_filter_max_new_tokens),
                batch_size=max(1, int(args.build_batch_size)),
                required_count=n,
            )
        edit_prompts = edit_prompts[:n]

        results = {}
        for name, prefix in prefix_specs:
            teacher_prompts = make_safe_reframe_prompts(
                edit_prompts,
                anchor_prefix=prefix,
            )
            anchors = _build_anchor_batch(
                model,
                tokenizer,
                device,
                teacher_prompts,
                anchor_prefix=prefix,
                steer_steps=int(args.steer_steps),
                batch_size=int(args.build_batch_size),
            )
            prefix_token_count = int(
                encode_anchor_prefix_ids(tokenizer, prefix, device).numel()
            )
            results[name] = _score_anchor_texts(
                _decode(tokenizer, anchors),
                prefix_token_count=prefix_token_count,
            )
            results[name]["anchor_prefix"] = prefix

        elapsed_s = time.time() - t0
        summary.update(
            {
                "status": "ok",
                "elapsed_s": elapsed_s,
                "n": len(edit_prompts),
                "steer_steps": int(args.steer_steps),
                "require_edit_base_refusal": bool(args.require_edit_base_refusal),
                "results": results,
            }
        )
        _write_json(run_root / "summary.json", summary)
        _write_run_note(run_root, summary)

        print("\nTarget-anchor smoke summary:")
        print(f"  {'Prefix':<18} {'Refusal':>8} {'Oper':>8} {'SafeNR':>8} {'Score':>8} {'Tok':>5}")
        print(f"  {'-' * 65}")
        for name, result in results.items():
            print(
                f"  {name:<18} "
                f"{result['anchor_refusal_rate']:>8.3f} "
                f"{result['anchor_operational_compliance_rate']:>8.3f} "
                f"{result['anchor_nonrefusal_nonoperational_rate']:>8.3f} "
                f"{result['target_anchor_score']:>8.3f} "
                f"{result['prefix_token_count']:>5d}"
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
        _write_run_note(run_root, summary)
        raise


if __name__ == "__main__":
    main()
