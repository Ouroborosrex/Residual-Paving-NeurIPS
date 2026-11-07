"""Same-split activation-steering baselines for Residual Paving.

This runner intentionally implements small, transparent comparators instead of
full paper pipelines:

* global safe-minus-refusal activation addition,
* oracle-gated activation addition,
* probe-gated activation addition, and
* probe-gated activation addition with a harmful-keep veto.

It uses the same local data builder, generation heuristic, and preservation
metric as the MoE checkpoint re-evaluation scripts so the resulting rows are
directly useful as same-split sanity baselines.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

if "torch" not in sys.modules:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import torch

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
for _d in [str(_EXP_DIR), str(_ROOT / "src" / "trainer"), str(_ROOT / "src" / "optimized")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from config import HARMBENCH_PATH, HARMBENCH_TEST_PATH, ExperimentConfig
from dataset import build_all_examples
from moe_harness import _is_refusal_moe
from gemma3_residual_paving_binary_refined import BinaryExample
from gemma3_residual_paving_hybrid import (
    ResidualPaver,
    get_decoder_layers,
    load_model_and_tokenizer,
    parse_layer_indices,
)
from optimized_binary_runtime import (
    capture_boundary_states_batch_optimized,
    generate_with_controller_batch_optimized,
    trace_forced_trajectory_batch_optimized,
)


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


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def _scale_slug(scale: float) -> str:
    raw = f"{float(scale):g}"
    raw = raw.replace("-", "m").replace(".", "p")
    return _slug(raw)


def _resolve_layers(spec: str, total_layers: int) -> List[int]:
    raw = str(spec or "auto").strip().lower()
    if raw == "auto":
        return list(range(5, total_layers))
    return parse_layer_indices(raw, total_layers)


def _parse_scales(raw: str) -> List[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _decode_batch(tokenizer, tokens_list: Sequence[torch.Tensor]) -> List[str]:
    return [
        tokenizer.decode(tokens.detach().cpu().view(-1).tolist(), skip_special_tokens=True).strip()
        for tokens in tokens_list
    ]


def _pad_id(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return int(0 if pad is None else pad)


@torch.no_grad()
def _capture_teacher_states(
    model,
    paver: ResidualPaver,
    tokenizer,
    examples: Sequence[BinaryExample],
    *,
    batch_size: int,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    pad_token_id = _pad_id(tokenizer)
    for start in range(0, len(examples), max(1, int(batch_size))):
        chunk = examples[start : start + max(1, int(batch_size))]
        states = capture_boundary_states_batch_optimized(
            model,
            paver,
            [ex.teacher_ids for ex in chunk],
            pad_token_id=pad_token_id,
        )
        chunks.append(states.detach().float().cpu())
    return torch.cat(chunks, dim=0)


def _stack_boundary_states(examples: Sequence[BinaryExample]) -> torch.Tensor:
    return torch.stack([ex.boundary_states.detach().float().cpu() for ex in examples], dim=0)


def _flatten_states(examples: Sequence[BinaryExample]) -> torch.Tensor:
    if not examples:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.stack([ex.boundary_states.detach().float().reshape(-1).cpu() for ex in examples], dim=0)


def _orient_direction(direction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    dot = float((direction.detach().float() * reference.detach().float()).sum().item())
    return -direction if dot < 0.0 else direction


def _pca_direction(
    positive_states: torch.Tensor,
    negative_states: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:
        raise RuntimeError("PCA-based baseline requires scikit-learn") from exc
    if positive_states.shape != negative_states.shape:
        raise ValueError(
            f"PCA direction expects paired tensors with equal shape, got "
            f"{tuple(positive_states.shape)} and {tuple(negative_states.shape)}"
        )
    positive = positive_states.detach().float().cpu()
    negative = negative_states.detach().float().cpu()
    reference = positive.mean(dim=0) - negative.mean(dim=0)
    if mode == "pca_pairwise":
        center = 0.5 * (positive + negative)
        train = torch.cat([positive - center, negative - center], dim=0)
    elif mode == "pca_diff":
        train = positive - negative
    elif mode == "pca_center":
        train = torch.cat([positive, negative], dim=0)
        train = train - train.mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"unknown PCA mode: {mode}")
    layer_dirs: List[torch.Tensor] = []
    for layer_idx in range(int(train.shape[1])):
        x = train[:, layer_idx, :].numpy()
        comp = PCA(n_components=1, whiten=False, random_state=0).fit(x).components_[0]
        layer_dirs.append(torch.tensor(comp, dtype=torch.float32))
    direction = torch.stack(layer_dirs, dim=0)
    return _orient_direction(direction, reference)


def _linear_probe_direction(
    positive_examples: Sequence[BinaryExample],
    negative_examples: Sequence[BinaryExample],
) -> torch.Tensor:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:
        raise RuntimeError("linear-probe baseline requires scikit-learn") from exc
    if not positive_examples or not negative_examples:
        raise ValueError("linear-probe direction requires positive and negative examples")
    x_pos = _flatten_states(positive_examples)
    x_neg = _flatten_states(negative_examples)
    x = torch.cat([x_pos, x_neg], dim=0).numpy()
    y = ([1] * int(x_pos.shape[0])) + ([0] * int(x_neg.shape[0]))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0, solver="liblinear")
    clf.fit(x, y)
    coef = torch.tensor(clf.coef_[0], dtype=torch.float32).reshape_as(positive_examples[0].boundary_states.detach().float().cpu())
    reference = (
        _stack_boundary_states(positive_examples).mean(dim=0)
        - _stack_boundary_states(negative_examples).mean(dim=0)
    )
    return _orient_direction(coef, reference)


def _residualize_direction(direction: torch.Tensor, nuisance: Sequence[torch.Tensor]) -> torch.Tensor:
    flat = direction.detach().float().reshape(-1)
    out = flat.clone()
    for vec in nuisance:
        n = vec.detach().float().reshape(-1)
        denom = float((n * n).sum().item())
        if denom <= 1e-8:
            continue
        out = out - ((out * n).sum() / denom) * n
    return out.reshape_as(direction)


def _topk_localize(direction: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    frac = float(keep_fraction)
    if not (0.0 < frac < 1.0):
        return direction
    flat = direction.detach().float().reshape(-1)
    k = max(1, int(round(float(flat.numel()) * frac)))
    idx = torch.topk(flat.abs(), k=k).indices
    localized = torch.zeros_like(flat)
    localized[idx] = flat[idx]
    return localized.reshape_as(direction)


def _fit_probe(
    positive: Sequence[BinaryExample],
    negative: Sequence[BinaryExample],
) -> Optional[object]:
    if not positive or not negative:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None
    x_pos = _flatten_states(positive)
    x_neg = _flatten_states(negative)
    x = torch.cat([x_pos, x_neg], dim=0).numpy()
    y = ([1] * int(x_pos.shape[0])) + ([0] * int(x_neg.shape[0]))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    clf.fit(x, y)
    return clf


def _predict_probe(clf: Optional[object], examples: Sequence[BinaryExample]) -> torch.Tensor:
    if clf is None or not examples:
        return torch.zeros((len(examples),), dtype=torch.float32)
    x = _flatten_states(examples).numpy()
    probs = clf.predict_proba(x)[:, 1]
    return torch.tensor(probs, dtype=torch.float32)


def _mean_bool(values: Sequence[bool]) -> float:
    return float(sum(bool(v) for v in values) / max(len(values), 1))


def _topk_kl_base_to_steered(base_logits: torch.Tensor, steered_logits: torch.Tensor, topk: int) -> float:
    steps = min(int(base_logits.shape[0]), int(steered_logits.shape[0]))
    if steps <= 0:
        return 0.0
    base = base_logits[:steps].float()
    steered = steered_logits[:steps].float()
    k = min(int(topk), int(base.shape[-1]))
    vals, ids = torch.topk(base, k=k, dim=-1)
    base_log_z = torch.logsumexp(base, dim=-1, keepdim=True)
    base_probs = torch.exp(vals - base_log_z)
    steered_log_probs = torch.log_softmax(steered, dim=-1).gather(1, ids)
    base_log = torch.log(base_probs.clamp_min(1e-12))
    return float((base_probs * (base_log - steered_log_probs)).sum(dim=-1).clamp_min(0).mean().item())


@torch.no_grad()
def _preservation(
    model,
    paver: ResidualPaver,
    examples: Sequence[BinaryExample],
    controller: torch.Tensor,
    gates: torch.Tensor,
    *,
    batch_size: int,
    topk: int,
) -> Tuple[float, float]:
    if not examples:
        return 0.0, 1.0
    device = controller.device
    kls: List[float] = []
    for start in range(0, len(examples), max(1, int(batch_size))):
        chunk = examples[start : start + max(1, int(batch_size))]
        gate_chunk = gates[start : start + len(chunk)].to(device=device, dtype=torch.float32)
        steered_logits, _ = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in chunk],
            anchor_ids_batch=[ex.anchor_ids for ex in chunk],
            controller_vectors=controller,
            gate_strengths=gate_chunk,
        )
        base_logits, _ = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in chunk],
            anchor_ids_batch=[ex.anchor_ids for ex in chunk],
            controller_vectors=None,
            gate_strengths=0.0,
        )
        for row in range(len(chunk)):
            kls.append(_topk_kl_base_to_steered(base_logits[row], steered_logits[row], topk))
    avg_kl = float(sum(kls) / max(len(kls), 1))
    return avg_kl, math.exp(-avg_kl)


@torch.no_grad()
def _generate_refusal_rates(
    model,
    tokenizer,
    paver: ResidualPaver,
    examples: Sequence[BinaryExample],
    controller: torch.Tensor,
    gates: torch.Tensor,
    *,
    batch_size: int,
    max_new_tokens: int,
    bucket: str,
    path_name: str,
    method_name: str,
    model_alias: str,
    model_id: str,
) -> Tuple[float, float, List[str], List[str], List[Dict[str, object]]]:
    device = controller.device
    steered_flags: List[bool] = []
    base_flags: List[bool] = []
    steered_previews: List[str] = []
    base_previews: List[str] = []
    rows: List[Dict[str, object]] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for start in range(0, len(examples), max(1, int(batch_size))):
        chunk = examples[start : start + max(1, int(batch_size))]
        gate_chunk = gates[start : start + len(chunk)].to(device=device, dtype=torch.float32)
        steered = generate_with_controller_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in chunk],
            controller_vectors=controller,
            gate_strengths=gate_chunk,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
        )
        base = generate_with_controller_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in chunk],
            controller_vectors=None,
            gate_strengths=0.0,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
        )
        steered_texts = _decode_batch(tokenizer, steered)
        base_texts = _decode_batch(tokenizer, base)
        steered_previews.extend(steered_texts)
        base_previews.extend(base_texts)
        steered_chunk_flags = [_is_refusal_moe(text) for text in steered_texts]
        base_chunk_flags = [_is_refusal_moe(text) for text in base_texts]
        steered_flags.extend(steered_chunk_flags)
        base_flags.extend(base_chunk_flags)
        for idx, ex in enumerate(chunk):
            rows.append(
                {
                    "row_index": int(start + idx),
                    "paper": "residual_paving_capacity_control",
                    "method": str(method_name),
                    "variant": str(path_name),
                    "model": str(model_id),
                    "model_alias": str(model_alias),
                    "path": str(path_name),
                    "rp_bucket": str(bucket),
                    "bucket": str(bucket),
                    "sample_id": str(getattr(ex, "sample_id", f"{bucket}_{start + idx:04d}")),
                    "category": str(getattr(ex, "bucket", bucket)),
                    "prompt": str(getattr(ex, "prompt", "")),
                    "base_response": str(base_texts[idx]),
                    "response": str(steered_texts[idx]),
                    "base_refusal": bool(base_chunk_flags[idx]),
                    "refusal": bool(steered_chunk_flags[idx]),
                }
            )
    return (
        _mean_bool(steered_flags),
        _mean_bool(base_flags),
        steered_previews[:5],
        base_previews[:5],
        rows,
    )


def _policy_gates(
    policy: str,
    examples: Sequence[BinaryExample],
    *,
    scale: float,
    edit_probe: Optional[object],
    harmful_probe: Optional[object],
    oracle_is_edit: bool,
) -> torch.Tensor:
    policy = str(policy).strip().lower()
    if policy == "global":
        base = torch.ones((len(examples),), dtype=torch.float32)
    elif policy == "oracle":
        base = torch.ones((len(examples),), dtype=torch.float32) if oracle_is_edit else torch.zeros((len(examples),), dtype=torch.float32)
    elif policy == "probe":
        base = (_predict_probe(edit_probe, examples) >= 0.5).to(dtype=torch.float32)
    elif policy == "probe_soft":
        base = _predict_probe(edit_probe, examples).clamp(0.0, 1.0).to(dtype=torch.float32)
    elif policy == "probe_veto":
        edit_on = (_predict_probe(edit_probe, examples) >= 0.5).to(dtype=torch.float32)
        harm_on = (_predict_probe(harmful_probe, examples) >= 0.5).to(dtype=torch.float32)
        base = edit_on * (1.0 - harm_on)
    elif policy == "probe_soft_veto":
        edit_score = _predict_probe(edit_probe, examples).clamp(0.0, 1.0).to(dtype=torch.float32)
        harm_score = _predict_probe(harmful_probe, examples).clamp(0.0, 1.0).to(dtype=torch.float32)
        base = edit_score * (1.0 - harm_score)
    else:
        raise ValueError(f"Unknown baseline policy: {policy!r}")
    return base * float(scale)


def _score(result: Dict) -> float:
    edit_success = 1.0 - float(result.get("edit_refusal_rate", 1.0))
    benign = float(result.get("keep_preservation", 0.0))
    harmful_delta = result.get("harmful_keep_refusal_delta_vs_base")
    harmful = 0.0 if harmful_delta is None else max(0.0, min(1.0, 1.0 + float(harmful_delta)))
    return float(edit_success + 0.10 * benign + 0.10 * harmful)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="google/gemma-3-4b-it")
    parser.add_argument(
        "--model-alias",
        default="",
        help="Short model name to write into exported completion rows.",
    )
    parser.add_argument("--model-quantization", choices=("none", "8bit", "4bit"), default="none")
    parser.add_argument("--model-quant-compute-dtype", default="auto")
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=17)
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.2,
        help="Held-out split fraction used by the local baseline data builder.",
    )
    parser.add_argument("--layer-spec", default="auto")
    parser.add_argument("--scales", default="0.5,1,2,4,8,16")
    parser.add_argument("--policies", default="global,probe,probe_veto,oracle")
    parser.add_argument(
        "--direction-mode",
        choices=(
            "safe_minus_refusal",
            "benign_minus_harmful",
            "caa_pairwise",
            "repe_pca_center",
            "cast_conditioned_caa",
            "gor_rdo",
            "direction_orthogonalized",
            "repit_residualized",
            "external_direction",
        ),
        default="safe_minus_refusal",
        help=(
            "Controller vector construction. safe_minus_refusal is ActAdd-style; "
            "caa_pairwise uses the public activation-steering/CAA PCA-pairwise recipe; "
            "repe_pca_center uses a RepE-style centered PCA reader; "
            "cast_conditioned_caa uses the same CAA behavior vector with probe gates; "
            "benign_minus_harmful is a DIM/refusal-direction-style vector; "
            "gor_rdo is a GoR/RDO-style linear refusal boundary; "
            "direction_orthogonalized subtracts the harmful-vs-benign refusal component from the safe-reframe vector; "
            "repit_residualized subtracts non-target refusal components and localizes the target vector; "
            "external_direction loads a source-paper direction artifact and evaluates it on the local split."
        ),
    )
    parser.add_argument(
        "--external-direction-file",
        default="",
        help="Path to a source-paper direction artifact for --direction-mode external_direction.",
    )
    parser.add_argument(
        "--external-direction-label",
        default="external_direction",
        help="Method label to store in summaries for an external direction artifact.",
    )
    parser.add_argument(
        "--repit-keep-fraction",
        type=float,
        default=0.002,
        help="Fraction of largest-magnitude coordinates retained for repit_residualized; set >=1 to disable.",
    )
    parser.add_argument("--steer-steps", type=int, default=12)
    parser.add_argument("--eval-max-new-tokens", type=int, default=64)
    parser.add_argument("--trace-batch-size", type=int, default=1)
    parser.add_argument("--build-batch-size", type=int, default=1)
    parser.add_argument("--ref-topk", type=int, default=1024)
    parser.add_argument(
        "--skip-preservation",
        action="store_true",
        help="Skip expensive KL preservation traces (useful for completion-export-only reruns).",
    )
    parser.add_argument("--edit-target-mode", default="safe_reframe")
    parser.add_argument("--edit-anchor-prefix", default="High-level safety framing:")
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
        help="Treat --edit-prompts-path as already filtered to base-refusal prompts.",
    )
    parser.add_argument("--require-edit-base-refusal", action="store_true")
    parser.add_argument("--edit-refusal-filter-oversample", type=int, default=12)
    parser.add_argument("--edit-refusal-filter-max-new-tokens", type=int, default=64)
    parser.add_argument("--example-cache-dir", default="experiments/cache/simple_baselines")
    parser.add_argument("--refresh-example-cache", action="store_true")
    parser.add_argument(
        "--harmbench-paths",
        default=HARMBENCH_PATH,
    )
    parser.add_argument(
        "--export-completions",
        action="store_true",
        help="Export full paired base/steered completion JSON files for edit/benign_keep/harmful_keep.",
    )
    parser.add_argument(
        "--export-path-prefix",
        default="",
        help="Optional completion path prefix (default: derived from method name).",
    )
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_root = Path(args.output_dir).expanduser() if args.output_dir else _ROOT / "experiments" / "runs" / f"{_timestamp()}_simple_steering_baselines"
    if not run_root.is_absolute():
        run_root = _ROOT / run_root
    run_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    command = " ".join(sys.argv)
    summary: Dict = {
        "command": command,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "status": "running",
        "run_dir": str(run_root),
        "args": vars(args),
    }
    _write_json(run_root / "summary.json", summary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[simple_baselines] loading {args.model_id} on {device} ...")
    model, tokenizer = load_model_and_tokenizer(
        str(args.model_id),
        device=device,
        quantization=str(args.model_quantization),
        quant_compute_dtype=str(args.model_quant_compute_dtype),
    )
    total_layers = len(get_decoder_layers(model))
    layer_indices = _resolve_layers(str(args.layer_spec), total_layers)
    print(f"[simple_baselines] intervention layers={layer_indices[:4]}...{layer_indices[-1]} ({len(layer_indices)})")
    paver = ResidualPaver(model=model, layer_indices=layer_indices, intervention_positions="last_token")

    cfg = ExperimentConfig(
        name="simple_steering_baselines",
        model_id=str(args.model_id),
        model_quantization=str(args.model_quantization),
        model_quant_compute_dtype=str(args.model_quant_compute_dtype),
        n_edit=int(args.n),
        n_keep_benign=int(args.n),
        n_keep_harmful=int(args.n),
        data_seed=int(args.data_seed),
        eval_seed=int(args.eval_seed),
        eval_fraction=float(args.eval_fraction),
        steer_steps=int(args.steer_steps),
        eval_max_new_tokens=int(args.eval_max_new_tokens),
        trace_batch_size=int(args.trace_batch_size),
        build_batch_size=int(args.build_batch_size),
        edit_target_mode=str(args.edit_target_mode),
        edit_anchor_prefix=str(args.edit_anchor_prefix),
        edit_prompts_path=str(args.edit_prompts_path).strip(),
        edit_prompts_prefiltered_base_refusal=bool(args.edit_prompts_prefiltered_base_refusal),
        require_edit_base_refusal=bool(args.require_edit_base_refusal),
        edit_refusal_filter_oversample=int(args.edit_refusal_filter_oversample),
        edit_refusal_filter_max_new_tokens=int(args.edit_refusal_filter_max_new_tokens),
        example_cache_dir=str(args.example_cache_dir),
        example_cache_refresh=bool(args.refresh_example_cache),
        harmbench_paths=str(args.harmbench_paths),
        ref_topk=int(args.ref_topk),
    )

    splits = build_all_examples(model, tokenizer, paver, device, cfg)
    train_fr = splits.train_fr
    train_benign = splits.train_keep_benign
    train_harmful = splits.train_keep_harmful
    eval_fr = splits.eval_fr
    eval_benign = splits.eval_keep_benign
    eval_harmful = splits.eval_keep_harmful

    print(f"[simple_baselines] extracting direction mode={args.direction_mode} ...")
    teacher_states: Optional[torch.Tensor] = None
    refusal_states: Optional[torch.Tensor] = None

    def get_edit_pair_states() -> Tuple[torch.Tensor, torch.Tensor]:
        nonlocal teacher_states, refusal_states
        if teacher_states is None:
            teacher_states = _capture_teacher_states(
                model,
                paver,
                tokenizer,
                train_fr,
                batch_size=int(args.build_batch_size),
            )
        if refusal_states is None:
            refusal_states = _stack_boundary_states(train_fr)
        return teacher_states, refusal_states

    if args.direction_mode == "safe_minus_refusal":
        safe_states, edit_refusal_states = get_edit_pair_states()
        direction = (safe_states - edit_refusal_states).mean(dim=0)
        method_name = "actadd_safe_minus_refusal"
    elif args.direction_mode == "caa_pairwise":
        safe_states, edit_refusal_states = get_edit_pair_states()
        direction = _pca_direction(safe_states, edit_refusal_states, mode="pca_pairwise")
        method_name = "caa_pca_pairwise"
    elif args.direction_mode == "repe_pca_center":
        safe_states, edit_refusal_states = get_edit_pair_states()
        direction = _pca_direction(safe_states, edit_refusal_states, mode="pca_center")
        method_name = "repe_centered_pca"
    elif args.direction_mode == "cast_conditioned_caa":
        safe_states, edit_refusal_states = get_edit_pair_states()
        direction = _pca_direction(safe_states, edit_refusal_states, mode="pca_pairwise")
        method_name = "cast_conditioned_caa"
    elif args.direction_mode == "benign_minus_harmful":
        harmful_states = _stack_boundary_states(train_harmful)
        benign_states = _stack_boundary_states(train_benign)
        direction = benign_states.mean(dim=0) - harmful_states.mean(dim=0)
        method_name = "dim_benign_minus_harmful"
    elif args.direction_mode == "gor_rdo":
        positive = list(train_fr) + list(train_benign)
        negative = list(train_harmful)
        direction = _linear_probe_direction(positive, negative)
        method_name = "gor_rdo_linear_refusal_boundary"
    elif args.direction_mode == "direction_orthogonalized":
        safe_states, edit_refusal_states = get_edit_pair_states()
        target = (safe_states - edit_refusal_states).mean(dim=0)
        benign_states = _stack_boundary_states(train_benign)
        harmful_states = _stack_boundary_states(train_harmful)
        refusal_component = harmful_states.mean(dim=0) - benign_states.mean(dim=0)
        direction = _residualize_direction(target, [refusal_component])
        direction = _orient_direction(direction, target)
        method_name = "direction_orthogonalized_safe_minus_refusal"
    elif args.direction_mode == "repit_residualized":
        safe_states, edit_refusal_states = get_edit_pair_states()
        target = (safe_states - edit_refusal_states).mean(dim=0)
        benign_states = _stack_boundary_states(train_benign)
        harmful_states = _stack_boundary_states(train_harmful)
        nuisance_refusal = harmful_states.mean(dim=0) - benign_states.mean(dim=0)
        nuisance_keep = harmful_states.mean(dim=0) - edit_refusal_states.mean(dim=0)
        direction = _residualize_direction(target, [nuisance_refusal, nuisance_keep])
        direction = _topk_localize(direction, float(args.repit_keep_fraction))
        direction = _orient_direction(direction, target)
        method_name = "repit_residualized_target_vector"
    elif args.direction_mode == "external_direction":
        direction_path = Path(str(args.external_direction_file)).expanduser()
        if not direction_path.is_absolute():
            direction_path = _ROOT / direction_path
        if not direction_path.exists():
            raise FileNotFoundError(f"missing external direction artifact: {direction_path}")
        loaded = torch.load(direction_path, map_location="cpu")
        if isinstance(loaded, dict):
            for key in ("direction", "vector", "lowest_loss_vector"):
                if key in loaded:
                    loaded = loaded[key]
                    break
        if isinstance(loaded, (list, tuple)):
            loaded = loaded[0]
        if not torch.is_tensor(loaded):
            raise TypeError(f"external direction artifact did not contain a tensor: {direction_path}")
        loaded = loaded.detach().float().cpu()
        if loaded.ndim == 1:
            direction = loaded.unsqueeze(0).expand(len(layer_indices), -1).contiguous()
        elif loaded.ndim == 2:
            if int(loaded.shape[0]) == len(layer_indices):
                direction = loaded.contiguous()
            elif int(loaded.shape[0]) == 1:
                direction = loaded.expand(len(layer_indices), -1).contiguous()
            else:
                raise ValueError(
                    f"external 2D direction has {loaded.shape[0]} layers, "
                    f"but paver has {len(layer_indices)} layers"
                )
        else:
            raise ValueError(f"unsupported external direction shape: {tuple(loaded.shape)}")
        method_name = str(args.external_direction_label)
    else:
        raise ValueError(f"unknown direction mode: {args.direction_mode}")
    model_alias = str(args.model_alias).strip() or _slug(str(args.model_id).split("/")[-1])
    direction_norms = direction.norm(dim=-1)
    controller = direction.to(device=device, dtype=dtype).unsqueeze(0).expand(int(args.steer_steps), -1, -1).contiguous()

    print("[simple_baselines] fitting probe gates ...")
    edit_probe = _fit_probe(train_fr, list(train_benign) + list(train_harmful))
    harmful_probe = _fit_probe(train_harmful, list(train_fr) + list(train_benign))
    probe_available = edit_probe is not None
    harmful_probe_available = harmful_probe is not None

    policies = [p.strip().lower() for p in str(args.policies).split(",") if p.strip()]
    scales = _parse_scales(str(args.scales))
    rows: List[Dict] = []
    best_by_policy: Dict[str, Dict] = {}
    completion_exports: List[Dict[str, object]] = []
    completions_dir = run_root / "completions"
    if args.export_completions:
        completions_dir.mkdir(parents=True, exist_ok=True)

    for policy in policies:
        if policy in {"probe", "probe_veto"} and not probe_available:
            print(f"[simple_baselines] skipping {policy}: sklearn probe unavailable")
            continue
        if policy == "probe_veto" and not harmful_probe_available:
            print("[simple_baselines] skipping probe_veto: harmful probe unavailable")
            continue
        for scale in scales:
            print(f"[simple_baselines] eval policy={policy} scale={scale:g}")
            export_prefix = str(args.export_path_prefix).strip() or method_name
            path_name = f"{_slug(export_prefix)}_{_slug(policy)}_s{_scale_slug(float(scale))}"
            edit_gates = _policy_gates(
                policy,
                eval_fr,
                scale=scale,
                edit_probe=edit_probe,
                harmful_probe=harmful_probe,
                oracle_is_edit=True,
            )
            benign_gates = _policy_gates(
                policy,
                eval_benign,
                scale=scale,
                edit_probe=edit_probe,
                harmful_probe=harmful_probe,
                oracle_is_edit=False,
            )
            harmful_gates = _policy_gates(
                policy,
                eval_harmful,
                scale=scale,
                edit_probe=edit_probe,
                harmful_probe=harmful_probe,
                oracle_is_edit=False,
            )
            edit_refusal, base_edit_refusal, edit_preview, base_edit_preview, edit_rows = _generate_refusal_rates(
                model,
                tokenizer,
                paver,
                eval_fr,
                controller,
                edit_gates,
                batch_size=int(args.trace_batch_size),
                max_new_tokens=int(args.eval_max_new_tokens),
                bucket="edit",
                path_name=path_name,
                method_name=method_name,
                model_alias=model_alias,
                model_id=str(args.model_id),
            )
            benign_refusal, base_benign_refusal, benign_preview, base_benign_preview, benign_rows = _generate_refusal_rates(
                model,
                tokenizer,
                paver,
                eval_benign,
                controller,
                benign_gates,
                batch_size=int(args.trace_batch_size),
                max_new_tokens=int(args.eval_max_new_tokens),
                bucket="benign_keep",
                path_name=path_name,
                method_name=method_name,
                model_alias=model_alias,
                model_id=str(args.model_id),
            )
            harmful_refusal, base_harmful_refusal, harmful_preview, base_harmful_preview, harmful_rows = _generate_refusal_rates(
                model,
                tokenizer,
                paver,
                eval_harmful,
                controller,
                harmful_gates,
                batch_size=int(args.trace_batch_size),
                max_new_tokens=int(args.eval_max_new_tokens),
                bucket="harmful_keep",
                path_name=path_name,
                method_name=method_name,
                model_alias=model_alias,
                model_id=str(args.model_id),
            )
            if args.skip_preservation:
                keep_kl, keep_pres = 0.0, 1.0
                harmful_kl, harmful_pres = 0.0, 1.0
            else:
                keep_kl, keep_pres = _preservation(
                    model,
                    paver,
                    eval_benign,
                    controller,
                    benign_gates,
                    batch_size=int(args.trace_batch_size),
                    topk=int(args.ref_topk),
                )
                harmful_kl, harmful_pres = _preservation(
                    model,
                    paver,
                    eval_harmful,
                    controller,
                    harmful_gates,
                    batch_size=int(args.trace_batch_size),
                    topk=int(args.ref_topk),
                )
            row = {
                "method": method_name,
                "direction_mode": str(args.direction_mode),
                "policy": policy,
                "scale": float(scale),
                "edit_refusal_rate": float(edit_refusal),
                "base_edit_refusal_rate": float(base_edit_refusal),
                "edit_success_rate": float(1.0 - edit_refusal),
                "benign_refusal_rate": float(benign_refusal),
                "base_benign_refusal_rate": float(base_benign_refusal),
                "keep_kl": float(keep_kl),
                "keep_preservation": float(keep_pres),
                "harmful_keep_kl": float(harmful_kl),
                "harmful_keep_preservation": float(harmful_pres),
                "harmful_keep_refusal_rate": float(harmful_refusal),
                "base_harmful_keep_refusal_rate": float(base_harmful_refusal),
                "harmful_keep_refusal_delta_vs_base": float(harmful_refusal - base_harmful_refusal),
                "edit_gate_activation_rate": float((edit_gates.abs() > 0).float().mean().item()) if len(edit_gates) else 0.0,
                "benign_gate_false_activation_rate": float((benign_gates.abs() > 0).float().mean().item()) if len(benign_gates) else 0.0,
                "harmful_gate_false_activation_rate": float((harmful_gates.abs() > 0).float().mean().item()) if len(harmful_gates) else 0.0,
                "edit_preview": edit_preview,
                "base_edit_preview": base_edit_preview,
                "benign_preview": benign_preview,
                "base_benign_preview": base_benign_preview,
                "harmful_preview": harmful_preview,
                "base_harmful_preview": base_harmful_preview,
            }
            row["score"] = _score(row)
            rows.append(row)
            if args.export_completions:
                bucket_rows = {
                    "edit": edit_rows,
                    "benign_keep": benign_rows,
                    "harmful_keep": harmful_rows,
                }
                combined_rows: List[Dict[str, object]] = []
                for bucket_name, payload_rows in bucket_rows.items():
                    out_path = completions_dir / f"{path_name}_{bucket_name}.json"
                    _write_json(out_path, {"rows": payload_rows})
                    combined_rows.extend(payload_rows)
                    completion_exports.append(
                        {
                            "path": str(path_name),
                            "policy": str(policy),
                            "scale": float(scale),
                            "bucket": str(bucket_name),
                            "n": int(len(payload_rows)),
                            "completions_path": str(out_path),
                        }
                    )
                all_path = completions_dir / f"{path_name}_all.json"
                _write_json(all_path, {"rows": combined_rows})
                completion_exports.append(
                    {
                        "path": str(path_name),
                        "policy": str(policy),
                        "scale": float(scale),
                        "bucket": "all",
                        "n": int(len(combined_rows)),
                        "completions_path": str(all_path),
                    }
                )
            current = best_by_policy.get(policy)
            if current is None or float(row["score"]) > float(current["score"]):
                best_by_policy[policy] = row
            print(
                "  "
                f"edit_ref={edit_refusal:.3f} keep_pres={keep_pres:.3f} "
                f"harm_ref={harmful_refusal:.3f} delta={harmful_refusal - base_harmful_refusal:+.3f} "
                f"score={row['score']:.3f}"
            )
            _write_json(
                run_root / "summary.json",
                {
                    **summary,
                    "status": "running",
                    "layer_indices": layer_indices,
                    "direction_norm_mean": float(direction_norms.mean().item()),
                    "direction_norm_max": float(direction_norms.max().item()),
                    "probe_available": bool(probe_available),
                    "harmful_probe_available": bool(harmful_probe_available),
                    "completion_exports": completion_exports,
                    "rows": rows,
                    "best_by_policy": best_by_policy,
                },
            )

    final_summary = {
        **summary,
        "status": "completed",
        "elapsed_s": float(time.time() - started),
        "layer_indices": layer_indices,
        "direction_norm_mean": float(direction_norms.mean().item()),
        "direction_norm_max": float(direction_norms.max().item()),
        "probe_available": bool(probe_available),
        "harmful_probe_available": bool(harmful_probe_available),
        "completion_exports": completion_exports,
        "rows": rows,
        "best_by_policy": best_by_policy,
    }
    _write_json(run_root / "summary.json", final_summary)
    (run_root / "run_note.md").write_text(
        "\n".join(
            [
                "# Simple Steering Baselines",
                "",
                f"- command: `{command}`",
                f"- timestamp: `{final_summary['timestamp']}`",
                f"- git_commit: `{final_summary['git_commit']}`",
                f"- status: `{final_summary['status']}`",
                f"- elapsed_s: `{final_summary['elapsed_s']:.2f}`",
                f"- model_id: `{args.model_id}`",
                f"- layers: `{layer_indices}`",
                f"- policies: `{', '.join(best_by_policy.keys())}`",
                f"- skip_preservation: `{bool(args.skip_preservation)}`",
                f"- export_completions: `{bool(args.export_completions)}`",
                f"- completion_exports: `{len(completion_exports)}`",
            ]
        )
        + "\n"
    )
    print(f"[simple_baselines] completed: {run_root}")


if __name__ == "__main__":
    main()
