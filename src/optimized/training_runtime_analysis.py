from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch


_ACTIVE_ANALYZER: Optional["TrainingRuntimeAnalyzer"] = None


def _maybe_storage_ptr(param: torch.Tensor) -> int:
    try:
        return int(param.untyped_storage().data_ptr())
    except Exception:
        return int(param.data_ptr())


def estimate_unique_parameter_count(model) -> int:
    seen = set()
    total = 0
    for param in model.parameters():
        ptr = _maybe_storage_ptr(param)
        if ptr in seen:
            continue
        seen.add(ptr)
        total += int(param.numel())
    return total


def _safe_config_attr(cfg, name: str) -> Any:
    if cfg is None:
        return None
    if hasattr(cfg, name):
        return getattr(cfg, name)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, name):
        return getattr(text_cfg, name)
    return None


@dataclass
class SectionStats:
    name: str
    calls: int = 0
    wall_time_s: float = 0.0
    model_calls: int = 0
    grad_model_calls: int = 0
    dense_tokens: int = 0
    logical_tokens: int = 0
    est_forward_flops: float = 0.0
    est_total_flops: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        utilization = float(self.logical_tokens) / float(self.dense_tokens) if self.dense_tokens > 0 else None
        return {
            "name": self.name,
            "calls": int(self.calls),
            "wall_time_s": float(self.wall_time_s),
            "model_calls": int(self.model_calls),
            "grad_model_calls": int(self.grad_model_calls),
            "dense_tokens": int(self.dense_tokens),
            "logical_tokens": int(self.logical_tokens),
            "padding_efficiency": utilization,
            "est_forward_flops": float(self.est_forward_flops),
            "est_total_flops": float(self.est_total_flops),
        }


class TrainingRuntimeAnalyzer:
    def __init__(
        self,
        *,
        enabled: bool,
        sync_cuda: bool = True,
        forward_flops_multiplier: float = 2.0,
        train_total_flops_multiplier: float = 4.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.sync_cuda = bool(sync_cuda)
        self.forward_flops_multiplier = float(forward_flops_multiplier)
        self.train_total_flops_multiplier = float(train_total_flops_multiplier)
        self.section_stats: Dict[str, SectionStats] = {}
        self.scope_stack: List[str] = []
        self.model_unique_params: Optional[int] = None
        self.model_metadata: Dict[str, Any] = {}
        self.created_at = time.time()

    def attach_model(self, model) -> None:
        self.model_unique_params = estimate_unique_parameter_count(model)
        cfg = getattr(model, "config", None)
        self.model_metadata = {
            "model_type": getattr(cfg, "model_type", None),
            "name_or_path": getattr(cfg, "name_or_path", None),
            "hidden_size": _safe_config_attr(cfg, "hidden_size"),
            "intermediate_size": _safe_config_attr(cfg, "intermediate_size"),
            "num_hidden_layers": _safe_config_attr(cfg, "num_hidden_layers"),
            "num_attention_heads": _safe_config_attr(cfg, "num_attention_heads"),
            "num_key_value_heads": _safe_config_attr(cfg, "num_key_value_heads"),
            "head_dim": _safe_config_attr(cfg, "head_dim"),
            "vocab_size": _safe_config_attr(cfg, "vocab_size"),
            "torch_dtype": str(getattr(cfg, "torch_dtype", None)),
        }

    def _sync(self) -> None:
        if not self.enabled or not self.sync_cuda:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _resolve_name(self, name: str, *, absolute: bool) -> str:
        if absolute or not self.scope_stack:
            return str(name)
        return f"{self.scope_stack[-1]}.{name}"

    def _stats(self, name: str) -> SectionStats:
        if name not in self.section_stats:
            self.section_stats[name] = SectionStats(name=name)
        return self.section_stats[name]

    @contextmanager
    def scope(self, name: str, *, absolute: bool = False) -> Iterator[str]:
        if not self.enabled:
            yield str(name)
            return
        resolved = self._resolve_name(name, absolute=absolute)
        self.scope_stack.append(resolved)
        try:
            yield resolved
        finally:
            popped = self.scope_stack.pop()
            if popped != resolved:
                raise RuntimeError(f"Profiler scope stack mismatch: expected {resolved!r}, got {popped!r}")

    @contextmanager
    def section(self, name: str, *, absolute: bool = False) -> Iterator[str]:
        if not self.enabled:
            yield str(name)
            return
        resolved = self._resolve_name(name, absolute=absolute)
        self._sync()
        start = time.perf_counter()
        self.scope_stack.append(resolved)
        try:
            yield resolved
        finally:
            popped = self.scope_stack.pop()
            if popped != resolved:
                raise RuntimeError(f"Profiler section stack mismatch: expected {resolved!r}, got {popped!r}")
            self._sync()
            elapsed = time.perf_counter() - start
            stats = self._stats(resolved)
            stats.calls += 1
            stats.wall_time_s += float(elapsed)

    def record_model_work(
        self,
        *,
        dense_tokens: int,
        logical_tokens: Optional[int] = None,
        requires_grad: bool,
        model_calls: int = 1,
        section: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        target_name = section or (self.scope_stack[-1] if self.scope_stack else "unscoped.model")
        stats = self._stats(target_name)
        dense = max(int(dense_tokens), 0)
        logical = dense if logical_tokens is None else max(int(logical_tokens), 0)
        stats.model_calls += int(model_calls)
        stats.grad_model_calls += int(model_calls) if requires_grad else 0
        stats.dense_tokens += dense
        stats.logical_tokens += logical
        if self.model_unique_params is not None and dense > 0:
            forward_flops = self.forward_flops_multiplier * float(self.model_unique_params) * float(dense)
            total_multiplier = self.train_total_flops_multiplier if requires_grad else self.forward_flops_multiplier
            total_flops = total_multiplier * float(self.model_unique_params) * float(dense)
            stats.est_forward_flops += forward_flops
            stats.est_total_flops += total_flops

    def _group_name(self, name: str) -> str:
        parts = str(name).split(".")
        if len(parts) >= 2 and parts[0] in {"setup", "build", "stage1", "stage2", "stage3"}:
            return ".".join(parts[:2])
        return parts[0]

    def grouped_stats(self) -> Dict[str, SectionStats]:
        grouped: Dict[str, SectionStats] = {}
        for name, stats in self.section_stats.items():
            group_name = self._group_name(name)
            out = grouped.get(group_name)
            if out is None:
                out = SectionStats(name=group_name)
                grouped[group_name] = out
            out.calls += stats.calls
            out.wall_time_s += stats.wall_time_s
            out.model_calls += stats.model_calls
            out.grad_model_calls += stats.grad_model_calls
            out.dense_tokens += stats.dense_tokens
            out.logical_tokens += stats.logical_tokens
            out.est_forward_flops += stats.est_forward_flops
            out.est_total_flops += stats.est_total_flops
        return grouped

    def summary_payload(self, *, top_k: int = 20) -> Dict[str, Any]:
        sections = sorted(self.section_stats.values(), key=lambda item: item.wall_time_s, reverse=True)
        groups = sorted(self.grouped_stats().values(), key=lambda item: item.wall_time_s, reverse=True)
        total_wall = sum(item.wall_time_s for item in sections)
        total_dense_tokens = sum(item.dense_tokens for item in sections)
        total_est_flops = sum(item.est_total_flops for item in sections)
        return {
            "created_at_unix": float(self.created_at),
            "config": {
                "enabled": bool(self.enabled),
                "sync_cuda": bool(self.sync_cuda),
                "forward_flops_multiplier": float(self.forward_flops_multiplier),
                "train_total_flops_multiplier": float(self.train_total_flops_multiplier),
            },
            "model": {
                "unique_parameter_count": self.model_unique_params,
                **self.model_metadata,
            },
            "totals": {
                "wall_time_s": float(total_wall),
                "dense_tokens": int(total_dense_tokens),
                "est_total_flops": float(total_est_flops),
            },
            "groups": [item.to_dict() for item in groups],
            "sections": [item.to_dict() for item in sections[: max(int(top_k), 1)]],
            "sections_all": [item.to_dict() for item in sections],
        }

    def format_summary(self, *, top_k: int = 20) -> str:
        payload = self.summary_payload(top_k=top_k)
        model_params = payload["model"]["unique_parameter_count"]
        total_wall = float(payload["totals"]["wall_time_s"])
        total_dense = int(payload["totals"]["dense_tokens"])
        total_flops = float(payload["totals"]["est_total_flops"])
        lines = [
            "[runtime analysis] "
            f"model_unique_params={model_params if model_params is not None else 'unknown'} "
            f"tracked_wall={total_wall:.2f}s "
            f"dense_tokens={total_dense} "
            f"est_total_flops={total_flops / 1e12:.3f} TFLOPs",
            "[runtime analysis groups]",
        ]
        for group in payload["groups"]:
            wall = float(group["wall_time_s"])
            pct = (100.0 * wall / total_wall) if total_wall > 0.0 else 0.0
            dense = int(group["dense_tokens"])
            logical = int(group["logical_tokens"])
            util = float(logical) / float(dense) if dense > 0 else 0.0
            total_tflops = float(group["est_total_flops"]) / 1e12
            throughput = total_tflops / wall if wall > 0.0 else 0.0
            lines.append(
                "  "
                f"{group['name']}: wall={wall:.2f}s ({pct:.1f}%) "
                f"model_calls={int(group['model_calls'])} "
                f"dense_tok={dense} "
                f"pad_eff={util:.3f} "
                f"est_total={total_tflops:.3f} TFLOPs "
                f"est_tflops_per_s={throughput:.3f}"
            )
        lines.append(f"[runtime analysis top {max(int(top_k), 1)} sections]")
        for section in payload["sections"]:
            wall = float(section["wall_time_s"])
            pct = (100.0 * wall / total_wall) if total_wall > 0.0 else 0.0
            dense = int(section["dense_tokens"])
            logical = int(section["logical_tokens"])
            util = float(logical) / float(dense) if dense > 0 else 0.0
            total_tflops = float(section["est_total_flops"]) / 1e12
            throughput = total_tflops / wall if wall > 0.0 else 0.0
            lines.append(
                "  "
                f"{section['name']}: wall={wall:.2f}s ({pct:.1f}%) "
                f"calls={int(section['calls'])} "
                f"model_calls={int(section['model_calls'])} "
                f"grad_calls={int(section['grad_model_calls'])} "
                f"dense_tok={dense} "
                f"pad_eff={util:.3f} "
                f"est_total={total_tflops:.3f} TFLOPs "
                f"est_tflops_per_s={throughput:.3f}"
            )
        return "\n".join(lines)

    def write_json(self, path: str | Path, *, top_k: int = 20, extra: Optional[Dict[str, Any]] = None) -> None:
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.summary_payload(top_k=top_k)
        if extra:
            payload["extra"] = extra
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def set_active_runtime_analyzer(analyzer: Optional[TrainingRuntimeAnalyzer]) -> None:
    global _ACTIVE_ANALYZER
    _ACTIVE_ANALYZER = analyzer


def get_active_runtime_analyzer() -> Optional[TrainingRuntimeAnalyzer]:
    return _ACTIVE_ANALYZER


@contextmanager
def profile_scope(name: str, *, absolute: bool = False) -> Iterator[str]:
    analyzer = get_active_runtime_analyzer()
    if analyzer is None:
        yield str(name)
        return
    with analyzer.scope(name, absolute=absolute) as resolved:
        yield resolved


@contextmanager
def profile_section(name: str, *, absolute: bool = False) -> Iterator[str]:
    analyzer = get_active_runtime_analyzer()
    if analyzer is None:
        yield str(name)
        return
    with analyzer.section(name, absolute=absolute) as resolved:
        yield resolved


def record_model_work(
    *,
    dense_tokens: int,
    logical_tokens: Optional[int] = None,
    requires_grad: bool,
    model_calls: int = 1,
    section: Optional[str] = None,
) -> None:
    analyzer = get_active_runtime_analyzer()
    if analyzer is None:
        return
    analyzer.record_model_work(
        dense_tokens=dense_tokens,
        logical_tokens=logical_tokens,
        requires_grad=requires_grad,
        model_calls=model_calls,
        section=section,
    )


def attach_profile_model(model) -> None:
    analyzer = get_active_runtime_analyzer()
    if analyzer is None:
        return
    analyzer.attach_model(model)
