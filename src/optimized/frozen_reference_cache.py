from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/trainer/ and src/optimized/ are on sys.path so that bare-name
# imports work regardless of the working directory.
_OPTIMIZED_DIR = Path(__file__).resolve().parent
_TRAINER_DIR = _OPTIMIZED_DIR.parent / "trainer"
for _d in [str(_OPTIMIZED_DIR), str(_TRAINER_DIR)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from gemma3_residual_paving_hybrid import ProjectedTrajectoryScorer, batched
from optimized_binary_runtime import trace_forced_trajectory_batch_optimized


_ACTIVE_FROZEN_REFERENCE_CACHE: Optional["FrozenReferenceCache"] = None


def _pooled_output_features(states: torch.Tensor) -> torch.Tensor:
    if states.ndim != 4:
        raise ValueError(f"Expected [B, T, L, H] states, got {tuple(states.shape)}")
    return states.float().mean(dim=(1, 2))


def _safe_feature_float(tensor: torch.Tensor, *, clamp: Optional[float] = None) -> torch.Tensor:
    out = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if clamp is not None:
        limit = float(max(clamp, 0.0))
        if limit > 0.0:
            out = out.clamp(min=-limit, max=limit)
    return out


def _safe_unit_features(tensor: torch.Tensor, *, eps: float, clamp: float) -> torch.Tensor:
    return F.normalize(_safe_feature_float(tensor, clamp=clamp), dim=-1, eps=float(eps))


def _resolve_cache_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported frozen reference cache dtype: {name!r}")


def _cat_or_none(chunks: List[torch.Tensor]) -> Optional[torch.Tensor]:
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def _tensor_nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


@dataclass
class CachedTeacherTopK:
    ids: torch.Tensor
    probs: torch.Tensor
    rest: torch.Tensor

    def nbytes(self) -> int:
        return _tensor_nbytes(self.ids) + _tensor_nbytes(self.probs) + _tensor_nbytes(self.rest)


@dataclass
class CachedTraceFeatures:
    prob: torch.Tensor
    logit: torch.Tensor
    state: torch.Tensor
    pooled: torch.Tensor

    def nbytes(self) -> int:
        return (
            _tensor_nbytes(self.prob)
            + _tensor_nbytes(self.logit)
            + _tensor_nbytes(self.state)
            + _tensor_nbytes(self.pooled)
        )


@dataclass
class FrozenFrenchReferenceCache:
    sample_index: Dict[str, int]
    pos_teacher: CachedTeacherTopK
    base: CachedTraceFeatures
    pos: CachedTraceFeatures
    neg: CachedTraceFeatures
    base_pos_dist: torch.Tensor
    hard_neg: Optional[CachedTraceFeatures] = None

    def nbytes(self) -> int:
        total = (
            self.pos_teacher.nbytes()
            + self.base.nbytes()
            + self.pos.nbytes()
            + self.neg.nbytes()
            + _tensor_nbytes(self.base_pos_dist)
        )
        if self.hard_neg is not None:
            total += self.hard_neg.nbytes()
        return total


@dataclass
class FrozenEnglishReferenceCache:
    sample_index: Dict[str, int]
    base_teacher: CachedTeacherTopK
    base: CachedTraceFeatures

    def nbytes(self) -> int:
        return self.base_teacher.nbytes() + self.base.nbytes()


@dataclass
class FrozenReferenceCache:
    mode: str
    topk: int
    student_topk: int
    cache_dtype_name: str
    fr: FrozenFrenchReferenceCache
    en: FrozenEnglishReferenceCache

    def nbytes(self) -> int:
        return self.fr.nbytes() + self.en.nbytes()


def set_active_frozen_reference_cache(cache: Optional[FrozenReferenceCache]) -> None:
    global _ACTIVE_FROZEN_REFERENCE_CACHE
    _ACTIVE_FROZEN_REFERENCE_CACHE = cache


def get_active_frozen_reference_cache() -> Optional[FrozenReferenceCache]:
    return _ACTIVE_FROZEN_REFERENCE_CACHE


def format_cache_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for suffix in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or suffix == "TB":
            return f"{value:.2f}{suffix}"
        value /= 1024.0
    return f"{num_bytes}B"


def _build_topk_teacher_cache(
    logits: torch.Tensor,
    *,
    topk: int,
    cache_dtype: torch.dtype,
) -> CachedTeacherTopK:
    vocab_size = int(logits.shape[-1])
    k = max(1, min(int(topk), vocab_size))
    logits_f = logits.float()
    values, ids = torch.topk(logits_f, k=k, dim=-1)
    log_z = torch.logsumexp(logits_f, dim=-1, keepdim=True)
    probs = torch.exp(values - log_z)
    rest = (1.0 - probs.sum(dim=-1)).clamp_min(0.0)
    return CachedTeacherTopK(
        ids=ids.cpu().to(torch.int32),
        probs=probs.cpu().to(cache_dtype),
        rest=rest.cpu().to(cache_dtype),
    )


def _compress_trace_features(
    scorer: ProjectedTrajectoryScorer,
    logits: torch.Tensor,
    states: torch.Tensor,
    *,
    cache_dtype: torch.dtype,
) -> CachedTraceFeatures:
    eps = float(scorer.loss_config.eps)
    state_cache_dtype = torch.float32 if cache_dtype == torch.float16 else cache_dtype
    pooled_cache_dtype = torch.float32 if cache_dtype == torch.float16 else cache_dtype
    return CachedTraceFeatures(
        prob=_safe_feature_float(scorer.embed_probs(logits), clamp=4.0).cpu().to(cache_dtype),
        logit=_safe_unit_features(scorer.embed_logit_dirs(logits), eps=eps, clamp=8.0).cpu().to(cache_dtype),
        state=_safe_unit_features(scorer.embed_states(states), eps=eps, clamp=1e4).cpu().to(state_cache_dtype),
        pooled=_safe_unit_features(_pooled_output_features(states), eps=eps, clamp=1e4).cpu().to(pooled_cache_dtype),
    )


def _gather_trace_features(
    cache: CachedTraceFeatures,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> CachedTraceFeatures:
    cpu_idx = indices.to(device="cpu", dtype=torch.long)
    return CachedTraceFeatures(
        prob=cache.prob.index_select(0, cpu_idx).to(device=device, dtype=torch.float32, non_blocking=True),
        logit=cache.logit.index_select(0, cpu_idx).to(device=device, dtype=torch.float32, non_blocking=True),
        state=cache.state.index_select(0, cpu_idx).to(device=device, dtype=torch.float32, non_blocking=True),
        pooled=cache.pooled.index_select(0, cpu_idx).to(device=device, dtype=torch.float32, non_blocking=True),
    )


def _gather_teacher_topk(
    cache: CachedTeacherTopK,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> CachedTeacherTopK:
    cpu_idx = indices.to(device="cpu", dtype=torch.long)
    return CachedTeacherTopK(
        ids=cache.ids.index_select(0, cpu_idx).to(device=device, dtype=torch.long, non_blocking=True),
        probs=cache.probs.index_select(0, cpu_idx).to(device=device, dtype=torch.float32, non_blocking=True),
        rest=cache.rest.index_select(0, cpu_idx).to(device=device, dtype=torch.float32, non_blocking=True),
    )


def _gather_fr_indices(cache: FrozenFrenchReferenceCache, examples: Sequence) -> torch.Tensor:
    return torch.tensor([cache.sample_index[ex.sample_id] for ex in examples], dtype=torch.long)


def _gather_en_indices(cache: FrozenEnglishReferenceCache, examples: Sequence) -> torch.Tensor:
    return torch.tensor([cache.sample_index[ex.sample_id] for ex in examples], dtype=torch.long)


def gather_french_reference_batch(
    cache: FrozenFrenchReferenceCache,
    examples: Sequence,
    *,
    device: torch.device,
) -> Dict[str, object]:
    indices = _gather_fr_indices(cache, examples)
    batch = {
        "pos_teacher": _gather_teacher_topk(cache.pos_teacher, indices, device=device),
        "base": _gather_trace_features(cache.base, indices, device=device),
        "pos": _gather_trace_features(cache.pos, indices, device=device),
        "neg": _gather_trace_features(cache.neg, indices, device=device),
        "base_pos_dist": cache.base_pos_dist.index_select(0, indices.to(device="cpu")).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
    }
    if cache.hard_neg is not None:
        batch["hard_neg"] = _gather_trace_features(cache.hard_neg, indices, device=device)
    else:
        batch["hard_neg"] = None
    return batch


def gather_english_reference_batch(
    cache: FrozenEnglishReferenceCache,
    examples: Sequence,
    *,
    device: torch.device,
) -> Dict[str, object]:
    indices = _gather_en_indices(cache, examples)
    return {
        "base_teacher": _gather_teacher_topk(cache.base_teacher, indices, device=device),
        "base": _gather_trace_features(cache.base, indices, device=device),
    }


def embed_live_trace_features(
    scorer: ProjectedTrajectoryScorer,
    logits: torch.Tensor,
    states: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    eps = float(scorer.loss_config.eps)
    return {
        "prob": _safe_feature_float(scorer.embed_probs(logits), clamp=4.0),
        "logit": _safe_unit_features(scorer.embed_logit_dirs(logits), eps=eps, clamp=8.0),
        "state": _safe_unit_features(scorer.embed_states(states), eps=eps, clamp=1e4),
        "pooled": _safe_unit_features(_pooled_output_features(states), eps=eps, clamp=1e4),
    }


def compressed_pairwise_paving_distance(
    *,
    scorer: ProjectedTrajectoryScorer,
    train_features: Dict[str, torch.Tensor],
    ref_features: CachedTraceFeatures,
    eps: float,
) -> torch.Tensor:
    train_prob = _safe_feature_float(train_features["prob"], clamp=4.0)
    ref_prob = _safe_feature_float(ref_features.prob, clamp=4.0)
    train_logit = _safe_unit_features(train_features["logit"], eps=eps, clamp=8.0)
    ref_logit = _safe_unit_features(ref_features.logit, eps=eps, clamp=8.0)
    train_state = _safe_unit_features(train_features["state"], eps=eps, clamp=1e4)
    ref_state = _safe_unit_features(ref_features.state, eps=eps, clamp=1e4)
    d_prob = torch.linalg.vector_norm(train_prob - ref_prob, ord=2, dim=-1) / math.sqrt(2.0)
    d_logit = 1.0 - F.cosine_similarity(
        train_logit,
        ref_logit,
        dim=-1,
        eps=float(eps),
    )
    d_state = 1.0 - F.cosine_similarity(
        train_state,
        ref_state,
        dim=-1,
        eps=float(eps),
    )
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
    return torch.nan_to_num(total / norm, nan=0.0, posinf=1e4, neginf=0.0)


def approximate_topk_union_kl(
    student_logits: torch.Tensor,
    teacher: CachedTeacherTopK,
    *,
    student_topk: int,
    extra_ids: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    logits_f = student_logits.float()
    batch_size, steps, vocab_size = logits_f.shape
    k = max(0, min(int(student_topk), vocab_size))
    flat_logits = logits_f.reshape(batch_size * steps, vocab_size)
    flat_log_z = torch.logsumexp(flat_logits, dim=-1, keepdim=True)

    flat_teacher_ids = teacher.ids.reshape(batch_size * steps, -1).to(device=logits_f.device, dtype=torch.long)
    flat_teacher_probs = teacher.probs.reshape(batch_size * steps, -1).float().to(logits_f.device)
    teacher_log_probs = flat_logits.gather(1, flat_teacher_ids) - flat_log_z
    teacher_term = (
        flat_teacher_probs
        * (torch.log(flat_teacher_probs.clamp_min(float(eps))) - teacher_log_probs)
    ).sum(dim=-1)

    union_parts = [flat_teacher_ids]
    if k > 0:
        union_parts.append(torch.topk(flat_logits, k=k, dim=-1).indices)
    if extra_ids is not None:
        if extra_ids.ndim == 2:
            union_parts.append(extra_ids.reshape(batch_size * steps, 1).to(device=logits_f.device, dtype=torch.long))
        elif extra_ids.ndim == 3:
            union_parts.append(extra_ids.reshape(batch_size * steps, extra_ids.shape[-1]).to(device=logits_f.device, dtype=torch.long))
        else:
            raise ValueError(f"Unsupported extra_ids rank: {extra_ids.ndim}")

    union_ids = torch.cat(union_parts, dim=-1)
    union_ids, _ = torch.sort(union_ids, dim=-1)
    unique_mask = torch.ones_like(union_ids, dtype=torch.bool)
    unique_mask[:, 1:] = union_ids[:, 1:] != union_ids[:, :-1]
    union_log_probs = flat_logits.gather(1, union_ids) - flat_log_z
    student_mass = (union_log_probs.exp() * unique_mask.to(union_log_probs.dtype)).sum(dim=-1).clamp(max=1.0)

    flat_teacher_rest = teacher.rest.reshape(batch_size * steps).float().to(logits_f.device)
    rest_mask = flat_teacher_rest > float(eps)
    rest_term = flat_teacher_rest.new_zeros(flat_teacher_rest.shape)
    if rest_mask.any():
        student_rest = (1.0 - student_mass[rest_mask]).clamp_min(float(eps))
        teacher_rest = flat_teacher_rest[rest_mask].clamp_min(float(eps))
        rest_term[rest_mask] = teacher_rest * (torch.log(teacher_rest) - torch.log(student_rest))

    losses = torch.clamp(teacher_term + rest_term, min=0.0)
    return losses.reshape(batch_size, steps)


def compressed_semantic_anchor_terms(
    *,
    train_features: Dict[str, torch.Tensor],
    ref_pooled_features: torch.Tensor,
    centroid_classification_loss_fn,
    examples: Sequence,
    bank: Dict[str, Optional[Dict[str, object]]],
    temperature: float,
    eps: float,
) -> Dict[str, torch.Tensor]:
    train_pooled = _safe_unit_features(train_features["pooled"], eps=eps, clamp=1e4)
    ref_pooled = _safe_unit_features(ref_pooled_features, eps=eps, clamp=1e4)
    content = 1.0 - F.cosine_similarity(train_pooled, ref_pooled.detach(), dim=-1, eps=float(eps))
    answer_mode_targets = [str(getattr(ex, "answer_mode", getattr(ex, "bucket", "general"))) for ex in examples]
    length_targets = [str(getattr(ex, "length_bucket", "medium")) for ex in examples]
    refusal_targets = ["refusal" if float(getattr(ex, "refusal_target", 0.0)) > 0.5 else "non_refusal" for ex in examples]
    return {
        "content": content.mean(),
        "mode": centroid_classification_loss_fn(train_pooled, answer_mode_targets, bank.get("answer_mode"), temperature=temperature, eps=eps),
        "length": centroid_classification_loss_fn(train_pooled, length_targets, bank.get("length_bucket"), temperature=temperature, eps=eps),
        "refusal": centroid_classification_loss_fn(train_pooled, refusal_targets, bank.get("refusal"), temperature=temperature, eps=eps),
    }


@torch.no_grad()
def build_frozen_reference_cache(
    *,
    model,
    paver,
    scorer: ProjectedTrajectoryScorer,
    fr_examples: Sequence,
    en_examples: Sequence,
    topk: int,
    student_topk: int,
    cache_dtype_name: str,
    chunk_size: int,
) -> FrozenReferenceCache:
    cache_dtype = _resolve_cache_dtype(cache_dtype_name)
    fr_items = list(fr_examples)
    en_items = list(en_examples)

    fr_base_prob: List[torch.Tensor] = []
    fr_base_logit: List[torch.Tensor] = []
    fr_base_state: List[torch.Tensor] = []
    fr_base_pooled: List[torch.Tensor] = []
    fr_pos_prob: List[torch.Tensor] = []
    fr_pos_logit: List[torch.Tensor] = []
    fr_pos_state: List[torch.Tensor] = []
    fr_pos_pooled: List[torch.Tensor] = []
    fr_neg_prob: List[torch.Tensor] = []
    fr_neg_logit: List[torch.Tensor] = []
    fr_neg_state: List[torch.Tensor] = []
    fr_neg_pooled: List[torch.Tensor] = []
    fr_hard_prob: List[torch.Tensor] = []
    fr_hard_logit: List[torch.Tensor] = []
    fr_hard_state: List[torch.Tensor] = []
    fr_hard_pooled: List[torch.Tensor] = []
    fr_pos_topk_ids: List[torch.Tensor] = []
    fr_pos_topk_probs: List[torch.Tensor] = []
    fr_pos_topk_rest: List[torch.Tensor] = []
    fr_base_pos_dist: List[torch.Tensor] = []

    fr_has_hard_neg = bool(fr_items) and all(ex.hard_negative_ids is not None for ex in fr_items)

    for chunk in batched(fr_items, max(1, int(chunk_size))):
        base_logits, base_states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in chunk],
            anchor_ids_batch=[ex.anchor_ids for ex in chunk],
            controller_vectors=None,
            gate_strengths=0.0,
        )
        pos_logits, pos_states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.teacher_ids for ex in chunk],
            anchor_ids_batch=[ex.anchor_ids for ex in chunk],
            controller_vectors=None,
            gate_strengths=0.0,
        )
        neg_logits, neg_states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in chunk],
            anchor_ids_batch=[ex.negative_ids for ex in chunk],
            controller_vectors=None,
            gate_strengths=0.0,
        )
        hard_logits = hard_states = None
        if fr_has_hard_neg:
            hard_logits, hard_states = trace_forced_trajectory_batch_optimized(
                model=model,
                paver=paver,
                base_ids_batch=[ex.base_ids for ex in chunk],
                anchor_ids_batch=[ex.hard_negative_ids for ex in chunk],
                controller_vectors=None,
                gate_strengths=0.0,
            )

        base_cache = _compress_trace_features(scorer, base_logits, base_states, cache_dtype=cache_dtype)
        pos_cache = _compress_trace_features(scorer, pos_logits, pos_states, cache_dtype=cache_dtype)
        neg_cache = _compress_trace_features(scorer, neg_logits, neg_states, cache_dtype=cache_dtype)
        hard_cache = (
            _compress_trace_features(scorer, hard_logits, hard_states, cache_dtype=cache_dtype)
            if hard_logits is not None and hard_states is not None
            else None
        )
        topk_cache = _build_topk_teacher_cache(pos_logits, topk=topk, cache_dtype=cache_dtype)
        base_pos = compressed_pairwise_paving_distance(
            scorer=scorer,
            train_features={
                "prob": base_cache.prob.to(device=base_logits.device, dtype=torch.float32),
                "logit": base_cache.logit.to(device=base_logits.device, dtype=torch.float32),
                "state": base_cache.state.to(device=base_logits.device, dtype=torch.float32),
                "pooled": base_cache.pooled.to(device=base_logits.device, dtype=torch.float32),
            },
            ref_features=CachedTraceFeatures(
                prob=pos_cache.prob.to(device=pos_logits.device, dtype=torch.float32),
                logit=pos_cache.logit.to(device=pos_logits.device, dtype=torch.float32),
                state=pos_cache.state.to(device=pos_logits.device, dtype=torch.float32),
                pooled=pos_cache.pooled.to(device=pos_logits.device, dtype=torch.float32),
            ),
            eps=float(scorer.loss_config.eps),
        )
        fr_base_prob.append(base_cache.prob)
        fr_base_logit.append(base_cache.logit)
        fr_base_state.append(base_cache.state)
        fr_base_pooled.append(base_cache.pooled)
        fr_pos_prob.append(pos_cache.prob)
        fr_pos_logit.append(pos_cache.logit)
        fr_pos_state.append(pos_cache.state)
        fr_pos_pooled.append(pos_cache.pooled)
        fr_neg_prob.append(neg_cache.prob)
        fr_neg_logit.append(neg_cache.logit)
        fr_neg_state.append(neg_cache.state)
        fr_neg_pooled.append(neg_cache.pooled)
        if hard_cache is not None:
            fr_hard_prob.append(hard_cache.prob)
            fr_hard_logit.append(hard_cache.logit)
            fr_hard_state.append(hard_cache.state)
            fr_hard_pooled.append(hard_cache.pooled)
        fr_pos_topk_ids.append(topk_cache.ids)
        fr_pos_topk_probs.append(topk_cache.probs)
        fr_pos_topk_rest.append(topk_cache.rest)
        fr_base_pos_dist.append(base_pos.cpu().to(cache_dtype))

    fr_cache = FrozenFrenchReferenceCache(
        sample_index={ex.sample_id: idx for idx, ex in enumerate(fr_items)},
        pos_teacher=CachedTeacherTopK(
            ids=torch.cat(fr_pos_topk_ids, dim=0),
            probs=torch.cat(fr_pos_topk_probs, dim=0),
            rest=torch.cat(fr_pos_topk_rest, dim=0),
        ),
        base=CachedTraceFeatures(
            prob=torch.cat(fr_base_prob, dim=0),
            logit=torch.cat(fr_base_logit, dim=0),
            state=torch.cat(fr_base_state, dim=0),
            pooled=torch.cat(fr_base_pooled, dim=0),
        ),
        pos=CachedTraceFeatures(
            prob=torch.cat(fr_pos_prob, dim=0),
            logit=torch.cat(fr_pos_logit, dim=0),
            state=torch.cat(fr_pos_state, dim=0),
            pooled=torch.cat(fr_pos_pooled, dim=0),
        ),
        neg=CachedTraceFeatures(
            prob=torch.cat(fr_neg_prob, dim=0),
            logit=torch.cat(fr_neg_logit, dim=0),
            state=torch.cat(fr_neg_state, dim=0),
            pooled=torch.cat(fr_neg_pooled, dim=0),
        ),
        hard_neg=(
            CachedTraceFeatures(
                prob=torch.cat(fr_hard_prob, dim=0),
                logit=torch.cat(fr_hard_logit, dim=0),
                state=torch.cat(fr_hard_state, dim=0),
                pooled=torch.cat(fr_hard_pooled, dim=0),
            )
            if fr_hard_prob
            else None
        ),
        base_pos_dist=torch.cat(fr_base_pos_dist, dim=0),
    )

    en_base_prob: List[torch.Tensor] = []
    en_base_logit: List[torch.Tensor] = []
    en_base_state: List[torch.Tensor] = []
    en_base_pooled: List[torch.Tensor] = []
    en_topk_ids: List[torch.Tensor] = []
    en_topk_probs: List[torch.Tensor] = []
    en_topk_rest: List[torch.Tensor] = []

    for chunk in batched(en_items, max(1, int(chunk_size))):
        base_logits, base_states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in chunk],
            anchor_ids_batch=[ex.anchor_ids for ex in chunk],
            controller_vectors=None,
            gate_strengths=0.0,
        )
        base_cache = _compress_trace_features(scorer, base_logits, base_states, cache_dtype=cache_dtype)
        topk_cache = _build_topk_teacher_cache(base_logits, topk=topk, cache_dtype=cache_dtype)
        en_base_prob.append(base_cache.prob)
        en_base_logit.append(base_cache.logit)
        en_base_state.append(base_cache.state)
        en_base_pooled.append(base_cache.pooled)
        en_topk_ids.append(topk_cache.ids)
        en_topk_probs.append(topk_cache.probs)
        en_topk_rest.append(topk_cache.rest)

    en_cache = FrozenEnglishReferenceCache(
        sample_index={ex.sample_id: idx for idx, ex in enumerate(en_items)},
        base_teacher=CachedTeacherTopK(
            ids=torch.cat(en_topk_ids, dim=0),
            probs=torch.cat(en_topk_probs, dim=0),
            rest=torch.cat(en_topk_rest, dim=0),
        ),
        base=CachedTraceFeatures(
            prob=torch.cat(en_base_prob, dim=0),
            logit=torch.cat(en_base_logit, dim=0),
            state=torch.cat(en_base_state, dim=0),
            pooled=torch.cat(en_base_pooled, dim=0),
        ),
    )

    return FrozenReferenceCache(
        mode="compressed_cache",
        topk=int(topk),
        student_topk=int(student_topk),
        cache_dtype_name=str(cache_dtype_name),
        fr=fr_cache,
        en=en_cache,
    )
