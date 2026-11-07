"""Standalone conditional edit loss utilities.

This module packages the strongest loss variant from ``experiment.ipynb`` into
small, pure-Python helpers that are easy to reuse in other projects.

Core concepts
-------------
- A step set is a list of records, one per generation step.
- Each record contains:
  - ``sample_id``: prompt/sample identifier
  - ``step``: generation step index
  - ``probs``: top-k token probabilities for that step
  - ``logits``: raw top-k logits for that step
  - ``condition``: the conditional label used for matching

The main objective is ``strong_conditional_prompt_match_with_logit_bonus``.
It compares:
- ``edit_train`` against ``keep``
- normalized by ``edit_base`` against ``keep``

The normalized loss is:
- ``0`` when ``edit_train`` matches ``keep``
- ``1`` when ``edit_train`` is no better than ``edit_base``
- ``> 1`` when ``edit_train`` is worse than ``edit_base``
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

Token = Union[int, str]
StateKey = Tuple[Any, int]

COND_LOGIT_OTHER_TOKEN = "__OTHER__"
COND_LOGIT_EPS = 1e-12
DEFAULT_LOGIT_BONUS_WEIGHT = 0.25


@dataclass(frozen=True)
class LossConfig:
    eps: float = COND_LOGIT_EPS
    other_token: Token = COND_LOGIT_OTHER_TOKEN
    logit_bonus_weight: float = DEFAULT_LOGIT_BONUS_WEIGHT
    logit_floor_margin: float = 1.0


def _sorted_tokens(tokens: Iterable[Token]) -> List[Token]:
    return sorted(tokens, key=lambda token: (str(type(token)), str(token)))


def load_trace_file(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def step_set_from_trace_dataset(
    dataset_trace: Mapping[str, Mapping[str, Any]],
    *,
    eps: float = COND_LOGIT_EPS,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Convert a traced dataset block into a reusable step set.

    Returns ``(step_set, used_log_probability_fallback)``.
    """
    step_set: List[Dict[str, Any]] = []
    missing_raw_logits = False

    for sample_name, sample in dataset_trace.items():
        for step in sample["steps"]:
            probs: Dict[Token, float] = {}
            logits: Dict[Token, float] = {}
            for token in step["top_tokens"]:
                token_id = int(token["token_id"])
                prob = float(token["probability"])
                probs[token_id] = prob
                if "logit" in token:
                    logits[token_id] = float(token["logit"])
                else:
                    missing_raw_logits = True
                    logits[token_id] = math.log(max(prob, eps))

            step_set.append(
                {
                    "sample_id": sample_name,
                    "step": int(step["step"]),
                    "probs": probs,
                    "logits": logits,
                }
            )

    return step_set, missing_raw_logits


def add_condition_labels(
    step_set: Sequence[Mapping[str, Any]],
    condition_labels: Sequence[Any],
) -> List[Dict[str, Any]]:
    if len(step_set) != len(condition_labels):
        raise ValueError("Condition labels must align with the step set.")

    conditioned: List[Dict[str, Any]] = []
    for item, condition in zip(step_set, condition_labels):
        updated = dict(item)
        updated["condition"] = condition
        conditioned.append(updated)
    return conditioned


def attach_constant_condition(
    step_set: Sequence[Mapping[str, Any]],
    *,
    condition_label: Any = "batch",
) -> List[Dict[str, Any]]:
    return add_condition_labels(step_set, [condition_label] * len(step_set))


def add_canonical_logits(
    step_set: Sequence[Mapping[str, Any]],
    *,
    eps: float = COND_LOGIT_EPS,
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for item in step_set:
        updated = dict(item)
        updated["logits"] = {
            token: math.log(max(prob, eps))
            for token, prob in item["probs"].items()
        }
        enriched.append(updated)
    return enriched


def split_step_set_by_sample_id(
    step_set: Sequence[Mapping[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    grouped: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in step_set:
        grouped[str(item["sample_id"])].append(dict(item))
    return [
        (sample_id, sorted(items, key=lambda item: int(item["step"])))
        for sample_id, items in sorted(grouped.items())
    ]


def build_sample_batches(
    sample_items: Sequence[Tuple[str, List[Dict[str, Any]]]],
    batch_size: int,
) -> List[List[Tuple[str, List[Dict[str, Any]]]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [list(sample_items[i : i + batch_size]) for i in range(0, len(sample_items), batch_size)]


def flatten_batch(
    batch_items: Sequence[Tuple[str, Sequence[Mapping[str, Any]]]],
) -> List[Dict[str, Any]]:
    flat_step_set: List[Dict[str, Any]] = []
    for _, sample_steps in batch_items:
        flat_step_set.extend(dict(step) for step in sample_steps)
    return flat_step_set


def conditional_state_key(item: Mapping[str, Any]) -> StateKey:
    return (item["condition"], int(item["step"]))


def _build_probability_support_by_state(
    *step_sets: Sequence[Mapping[str, Any]],
    other_token: Token,
) -> Dict[StateKey, List[Token]]:
    support: MutableMapping[StateKey, set] = defaultdict(set)
    for step_set in step_sets:
        for item in step_set:
            support[conditional_state_key(item)].update(item["probs"].keys())
    return {
        state: _sorted_tokens(tokens) + [other_token]
        for state, tokens in support.items()
    }


def _build_logit_support_by_state(
    *step_sets: Sequence[Mapping[str, Any]],
) -> Dict[StateKey, List[Token]]:
    support: MutableMapping[StateKey, set] = defaultdict(set)
    for step_set in step_sets:
        for item in step_set:
            support[conditional_state_key(item)].update(item["logits"].keys())
    return {state: _sorted_tokens(tokens) for state, tokens in support.items()}


def probability_vector_for_state(
    item: Mapping[str, Any],
    support: Sequence[Token],
    *,
    other_token: Token = COND_LOGIT_OTHER_TOKEN,
) -> List[float]:
    probs = item["probs"]
    observed_mass = float(sum(probs.values()))
    tail_mass = max(0.0, 1.0 - observed_mass)
    vector = [probs.get(token, 0.0) if token != other_token else tail_mass for token in support]
    total_mass = sum(vector)
    if not math.isclose(total_mass, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        vector[-1] = max(0.0, vector[-1] + (1.0 - total_mass))
    return vector


def normalized_logit_direction_for_state(
    item: Mapping[str, Any],
    support: Sequence[Token],
    *,
    eps: float = COND_LOGIT_EPS,
    floor_margin: float = 1.0,
) -> List[float]:
    logits = item["logits"]
    floor_value = (min(logits.values()) - floor_margin) if logits else -floor_margin
    vector = [logits.get(token, floor_value) for token in support]
    mean_value = sum(vector) / len(vector)
    centered = [value - mean_value for value in vector]
    norm = math.sqrt(sum(value * value for value in centered))
    if norm <= eps:
        return [0.0 for _ in centered]
    return [value / norm for value in centered]


def _grouped_probability_features_by_state(
    step_set: Sequence[Mapping[str, Any]],
    support_by_state: Mapping[StateKey, Sequence[Token]],
    *,
    other_token: Token,
) -> Dict[StateKey, List[List[float]]]:
    grouped: MutableMapping[StateKey, List[List[float]]] = defaultdict(list)
    for item in step_set:
        state = conditional_state_key(item)
        vector = probability_vector_for_state(item, support_by_state[state], other_token=other_token)
        grouped[state].append([math.sqrt(max(value, 0.0)) for value in vector])
    return dict(grouped)


def _grouped_logit_direction_features_by_state(
    step_set: Sequence[Mapping[str, Any]],
    support_by_state: Mapping[StateKey, Sequence[Token]],
    *,
    eps: float,
    floor_margin: float,
) -> Dict[StateKey, List[List[float]]]:
    grouped: MutableMapping[StateKey, List[List[float]]] = defaultdict(list)
    for item in step_set:
        state = conditional_state_key(item)
        grouped[state].append(
            normalized_logit_direction_for_state(
                item,
                support_by_state[state],
                eps=eps,
                floor_margin=floor_margin,
            )
        )
    return dict(grouped)


def empirical_state_weights(
    step_set: Sequence[Mapping[str, Any]],
) -> Dict[StateKey, float]:
    counts = Counter(conditional_state_key(item) for item in step_set)
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {state: count / total for state, count in counts.items()}


def l2_distance(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector_a, vector_b)))


def mean_pairwise_l2(
    feature_matrix_a: Sequence[Sequence[float]],
    feature_matrix_b: Sequence[Sequence[float]],
) -> float:
    total = 0.0
    count = 0
    for row_a in feature_matrix_a:
        for row_b in feature_matrix_b:
            total += l2_distance(row_a, row_b)
            count += 1
    if count <= 0:
        return 0.0
    return total / count


def empirical_energy_distance(
    feature_matrix_a: Sequence[Sequence[float]],
    feature_matrix_b: Sequence[Sequence[float]],
) -> float:
    if not feature_matrix_a or not feature_matrix_b:
        return 0.0
    cross = mean_pairwise_l2(feature_matrix_a, feature_matrix_b)
    within_a = mean_pairwise_l2(feature_matrix_a, feature_matrix_a)
    within_b = mean_pairwise_l2(feature_matrix_b, feature_matrix_b)
    return max((2.0 * cross) - within_a - within_b, 0.0)


def conditional_policy_distance_with_logit_bonus(
    candidate_set: Sequence[Mapping[str, Any]],
    reference_set: Sequence[Mapping[str, Any]],
    *,
    config: LossConfig = LossConfig(),
) -> Dict[str, Any]:
    candidate_weights = empirical_state_weights(candidate_set)
    reference_weights = empirical_state_weights(reference_set)
    state_support = sorted(set(candidate_weights) | set(reference_weights), key=str)
    state_marginal_tv = 0.5 * sum(
        abs(candidate_weights.get(state, 0.0) - reference_weights.get(state, 0.0))
        for state in state_support
    )

    probability_support_by_state = _build_probability_support_by_state(
        candidate_set,
        reference_set,
        other_token=config.other_token,
    )
    grouped_candidate_probs = _grouped_probability_features_by_state(
        candidate_set,
        probability_support_by_state,
        other_token=config.other_token,
    )
    grouped_reference_probs = _grouped_probability_features_by_state(
        reference_set,
        probability_support_by_state,
        other_token=config.other_token,
    )
    common_probability_states = sorted(set(grouped_candidate_probs) & set(grouped_reference_probs), key=str)

    conditional_policy_component = 0.0
    policy_per_state: Dict[str, Dict[str, float]] = {}
    for state in common_probability_states:
        state_distance = empirical_energy_distance(
            grouped_candidate_probs[state],
            grouped_reference_probs[state],
        )
        weighted_distance = reference_weights.get(state, 0.0) * state_distance
        conditional_policy_component += weighted_distance
        policy_per_state[str(state)] = {
            "energy_distance": float(state_distance),
            "reference_weight": float(reference_weights.get(state, 0.0)),
            "weighted_energy_distance": float(weighted_distance),
        }

    logit_support_by_state = _build_logit_support_by_state(candidate_set, reference_set)
    grouped_candidate_logits = _grouped_logit_direction_features_by_state(
        candidate_set,
        logit_support_by_state,
        eps=config.eps,
        floor_margin=config.logit_floor_margin,
    )
    grouped_reference_logits = _grouped_logit_direction_features_by_state(
        reference_set,
        logit_support_by_state,
        eps=config.eps,
        floor_margin=config.logit_floor_margin,
    )
    common_logit_states = sorted(set(grouped_candidate_logits) & set(grouped_reference_logits), key=str)

    conditional_logit_bonus = 0.0
    logit_per_state: Dict[str, Dict[str, float]] = {}
    for state in common_logit_states:
        state_distance = empirical_energy_distance(
            grouped_candidate_logits[state],
            grouped_reference_logits[state],
        )
        weighted_distance = reference_weights.get(state, 0.0) * state_distance
        conditional_logit_bonus += weighted_distance
        logit_per_state[str(state)] = {
            "energy_distance": float(state_distance),
            "reference_weight": float(reference_weights.get(state, 0.0)),
            "weighted_energy_distance": float(weighted_distance),
        }

    policy_distance = state_marginal_tv + conditional_policy_component
    total_distance = policy_distance + (config.logit_bonus_weight * conditional_logit_bonus)

    return {
        "distance": float(total_distance),
        "state_marginal_tv": float(state_marginal_tv),
        "conditional_policy_component": float(conditional_policy_component),
        "conditional_logit_bonus": float(conditional_logit_bonus),
        "logit_bonus_weight": float(config.logit_bonus_weight),
        "policy_per_state": policy_per_state,
        "logit_per_state": logit_per_state,
    }


def strong_conditional_prompt_match_with_logit_bonus(
    keep_set: Sequence[Mapping[str, Any]],
    edit_base_set: Sequence[Mapping[str, Any]],
    edit_train_set: Sequence[Mapping[str, Any]],
    *,
    config: LossConfig = LossConfig(),
) -> Dict[str, Any]:
    train_vs_keep = conditional_policy_distance_with_logit_bonus(
        edit_train_set,
        keep_set,
        config=config,
    )
    base_vs_keep = conditional_policy_distance_with_logit_bonus(
        edit_base_set,
        keep_set,
        config=config,
    )

    if base_vs_keep["distance"] <= config.eps:
        normalized_loss = 0.0 if train_vs_keep["distance"] <= config.eps else float("inf")
    else:
        normalized_loss = train_vs_keep["distance"] / (base_vs_keep["distance"] + config.eps)

    return {
        "loss": float(normalized_loss),
        "train_vs_keep": train_vs_keep,
        "base_vs_keep": base_vs_keep,
    }


def compare_batched_step_sets(
    edit_step_set: Sequence[Mapping[str, Any]],
    keep_step_set: Sequence[Mapping[str, Any]],
    *,
    edit_label: str = "edit",
    keep_label: str = "keep",
    batch_size: int = 32,
    max_batches: Optional[int] = None,
    condition_label: Any = "batch",
    config: LossConfig = LossConfig(),
) -> List[Dict[str, Any]]:
    edit_samples = split_step_set_by_sample_id(edit_step_set)
    keep_samples = split_step_set_by_sample_id(keep_step_set)
    edit_batches = build_sample_batches(edit_samples, batch_size)
    keep_batches = build_sample_batches(keep_samples, batch_size)

    num_batches = min(len(edit_batches), len(keep_batches))
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)

    batch_comparisons: List[Dict[str, Any]] = []
    for batch_index in range(num_batches):
        edit_batch_items = edit_batches[batch_index]
        keep_batch_items = keep_batches[batch_index]
        conditioned_edit_batch = attach_constant_condition(
            flatten_batch(edit_batch_items),
            condition_label=condition_label,
        )
        conditioned_keep_batch = attach_constant_condition(
            flatten_batch(keep_batch_items),
            condition_label=condition_label,
        )
        report = conditional_policy_distance_with_logit_bonus(
            conditioned_edit_batch,
            conditioned_keep_batch,
            config=config,
        )
        batch_comparisons.append(
            {
                "batch_index": batch_index,
                "edit_label": edit_label,
                "keep_label": keep_label,
                "edit_batch_size": len(edit_batch_items),
                "keep_batch_size": len(keep_batch_items),
                "edit_sample_ids": [sample_id for sample_id, _ in edit_batch_items],
                "keep_sample_ids": [sample_id for sample_id, _ in keep_batch_items],
                "distance": report["distance"],
                "state_marginal_tv": report["state_marginal_tv"],
                "conditional_policy_component": report["conditional_policy_component"],
                "conditional_logit_bonus": report["conditional_logit_bonus"],
                "logit_bonus_weight": report["logit_bonus_weight"],
            }
        )

    return batch_comparisons


__all__ = [
    "COND_LOGIT_EPS",
    "COND_LOGIT_OTHER_TOKEN",
    "DEFAULT_LOGIT_BONUS_WEIGHT",
    "LossConfig",
    "add_canonical_logits",
    "add_condition_labels",
    "attach_constant_condition",
    "build_sample_batches",
    "compare_batched_step_sets",
    "conditional_policy_distance_with_logit_bonus",
    "empirical_energy_distance",
    "flatten_batch",
    "load_trace_file",
    "split_step_set_by_sample_id",
    "step_set_from_trace_dataset",
    "strong_conditional_prompt_match_with_logit_bonus",
]
