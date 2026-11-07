"""Standalone attention-map change loss utilities.

This module packages the reusable loss from ``experiment_attention.ipynb`` into
small PyTorch helpers that are easy to reuse in training loops and analysis
scripts.

Core concepts
-------------
- A single-layer attention map has shape ``[num_heads, sequence_length]``.
- A whole-model attention probe stacks layers into
  ``[num_layers, num_heads, sequence_length]``.
- The loss compares a current attention probe against a reference probe with
  mean squared error (MSE).
- Reports can include per-layer losses and the most changed layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
LayerWeights = Optional[Union[Sequence[float], Tensor]]

DEFAULT_ATTENTION_EPS = 1e-12
DEFAULT_TOP_CHANGED_LAYER_COUNT = 5


@dataclass(frozen=True)
class LossConfig:
    eps: float = DEFAULT_ATTENTION_EPS
    detach_reference: bool = True
    normalize_maps: bool = False


def token_labels_from_ids(tokenizer: Any, token_ids: Tensor) -> List[str]:
    labels: List[str] = []
    for token_id in token_ids.tolist():
        label = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        label = label.replace("\n", "\\n")
        if not label:
            label = tokenizer.convert_ids_to_tokens(int(token_id))
        labels.append(label)
    return labels


def find_query_position(labels: Sequence[str], needle: Optional[str] = None) -> int:
    if not labels:
        raise ValueError("labels must not be empty.")

    if needle is not None:
        lowered_needle = needle.lower()
        for idx, label in enumerate(labels):
            if lowered_needle in label.lower():
                return idx

    return len(labels) - 1


def _coerce_attention_tensor(attention_map: Tensor, *, name: str) -> Tensor:
    if not isinstance(attention_map, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if attention_map.ndim not in {2, 3}:
        raise ValueError(f"{name} must have shape [heads, seq] or [layers, heads, seq].")
    if not attention_map.is_floating_point():
        attention_map = attention_map.float()
    return attention_map


def _prepare_attention_pair(
    reference_attention_map: Tensor,
    current_attention_map: Tensor,
    *,
    config: LossConfig,
) -> tuple[Tensor, Tensor]:
    reference = _coerce_attention_tensor(reference_attention_map, name="reference_attention_map")
    current = _coerce_attention_tensor(current_attention_map, name="current_attention_map")

    if reference.shape != current.shape:
        raise ValueError(
            "reference_attention_map and current_attention_map must have the same shape."
        )

    if config.detach_reference:
        reference = reference.detach()

    if config.normalize_maps:
        reference = normalize_attention_maps(reference, eps=config.eps)
        current = normalize_attention_maps(current, eps=config.eps)

    return reference, current


def _as_multi_layer_attention_maps(attention_map: Tensor) -> Tensor:
    if attention_map.ndim == 2:
        return attention_map.unsqueeze(0)
    return attention_map


def _resolve_query_index(query_index: int, query_length: int) -> int:
    resolved = query_index if query_index >= 0 else query_length + query_index
    if not 0 <= resolved < query_length:
        raise IndexError(
            f"query_index {query_index} resolved to {resolved}, outside [0, {query_length})."
        )
    return resolved


def normalize_attention_maps(attention_map: Tensor, *, eps: float = DEFAULT_ATTENTION_EPS) -> Tensor:
    attention_map = _coerce_attention_tensor(attention_map, name="attention_map")
    normalizer = attention_map.sum(dim=-1, keepdim=True).clamp_min(eps)
    return attention_map / normalizer


def extract_attention_maps_for_query(
    attentions: Sequence[Tensor],
    query_index: int,
    *,
    batch_index: int = 0,
    sequence_length: Optional[int] = None,
    detach: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Extract ``[layers, heads, sequence_length]`` attention maps for one query token."""
    if not attentions:
        raise ValueError("attentions must not be empty.")
    if sequence_length is not None and sequence_length <= 0:
        raise ValueError("sequence_length must be positive when provided.")

    attention_maps: List[Tensor] = []
    for layer_idx, layer_attention in enumerate(attentions):
        if not isinstance(layer_attention, torch.Tensor):
            raise TypeError(f"Layer {layer_idx} attention must be a torch.Tensor.")
        if layer_attention.ndim != 4:
            raise ValueError(
                f"Layer {layer_idx} attention must have shape [batch, heads, query, key]."
            )
        if not 0 <= batch_index < layer_attention.shape[0]:
            raise IndexError(
                f"batch_index {batch_index} is outside [0, {layer_attention.shape[0]})."
            )

        resolved_query_index = _resolve_query_index(query_index, layer_attention.shape[-2])
        if sequence_length is None:
            resolved_sequence_length = layer_attention.shape[-1]
        else:
            resolved_sequence_length = min(sequence_length, layer_attention.shape[-1])

        attention_map = layer_attention[
            batch_index, :, resolved_query_index, :resolved_sequence_length
        ]
        if detach:
            attention_map = attention_map.detach()
        if dtype is not None or device is not None:
            attention_map = attention_map.to(
                device=device if device is not None else attention_map.device,
                dtype=dtype if dtype is not None else attention_map.dtype,
            )
        attention_maps.append(attention_map.contiguous())

    return torch.stack(attention_maps, dim=0)


def extract_attention_map_for_layer(
    attentions: Sequence[Tensor],
    layer_idx: int,
    query_index: int,
    *,
    batch_index: int = 0,
    sequence_length: Optional[int] = None,
    detach: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Extract a single-layer attention map with shape ``[heads, sequence_length]``."""
    attention_maps = extract_attention_maps_for_query(
        attentions,
        query_index,
        batch_index=batch_index,
        sequence_length=sequence_length,
        detach=detach,
        dtype=dtype,
        device=device,
    )

    if not 0 <= layer_idx < attention_maps.shape[0]:
        raise IndexError(f"layer_idx {layer_idx} is outside [0, {attention_maps.shape[0]}).")
    return attention_maps[layer_idx]


def attention_map_change_loss(
    reference_attention_map: Tensor,
    current_attention_map: Tensor,
    *,
    config: LossConfig = LossConfig(),
) -> Tensor:
    """Compute the single-layer attention-map change loss."""
    reference, current = _prepare_attention_pair(
        reference_attention_map,
        current_attention_map,
        config=config,
    )
    if reference.ndim != 2:
        raise ValueError("attention_map_change_loss expects shape [heads, sequence_length].")
    return F.mse_loss(current, reference)


def layerwise_attention_map_change_loss(
    reference_attention_maps: Tensor,
    current_attention_maps: Tensor,
    *,
    config: LossConfig = LossConfig(),
) -> Tensor:
    """Compute one MSE loss per layer."""
    reference, current = _prepare_attention_pair(
        reference_attention_maps,
        current_attention_maps,
        config=config,
    )
    reference = _as_multi_layer_attention_maps(reference)
    current = _as_multi_layer_attention_maps(current)
    return (current - reference).pow(2).mean(dim=(1, 2))


def _canonical_layer_weights(
    layer_weights: LayerWeights,
    *,
    num_layers: int,
    reference: Tensor,
) -> Optional[Tensor]:
    if layer_weights is None:
        return None

    weights = torch.as_tensor(
        layer_weights,
        dtype=reference.dtype,
        device=reference.device,
    )
    if weights.ndim != 1 or weights.shape[0] != num_layers:
        raise ValueError(f"layer_weights must have shape [{num_layers}].")
    if torch.any(weights < 0):
        raise ValueError("layer_weights must be non-negative.")

    weight_sum = weights.sum()
    if float(weight_sum.abs().detach().cpu()) <= DEFAULT_ATTENTION_EPS:
        raise ValueError("layer_weights must sum to a non-zero value.")

    return weights / weight_sum


def multi_layer_attention_map_change_loss(
    reference_attention_maps: Tensor,
    current_attention_maps: Tensor,
    *,
    layer_weights: LayerWeights = None,
    config: LossConfig = LossConfig(),
) -> Tensor:
    """Compute a whole-model attention-map loss.

    Without ``layer_weights`` this matches the notebook behavior and computes a
    plain mean over every compared attention entry.
    """
    per_layer_losses = layerwise_attention_map_change_loss(
        reference_attention_maps,
        current_attention_maps,
        config=config,
    )
    weights = _canonical_layer_weights(
        layer_weights,
        num_layers=per_layer_losses.shape[0],
        reference=per_layer_losses,
    )
    if weights is None:
        return per_layer_losses.mean()
    return torch.sum(per_layer_losses * weights)


def top_changed_layers(
    per_layer_losses: Tensor,
    *,
    k: int = DEFAULT_TOP_CHANGED_LAYER_COUNT,
) -> List[Dict[str, float]]:
    if per_layer_losses.ndim != 1:
        raise ValueError("per_layer_losses must be a 1D tensor.")
    if k <= 0 or per_layer_losses.numel() == 0:
        return []

    topk = torch.topk(per_layer_losses.detach(), k=min(k, per_layer_losses.numel()))
    return [
        {
            "layer_idx": int(layer_idx),
            "loss": float(loss_value),
        }
        for loss_value, layer_idx in zip(topk.values.tolist(), topk.indices.tolist())
    ]


def attention_map_change_report(
    reference_attention_maps: Tensor,
    current_attention_maps: Tensor,
    *,
    layer_weights: LayerWeights = None,
    top_k_layers: int = DEFAULT_TOP_CHANGED_LAYER_COUNT,
    config: LossConfig = LossConfig(),
) -> Dict[str, Any]:
    """Compute a scalar attention loss plus a per-layer breakdown."""
    per_layer_losses = layerwise_attention_map_change_loss(
        reference_attention_maps,
        current_attention_maps,
        config=config,
    )
    weights = _canonical_layer_weights(
        layer_weights,
        num_layers=per_layer_losses.shape[0],
        reference=per_layer_losses,
    )
    loss = per_layer_losses.mean() if weights is None else torch.sum(per_layer_losses * weights)

    return {
        "loss": loss,
        "loss_value": float(loss.detach().cpu()),
        "per_layer_losses": per_layer_losses,
        "per_layer_loss_values": [float(value) for value in per_layer_losses.detach().cpu().tolist()],
        "layer_weights": None if weights is None else [float(value) for value in weights.detach().cpu().tolist()],
        "top_changed_layers": top_changed_layers(per_layer_losses, k=top_k_layers),
    }


__all__ = [
    "DEFAULT_ATTENTION_EPS",
    "DEFAULT_TOP_CHANGED_LAYER_COUNT",
    "LossConfig",
    "attention_map_change_loss",
    "attention_map_change_report",
    "extract_attention_map_for_layer",
    "extract_attention_maps_for_query",
    "find_query_position",
    "layerwise_attention_map_change_loss",
    "multi_layer_attention_map_change_loss",
    "normalize_attention_maps",
    "token_labels_from_ids",
    "top_changed_layers",
]
