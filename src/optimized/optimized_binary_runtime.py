from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/trainer/ and src/optimized/ are on sys.path so that bare-name
# imports work regardless of the working directory.
_OPTIMIZED_DIR = Path(__file__).resolve().parent
_TRAINER_DIR = _OPTIMIZED_DIR.parent / "trainer"
for _d in [str(_OPTIMIZED_DIR), str(_TRAINER_DIR)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import gemma3_residual_paving_binary_refined as refined
from gemma3_residual_paving_hybrid import batched
from training_runtime_analysis import record_model_work


_TRACE_BATCH_SIZE = 0
_TRACE_BATCH_CAP = 8
_TRACE_CACHE_WARNING_EMITTED = False


def configure_optimized_runtime(args) -> None:
    global _TRACE_BATCH_SIZE
    global _TRACE_BATCH_CAP
    _TRACE_BATCH_SIZE = max(0, int(getattr(args, "trace_batch_size", 0)))
    _TRACE_BATCH_CAP = max(1, int(getattr(args, "trace_batch_cap", 8)))


def _choose_chunk_size(num_items: int) -> int:
    if num_items <= 0:
        return 1
    if _TRACE_BATCH_SIZE > 0:
        return max(1, min(int(_TRACE_BATCH_SIZE), int(num_items)))
    return max(1, min(int(_TRACE_BATCH_CAP), int(num_items)))


def _truncate_token_ids(token_ids: torch.Tensor, max_steps: Optional[int]) -> torch.Tensor:
    flat = token_ids.view(-1)
    if max_steps is None:
        return flat
    resolved = int(max_steps)
    if resolved <= 0:
        return flat
    return flat[:resolved]


def _resolve_pad_token_id(tokenizer) -> int:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        return int(eos_token_id)
    raise ValueError("Tokenizer must define pad_token_id or eos_token_id for optimized batched tracing.")


def _left_pad_batch(sequences: Sequence[torch.Tensor], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Expected at least one sequence to batch.")
    max_len = max(int(seq.numel()) for seq in sequences)
    device = sequences[0].device
    input_ids = torch.full((len(sequences), max_len), int(pad_token_id), device=device, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), device=device, dtype=torch.long)
    for idx, seq in enumerate(sequences):
        flat = seq.view(-1).to(device=device, dtype=torch.long)
        width = int(flat.numel())
        input_ids[idx, -width:] = flat
        attention_mask[idx, -width:] = 1
    return input_ids, attention_mask


def _model_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    use_cache: bool = False,
    past_key_values=None,
):
    # Gemma3ForConditionalGeneration requires token_type_ids when model.training=True
    # (it uses them to distinguish image vs text tokens).  All our inputs are text
    # so we pass an all-zeros tensor of the same shape.
    kwargs: dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": bool(use_cache),
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        kwargs["token_type_ids"] = torch.zeros_like(input_ids)
    return model(**kwargs)


@contextmanager
def _intervene_and_capture_batch(
    paver,
    *,
    step_vectors: Optional[torch.Tensor],
    batch_scale: Optional[torch.Tensor | float],
):
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    mode = str(getattr(paver, "intervention_positions", "last_token") or "last_token").strip().lower()
    if mode not in {"last_token", "all_positions"}:
        raise ValueError(f"Unsupported intervention_positions: {mode!r}")

    if step_vectors is not None and step_vectors.ndim == 2:
        step_vectors = step_vectors.unsqueeze(0)

    for slot, module in enumerate(paver.modules):
        vec = None if step_vectors is None else step_vectors[:, slot, :]

        def hook_fn(_module, _args, output, slot=slot, vec=vec):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            if vec is not None and batch_scale is not None:
                if isinstance(batch_scale, torch.Tensor):
                    gate = batch_scale.to(hidden.device, hidden.dtype).view(-1, 1)
                else:
                    gate = torch.full((hidden.shape[0], 1), float(batch_scale), device=hidden.device, dtype=hidden.dtype)
                delta = vec.to(hidden.device, hidden.dtype)
                if delta.shape[0] == 1 and hidden.shape[0] != 1:
                    delta = delta.expand(hidden.shape[0], -1)
                delta = delta / delta.norm(dim=-1, keepdim=True).clamp(min=1.0)
                if mode == "all_positions":
                    hidden = hidden + gate.unsqueeze(1) * delta.unsqueeze(1)
                else:
                    hidden[:, -1, :] = hidden[:, -1, :] + gate * delta
            captured[slot] = hidden[:, -1, :]
            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden

        handles.append(module.register_forward_hook(hook_fn))

    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def capture_boundary_states_batch_optimized(
    model,
    paver,
    prompt_ids_batch: Sequence[torch.Tensor],
    *,
    pad_token_id: int,
) -> torch.Tensor:
    if len(prompt_ids_batch) > 1 and _model_type(model) == "gemma3":
        # Gemma 3 can emit non-finite hidden states for padded batched prefix
        # captures. Keep upstream generation batched, but capture router
        # boundary states one prompt at a time.
        return torch.cat(
            [
                capture_boundary_states_batch_optimized(
                    model,
                    paver,
                    [prompt_ids],
                    pad_token_id=pad_token_id,
                )
                for prompt_ids in prompt_ids_batch
            ],
            dim=0,
        )
    input_ids, attention_mask = _left_pad_batch(prompt_ids_batch, pad_token_id)
    record_model_work(
        dense_tokens=int(input_ids.numel()),
        logical_tokens=int(attention_mask.sum().item()),
        requires_grad=False,
    )
    with _intervene_and_capture_batch(paver, step_vectors=None, batch_scale=None) as captured:
        _ = _model_forward(model, input_ids, attention_mask)
    states = [captured[i].float().clone() for i in range(len(paver.modules))]
    return torch.stack(states, dim=1)


def _normalize_controller_batch(
    controller_vectors: Optional[torch.Tensor],
    batch_size: int,
) -> Optional[torch.Tensor]:
    if controller_vectors is None:
        return None
    if controller_vectors.ndim == 3:
        return controller_vectors.unsqueeze(0).expand(batch_size, -1, -1, -1)
    if controller_vectors.ndim == 4:
        if controller_vectors.shape[0] != batch_size:
            raise ValueError(
                f"Expected controller batch size {batch_size}, got {controller_vectors.shape[0]}"
            )
        return controller_vectors
    raise ValueError(f"Unsupported controller shape: {tuple(controller_vectors.shape)}")


def _normalize_gate_strengths(
    gate_strengths: torch.Tensor | float,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(gate_strengths, torch.Tensor):
        out = gate_strengths.to(device=device, dtype=torch.float32).view(-1)
        if out.numel() == 1 and batch_size != 1:
            out = out.expand(batch_size)
        if out.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} gate strengths, got {out.numel()}")
        return out
    return torch.full((batch_size,), float(gate_strengths), device=device, dtype=torch.float32)


def _model_type(model) -> Optional[str]:
    marker = getattr(model, "_rp_original_model_type", None)
    if marker is not None:
        return str(marker)
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    if hasattr(cfg, "model_type") and cfg.model_type is not None:
        return str(cfg.model_type)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "model_type") and text_cfg.model_type is not None:
        return str(text_cfg.model_type)
    return None


def _should_use_incremental_kv_trace(model) -> bool:
    override = os.environ.get("RESIDUAL_PAVING_ENABLE_UNSAFE_KV_TRACE", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    # Gemma 3's incremental cache path currently does not reproduce the teacher-forced
    # full-prefix trace exactly, which makes the cached trace unsafe for training/eval.
    return _model_type(model) not in {"gemma3"}


def _maybe_warn_trace_cache_disabled(model) -> None:
    global _TRACE_CACHE_WARNING_EMITTED
    if _TRACE_CACHE_WARNING_EMITTED:
        return
    print(
        "[optimized runtime] incremental KV trace disabled for "
        f"{_model_type(model) or 'unknown model'}; "
        "set RESIDUAL_PAVING_ENABLE_UNSAFE_KV_TRACE=1 to override."
    )
    _TRACE_CACHE_WARNING_EMITTED = True


def _trace_forced_trajectory_batch_full_recompute(
    *,
    model,
    paver,
    flat_bases: Sequence[torch.Tensor],
    flat_anchors: Sequence[torch.Tensor],
    controller_batch: Optional[torch.Tensor],
    gate_batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_steps: List[torch.Tensor] = []
    states_steps: List[torch.Tensor] = []

    # The batched trace callers operate on already-tokenized ids and do not carry a tokenizer handle.
    # Using zero as the left-pad id is safe here because the attention mask fully suppresses padded positions.
    trace_pad_token_id = 0
    total_steps = int(flat_anchors[0].shape[0])

    for step in range(total_steps):
        curr_sequences: List[torch.Tensor] = []
        for base_ids, anchor_ids in zip(flat_bases, flat_anchors):
            if step > 0:
                curr_sequences.append(torch.cat([base_ids, anchor_ids[:step].to(base_ids.device)], dim=0))
            else:
                curr_sequences.append(base_ids)

        input_ids, attention_mask = _left_pad_batch(curr_sequences, trace_pad_token_id)
        record_model_work(
            dense_tokens=int(input_ids.numel()),
            logical_tokens=int(attention_mask.sum().item()),
            requires_grad=bool(torch.is_grad_enabled()),
        )
        step_vectors = None
        if controller_batch is not None and step < controller_batch.shape[1]:
            step_vectors = controller_batch[:, step, :, :]

        with _intervene_and_capture_batch(
            paver,
            step_vectors=step_vectors,
            batch_scale=gate_batch if step_vectors is not None else None,
        ) as captured:
            outputs = _model_forward(model, input_ids, attention_mask, use_cache=False)

        logits_steps.append(outputs.logits[:, -1, :].float())
        states_steps.append(torch.stack([captured[i].float() for i in range(len(paver.modules))], dim=1))

    return torch.stack(logits_steps, dim=1), torch.stack(states_steps, dim=1)


def trace_forced_trajectory_batch_optimized(
    *,
    model,
    paver,
    base_ids_batch: Sequence[torch.Tensor],
    anchor_ids_batch: Sequence[torch.Tensor],
    controller_vectors: Optional[torch.Tensor],
    gate_strengths: torch.Tensor | float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not base_ids_batch:
        raise ValueError("Expected at least one example for batched tracing.")

    batch_size = len(base_ids_batch)
    controller_batch = _normalize_controller_batch(controller_vectors, batch_size)
    gate_batch = _normalize_gate_strengths(
        gate_strengths,
        batch_size=batch_size,
        device=base_ids_batch[0].device,
    )

    flat_bases = [base_ids.view(-1) for base_ids in base_ids_batch]
    flat_anchors = [anchor_ids.view(-1) for anchor_ids in anchor_ids_batch]
    step_counts = {int(anchor.shape[0]) for anchor in flat_anchors}
    if len(step_counts) != 1:
        raise ValueError(f"Optimized batched tracing requires equal anchor lengths, got {sorted(step_counts)}")
    total_steps = step_counts.pop()
    logits_steps: List[torch.Tensor] = []
    states_steps: List[torch.Tensor] = []

    if not _should_use_incremental_kv_trace(model):
        _maybe_warn_trace_cache_disabled(model)
        return _trace_forced_trajectory_batch_full_recompute(
            model=model,
            paver=paver,
            flat_bases=flat_bases,
            flat_anchors=flat_anchors,
            controller_batch=controller_batch,
            gate_batch=gate_batch,
        )

    trace_pad_token_id = 0
    prompt_input_ids, prompt_attention_mask = _left_pad_batch(flat_bases, trace_pad_token_id)
    running_attention_mask = prompt_attention_mask
    past_key_values = None

    for step in range(total_steps):
        if step == 0:
            curr_input_ids = prompt_input_ids
            curr_attention_mask = running_attention_mask
            logical_tokens = int(prompt_attention_mask.sum().item())
        else:
            curr_input_ids = torch.stack(
                [anchor_ids[step - 1].to(base_ids.device) for base_ids, anchor_ids in zip(flat_bases, flat_anchors)],
                dim=0,
            ).view(batch_size, 1)
            next_mask = torch.ones((batch_size, 1), device=running_attention_mask.device, dtype=running_attention_mask.dtype)
            running_attention_mask = torch.cat([running_attention_mask, next_mask], dim=1)
            curr_attention_mask = running_attention_mask
            logical_tokens = int(curr_input_ids.shape[0])

        record_model_work(
            dense_tokens=int(curr_input_ids.numel()),
            logical_tokens=logical_tokens,
            requires_grad=bool(torch.is_grad_enabled()),
        )

        step_vectors = None
        if controller_batch is not None and step < controller_batch.shape[1]:
            step_vectors = controller_batch[:, step, :, :]

        with _intervene_and_capture_batch(
            paver,
            step_vectors=step_vectors,
            batch_scale=gate_batch if step_vectors is not None else None,
        ) as captured:
            outputs = _model_forward(
                model,
                curr_input_ids,
                curr_attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

        next_past_key_values = getattr(outputs, "past_key_values", None)
        if next_past_key_values is None:
            return _trace_forced_trajectory_batch_full_recompute(
                model=model,
                paver=paver,
                flat_bases=flat_bases,
                flat_anchors=flat_anchors,
                controller_batch=controller_batch,
                gate_batch=gate_batch,
            )

        past_key_values = next_past_key_values
        logits_steps.append(outputs.logits[:, -1, :].float())
        states_steps.append(torch.stack([captured[i].float() for i in range(len(paver.modules))], dim=1))

    return torch.stack(logits_steps, dim=1), torch.stack(states_steps, dim=1)


@torch.no_grad()
def generate_with_controller_batch_optimized(
    *,
    model,
    paver,
    base_ids_batch: Sequence[torch.Tensor],
    controller_vectors: Optional[torch.Tensor],
    gate_strengths: torch.Tensor | float,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> List[torch.Tensor]:
    if not base_ids_batch:
        return []

    batch_size = len(base_ids_batch)
    if batch_size > 1 and _model_type(model) == "gemma3":
        generated: List[torch.Tensor] = []
        for idx, base_ids in enumerate(base_ids_batch):
            controller_one = None
            if controller_vectors is not None:
                if controller_vectors.ndim == 3:
                    controller_one = controller_vectors
                elif controller_vectors.ndim == 4:
                    controller_one = controller_vectors[idx : idx + 1]
                else:
                    raise ValueError(f"Unsupported controller shape: {tuple(controller_vectors.shape)}")
            if isinstance(gate_strengths, torch.Tensor):
                gate_one = gate_strengths.view(-1)[idx : idx + 1]
            else:
                gate_one = gate_strengths
            generated.extend(
                generate_with_controller_batch_optimized(
                    model=model,
                    paver=paver,
                    base_ids_batch=[base_ids],
                    controller_vectors=controller_one,
                    gate_strengths=gate_one,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                )
            )
        return generated

    controller_batch = _normalize_controller_batch(controller_vectors, batch_size)
    gate_batch = _normalize_gate_strengths(
        gate_strengths,
        batch_size=batch_size,
        device=base_ids_batch[0].device,
    )
    prompt_input_ids, prompt_attention_mask = _left_pad_batch([base_ids.view(-1) for base_ids in base_ids_batch], 0)
    running_attention_mask = prompt_attention_mask
    past_key_values = None
    generated_steps: List[torch.Tensor] = []
    finished = torch.zeros((batch_size,), dtype=torch.bool, device=prompt_input_ids.device)

    for step in range(max(0, int(max_new_tokens))):
        if step == 0:
            curr_input_ids = prompt_input_ids
            curr_attention_mask = running_attention_mask
            logical_tokens = int(prompt_attention_mask.sum().item())
        else:
            curr_input_ids = generated_steps[-1]
            next_mask = torch.ones((batch_size, 1), device=running_attention_mask.device, dtype=running_attention_mask.dtype)
            running_attention_mask = torch.cat([running_attention_mask, next_mask], dim=1)
            curr_attention_mask = running_attention_mask
            logical_tokens = int(curr_input_ids.shape[0])

        record_model_work(
            dense_tokens=int(curr_input_ids.numel()),
            logical_tokens=logical_tokens,
            requires_grad=False,
        )

        step_vectors = None
        if controller_batch is not None and step < controller_batch.shape[1]:
            step_vectors = controller_batch[:, step, :, :]

        with _intervene_and_capture_batch(
            paver,
            step_vectors=step_vectors,
            batch_scale=gate_batch if step_vectors is not None else None,
        ):
            outputs = _model_forward(
                model,
                curr_input_ids,
                curr_attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

        next_past_key_values = getattr(outputs, "past_key_values", None)
        if next_past_key_values is None:
            raise RuntimeError("Autoregressive generation requires a model/cache path that returns past_key_values.")
        past_key_values = next_past_key_values

        next_tokens = outputs.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
        if eos_token_id is not None:
            next_tokens = torch.where(
                finished.view(-1, 1),
                torch.full_like(next_tokens, int(eos_token_id)),
                next_tokens,
            )
        generated_steps.append(next_tokens)
        if eos_token_id is not None:
            finished = finished | (next_tokens.view(-1) == int(eos_token_id))
            if bool(finished.all().item()):
                break

    if not generated_steps:
        return [torch.empty(0, dtype=torch.long, device=prompt_input_ids.device) for _ in range(batch_size)]
    generated = torch.cat(generated_steps, dim=1)
    return [generated[idx].detach().clone() for idx in range(batch_size)]


def _cat_tensors(chunks: List[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        raise ValueError("Expected at least one chunk to concatenate.")
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def build_prompt_variant_examples_optimized(
    model,
    tokenizer,
    paver,
    device: torch.device,
    examples: Sequence[refined.BinaryExample],
    *,
    max_variants: int = 1,
    variant_index: int = 0,
    max_prompt_tokens: int = 0,
) -> List[refined.BinaryExample]:
    if max_variants <= 0:
        return []

    payloads: List[Tuple[refined.BinaryExample, str, torch.Tensor, int]] = []
    for ex in examples:
        variants = refined.make_prompt_variants(
            ex.prompt,
            ex.bucket,
            max_variants=max(max_variants, variant_index + 1),
        )
        if not variants:
            continue
        chosen_index = min(int(variant_index), len(variants) - 1)
        prompt_variant = variants[chosen_index]
        base_ids = refined.encode_chat_prompt_with_cap(
            tokenizer,
            prompt_variant,
            None,
            device,
            max_prompt_tokens=max_prompt_tokens,
        )
        payloads.append((ex, prompt_variant, base_ids, chosen_index))

    if not payloads:
        return []

    pad_token_id = _resolve_pad_token_id(tokenizer)
    boundary_chunks: List[torch.Tensor] = []
    base_batches = [item[2] for item in payloads]
    for base_chunk in batched(base_batches, _choose_chunk_size(len(base_batches))):
        boundary_chunks.append(
            capture_boundary_states_batch_optimized(
                model,
                paver,
                base_chunk,
                pad_token_id=pad_token_id,
            )
        )
    all_boundaries = _cat_tensors(boundary_chunks)

    variant_examples: List[refined.BinaryExample] = []
    for (ex, prompt_variant, base_ids, chosen_index), boundary_states in zip(payloads, all_boundaries):
        variant_examples.append(
            refined.BinaryExample(
                sample_id=f"{ex.sample_id}_variant_{chosen_index}",
                kind=ex.kind,
                prompt=prompt_variant,
                paired_prompt=ex.paired_prompt,
                base_ids=base_ids,
                anchor_ids=ex.anchor_ids,
                boundary_states=boundary_states,
                gate_target=ex.gate_target,
                teacher_ids=ex.teacher_ids,
                negative_ids=ex.negative_ids,
                bucket=ex.bucket,
                hard_negative_ids=ex.hard_negative_ids,
                answer_mode=ex.answer_mode,
                refusal_target=ex.refusal_target,
                length_bucket=ex.length_bucket,
            )
        )
    return variant_examples


@torch.no_grad()
def collect_fr_references_optimized(
    model,
    paver,
    examples: Sequence[refined.BinaryExample],
    max_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_logits_chunks: List[torch.Tensor] = []
    base_states_chunks: List[torch.Tensor] = []
    pos_logits_chunks: List[torch.Tensor] = []
    pos_states_chunks: List[torch.Tensor] = []
    neg_logits_chunks: List[torch.Tensor] = []
    neg_states_chunks: List[torch.Tensor] = []

    for chunk in batched(list(examples), _choose_chunk_size(len(examples))):
        assert all(ex.teacher_ids is not None and ex.negative_ids is not None for ex in chunk)
        base_ids_batch = [ex.base_ids for ex in chunk]
        anchor_ids_batch = [_truncate_token_ids(ex.anchor_ids, max_steps) for ex in chunk]
        teacher_ids_batch = [ex.teacher_ids for ex in chunk]
        negative_ids_batch = [_truncate_token_ids(ex.negative_ids, max_steps) for ex in chunk]

        base_logits, base_states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=base_ids_batch,
            anchor_ids_batch=anchor_ids_batch,
            controller_vectors=None,
            gate_strengths=0.0,
        )
        pos_logits, pos_states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=teacher_ids_batch,
            anchor_ids_batch=anchor_ids_batch,
            controller_vectors=None,
            gate_strengths=0.0,
        )
        neg_logits, neg_states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=base_ids_batch,
            anchor_ids_batch=negative_ids_batch,
            controller_vectors=None,
            gate_strengths=0.0,
        )

        base_logits_chunks.append(base_logits)
        base_states_chunks.append(base_states)
        pos_logits_chunks.append(pos_logits)
        pos_states_chunks.append(pos_states)
        neg_logits_chunks.append(neg_logits)
        neg_states_chunks.append(neg_states)

    return (
        _cat_tensors(base_logits_chunks),
        _cat_tensors(base_states_chunks),
        _cat_tensors(pos_logits_chunks),
        _cat_tensors(pos_states_chunks),
        _cat_tensors(neg_logits_chunks),
        _cat_tensors(neg_states_chunks),
    )


@torch.no_grad()
def collect_en_references_optimized(
    model,
    paver,
    examples: Sequence[refined.BinaryExample],
    max_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_chunks: List[torch.Tensor] = []
    states_chunks: List[torch.Tensor] = []
    for chunk in batched(list(examples), _choose_chunk_size(len(examples))):
        logits, states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in chunk],
            anchor_ids_batch=[_truncate_token_ids(ex.anchor_ids, max_steps) for ex in chunk],
            controller_vectors=None,
            gate_strengths=0.0,
        )
        logits_chunks.append(logits)
        states_chunks.append(states)
    return _cat_tensors(logits_chunks), _cat_tensors(states_chunks)


@torch.no_grad()
def collect_hard_negative_references_optimized(
    model,
    paver,
    examples: Sequence[refined.BinaryExample],
    max_steps: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not examples or any(ex.hard_negative_ids is None for ex in examples):
        return None, None

    logits_chunks: List[torch.Tensor] = []
    states_chunks: List[torch.Tensor] = []
    for chunk in batched(list(examples), _choose_chunk_size(len(examples))):
        logits, states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in chunk],
            anchor_ids_batch=[_truncate_token_ids(ex.hard_negative_ids, max_steps) for ex in chunk],
            controller_vectors=None,
            gate_strengths=0.0,
        )
        logits_chunks.append(logits)
        states_chunks.append(states)
    return _cat_tensors(logits_chunks), _cat_tensors(states_chunks)


def gather_train_traces_with_controller_batch_optimized(
    model,
    paver,
    controller_batch: torch.Tensor,
    examples: Sequence[refined.BinaryExample],
    gate_strengths: torch.Tensor,
    max_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_chunks: List[torch.Tensor] = []
    states_chunks: List[torch.Tensor] = []
    chunk_size = _choose_chunk_size(len(examples))
    for start in range(0, len(examples), chunk_size):
        end = min(start + chunk_size, len(examples))
        ex_chunk = list(examples[start:end])
        logits, states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in ex_chunk],
            anchor_ids_batch=[_truncate_token_ids(ex.anchor_ids, max_steps) for ex in ex_chunk],
            controller_vectors=controller_batch[start:end],
            gate_strengths=gate_strengths[start:end],
        )
        logits_chunks.append(logits)
        states_chunks.append(states)
    return _cat_tensors(logits_chunks), _cat_tensors(states_chunks)


@torch.no_grad()
def compute_gate_utility_weights_optimized(
    model,
    paver,
    scorer,
    controller_vectors: torch.Tensor,
    fr_examples: Sequence[refined.BinaryExample],
    en_examples: Sequence[refined.BinaryExample],
    step_weights: torch.Tensor,
    sep_reward: float,
    utility_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    from frozen_reference_cache import (
        compressed_pairwise_paving_distance,
        embed_live_trace_features,
        gather_english_reference_batch,
        gather_french_reference_batch,
        get_active_frozen_reference_cache,
    )

    fr_scores: List[float] = []
    en_scores: List[float] = []
    frozen_cache = get_active_frozen_reference_cache()
    use_cached_refs = frozen_cache is not None and str(frozen_cache.mode) == "compressed_cache"

    def _weighted_rows(values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError(f"Expected [B, T] utility values, got {tuple(values.shape)}")
        k = min(values.shape[1], step_weights.shape[0])
        if k <= 0:
            return values.new_zeros((values.shape[0],), dtype=torch.float32)
        w = step_weights[:k].to(device=values.device, dtype=torch.float32).view(1, k)
        return (values[:, :k].float() * w).sum(dim=1) / w.sum().clamp_min(1e-6)

    fr_chunk_size = _choose_chunk_size(len(fr_examples))
    for chunk in batched(list(fr_examples), fr_chunk_size):
        assert all(ex.teacher_ids is not None and ex.negative_ids is not None for ex in chunk)
        batch_size = len(chunk)
        on_logits, on_states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in chunk],
            anchor_ids_batch=[ex.anchor_ids for ex in chunk],
            controller_vectors=controller_vectors,
            gate_strengths=torch.ones(batch_size, device=controller_vectors.device, dtype=torch.float32),
        )
        if use_cached_refs:
            on_features = embed_live_trace_features(scorer, on_logits, on_states)
            fr_cached = gather_french_reference_batch(frozen_cache.fr, chunk, device=controller_vectors.device)
            d_on_pos = compressed_pairwise_paving_distance(
                scorer=scorer,
                train_features=on_features,
                ref_features=fr_cached["pos"],
                eps=float(scorer.loss_config.eps),
            )
            d_off_pos = fr_cached["base_pos_dist"].float()
            d_on_neg = compressed_pairwise_paving_distance(
                scorer=scorer,
                train_features=on_features,
                ref_features=fr_cached["neg"],
                eps=float(scorer.loss_config.eps),
            )
            benefit = _weighted_rows(d_off_pos - d_on_pos) + float(sep_reward) * _weighted_rows(d_on_neg - d_on_pos)
            fr_scores.extend(torch.clamp(benefit, min=0.0).detach().cpu().tolist())
        else:
            off_logits, off_states = trace_forced_trajectory_batch_optimized(
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

            for idx in range(batch_size):
                d_on_pos = scorer.stepwise_distances(
                    on_logits[idx : idx + 1],
                    pos_logits[idx : idx + 1],
                    on_states[idx : idx + 1],
                    pos_states[idx : idx + 1],
                )
                d_off_pos = scorer.stepwise_distances(
                    off_logits[idx : idx + 1],
                    pos_logits[idx : idx + 1],
                    off_states[idx : idx + 1],
                    pos_states[idx : idx + 1],
                )
                d_on_neg = scorer.stepwise_distances(
                    on_logits[idx : idx + 1],
                    neg_logits[idx : idx + 1],
                    on_states[idx : idx + 1],
                    neg_states[idx : idx + 1],
                )
                benefit = refined.weighted_mean(d_off_pos - d_on_pos, step_weights) + float(sep_reward) * refined.weighted_mean(
                    d_on_neg - d_on_pos,
                    step_weights,
                )
                fr_scores.append(float(torch.clamp(benefit, min=0.0).item()))

    en_chunk_size = _choose_chunk_size(len(en_examples))
    for chunk in batched(list(en_examples), en_chunk_size):
        batch_size = len(chunk)
        on_logits, on_states = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=paver,
            base_ids_batch=[ex.base_ids for ex in chunk],
            anchor_ids_batch=[ex.anchor_ids for ex in chunk],
            controller_vectors=controller_vectors,
            gate_strengths=torch.ones(batch_size, device=controller_vectors.device, dtype=torch.float32),
        )
        if use_cached_refs:
            on_features = embed_live_trace_features(scorer, on_logits, on_states)
            en_cached = gather_english_reference_batch(frozen_cache.en, chunk, device=controller_vectors.device)
            d_harm = compressed_pairwise_paving_distance(
                scorer=scorer,
                train_features=on_features,
                ref_features=en_cached["base"],
                eps=float(scorer.loss_config.eps),
            )
            harm = _weighted_rows(d_harm)
            en_scores.extend(torch.clamp(harm, min=0.0).detach().cpu().tolist())
        else:
            off_logits, off_states = trace_forced_trajectory_batch_optimized(
                model=model,
                paver=paver,
                base_ids_batch=[ex.base_ids for ex in chunk],
                anchor_ids_batch=[ex.anchor_ids for ex in chunk],
                controller_vectors=None,
                gate_strengths=0.0,
            )
            for idx in range(batch_size):
                d_harm = scorer.stepwise_distances(
                    on_logits[idx : idx + 1],
                    off_logits[idx : idx + 1],
                    on_states[idx : idx + 1],
                    off_states[idx : idx + 1],
                )
                harm = refined.weighted_mean(d_harm, step_weights)
                en_scores.append(float(torch.clamp(harm, min=0.0).item()))

    fr_tensor = torch.tensor(fr_scores, device=controller_vectors.device, dtype=torch.float32)
    en_tensor = torch.tensor(en_scores, device=controller_vectors.device, dtype=torch.float32)
    if fr_tensor.numel() > 0 and fr_tensor.max().item() > 0:
        fr_weights = 1.0 + float(utility_scale) * (fr_tensor / fr_tensor.max().clamp_min(1e-6))
    else:
        fr_weights = torch.ones_like(fr_tensor)
    if en_tensor.numel() > 0 and en_tensor.max().item() > 0:
        en_weights = 1.0 + float(utility_scale) * (en_tensor / en_tensor.max().clamp_min(1e-6))
    else:
        en_weights = torch.ones_like(en_tensor)

    stats = {
        "fr_benefit_mean": float(fr_tensor.mean().item()) if fr_tensor.numel() > 0 else 0.0,
        "fr_benefit_max": float(fr_tensor.max().item()) if fr_tensor.numel() > 0 else 0.0,
        "en_harm_mean": float(en_tensor.mean().item()) if en_tensor.numel() > 0 else 0.0,
        "en_harm_max": float(en_tensor.max().item()) if en_tensor.numel() > 0 else 0.0,
    }
    return fr_weights, en_weights, stats
