#!/usr/bin/env python3
"""
gemma3_residual_paving_binary_refined.py

Binary residual-paving trainer for the French -> English steering problem on
Gemma 3 text models.

This version is deliberately narrower than the previous multilingual trainers:
it only models two prompt sets, French and English. The goal is to improve two
weak spots seen in the earlier run:

1. Gate selectivity
   - The gate is now *anchored* to the initial French-vs-English language
     direction instead of being allowed to drift freely.
   - During joint training, the gate learns mostly threshold, temperature, and
     layer weighting, with an optional tiny residual direction.
   - A third calibration stage freezes the controller and calibrates the gate
     using utility-weighted binary losses and a weighted threshold search.

2. Distributional usefulness of the trajectory term
   - Token-level CE / KL are retained for strong local gradients.
   - The trajectory term is made more actionable with three changes:
       * early-step prefix weighting
       * explicit English-vs-French contrastive trajectory margin
       * monotonic paving loss that discourages later drift away from the
         English teacher trajectory
   - The trajectory features are still full-support and differentiable via the
     projected-energy machinery from gemma3_residual_paving_hybrid.py, which
     avoids brittle top-k support transitions.

Conceptually, the controller still learns a multi-layer, multi-step residual
corridor. The training recipe simply makes that corridor more useful at the
start of decoding and makes the gate more trustworthy in the binary setting.

Install:
    pip install -U torch transformers accelerate sentencepiece

Example:
    python gemma3_residual_paving_binary_refined.py train \
      --save_path gemma3_fr_en_binary_refined.pt \
      --preview_hard_gate

    python gemma3_residual_paving_binary_refined.py infer \
      --load_path gemma3_fr_en_binary_refined.pt \
      --prompt "Explique la récursion simplement." \
      --hard_gate

Dependency:
    This script expects gemma3_residual_paving_hybrid.py to be in the same
    directory because it reuses the projected trajectory machinery and the
    whitebox residual-paving hooks.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from gemma3_residual_paving_hybrid import (
    DEFAULT_MODEL_ID,
    PARALLEL_LANGUAGE_PROMPTS,
    CountSketchProjector,  # re-export dependency safety
    HybridLossConfig,
    ProjectionConfig,
    ProjectedTrajectoryScorer,
    ResidualPaver,
    average_dicts,
    batched,
    build_messages,
    capture_boundary_states,
    choose_best_threshold,
    encode_chat_prompt,
    finalize_paving_vectors,
    generate_ids,
    get_decoder_layers,
    get_hidden_size,
    load_model_and_tokenizer,
    model_forward,
    parse_layer_indices,
    paving_regularizer,
    sample_next_token,
    set_seed,
    trace_forced_trajectory,
)


@dataclass
class BinaryExample:
    sample_id: str
    kind: str  # "fr" or "en"
    prompt: str
    paired_prompt: str
    base_ids: torch.Tensor
    anchor_ids: torch.Tensor
    boundary_states: torch.Tensor
    gate_target: float
    teacher_ids: Optional[torch.Tensor] = None
    negative_ids: Optional[torch.Tensor] = None
    bucket: str = "general"
    hard_negative_ids: Optional[torch.Tensor] = None
    answer_mode: str = "general"
    refusal_target: float = 0.0
    length_bucket: str = "medium"


@dataclass
class BinaryTrainState:
    raw_vectors: Optional[torch.Tensor]
    layer_indices: List[int]
    gate_state_dict: Dict[str, torch.Tensor]
    gate_base_direction: torch.Tensor
    gate_threshold: float
    gate_residual_scale: float
    gate_sharpness_init: float
    learn_direction_residual: bool
    model_id: str
    steer_steps: int
    layer_ramp_start: float
    layer_ramp_end: float
    time_decay_tau: float
    post_smooth_layer: float
    post_smooth_time: float
    loss_config: Dict[str, float]
    projection_config: Dict[str, int]
    checkpoint_version: int = 1
    controller_state_dict: Optional[Dict[str, torch.Tensor]] = None
    controller_config: Optional[Dict[str, Any]] = None
    router_state_dict: Optional[Dict[str, torch.Tensor]] = None
    router_config: Optional[Dict[str, Any]] = None
    gate_config: Optional[Dict[str, Any]] = None
    intervention_positions: str = "last_token"


DEFAULT_BUCKET_TO_ANSWER_MODE: Dict[str, str] = {
    "greeting_social_chat": "chat",
    "short_factual_qa": "fact",
    "long_explanation": "explanation",
    "instruction_following": "instruction",
    "safety_refusal": "refusal",
    "code_help": "code",
    "opinion_open_ended": "opinion",
    "summarization_rewrite": "rewrite",
    "list_generation": "list",
    "ambiguous_slang_typo": "slang",
}

DEFAULT_BUCKET_TO_LENGTH_BUCKET: Dict[str, str] = {
    "greeting_social_chat": "short",
    "short_factual_qa": "short",
    "long_explanation": "long",
    "instruction_following": "medium",
    "safety_refusal": "medium",
    "code_help": "medium",
    "opinion_open_ended": "medium",
    "summarization_rewrite": "medium",
    "list_generation": "medium",
    "ambiguous_slang_typo": "short",
}

REFUSAL_BUCKETS = {"safety_refusal"}
SUPPORTED_LENGTH_BUCKETS = ("short", "medium", "long")


def _parse_booleanish(value: str) -> Optional[bool]:
    lowered = (value or "").strip().lower()
    if not lowered:
        return None
    if lowered in {"1", "true", "yes", "y", "oui"}:
        return True
    if lowered in {"0", "false", "no", "n", "non"}:
        return False
    return None


def encode_chat_prompt_with_cap(
    tokenizer,
    user_prompt: str,
    system_prompt: Optional[str],
    device: torch.device,
    *,
    max_prompt_tokens: int = 0,
) -> torch.Tensor:
    if int(max_prompt_tokens) <= 0:
        return encode_chat_prompt(tokenizer, user_prompt, system_prompt, device)

    batch = tokenizer.apply_chat_template(
        build_messages(user_prompt, system_prompt),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"]
    if input_ids.shape[1] > int(max_prompt_tokens):
        input_ids = input_ids[:, -int(max_prompt_tokens) :]
    return input_ids.to(device)


def derive_semantic_targets(prompt_record: Dict[str, str]) -> Tuple[str, float, str]:
    bucket = str(prompt_record.get("bucket", "general")).strip() or "general"
    answer_mode = str(prompt_record.get("answer_mode", "")).strip() or DEFAULT_BUCKET_TO_ANSWER_MODE.get(bucket, bucket)

    refusal_value = _parse_booleanish(str(prompt_record.get("refusal_target", "")))
    if refusal_value is None:
        refusal_target = 1.0 if bucket in REFUSAL_BUCKETS else 0.0
    else:
        refusal_target = 1.0 if refusal_value else 0.0

    length_bucket = str(prompt_record.get("length_bucket", "")).strip().lower()
    if length_bucket not in SUPPORTED_LENGTH_BUCKETS:
        length_bucket = DEFAULT_BUCKET_TO_LENGTH_BUCKET.get(bucket, "medium")

    return answer_mode, float(refusal_target), length_bucket


class AnchoredBinaryGate(nn.Module):
    """Binary language gate anchored to an initial French-vs-English direction.

    The earlier soft gate could drift because its full direction vector was free
    to adapt to controller pressure. Here we keep the initial language-separation
    direction as an anchor and mostly learn:
      - bias / threshold
      - temperature / sharpness
      - per-layer weighting

    An optional tiny residual direction can be enabled, but it is regularized to
    stay small relative to the anchored base direction.
    """

    def __init__(
        self,
        init_direction: torch.Tensor,
        init_threshold: float,
        sharpness: float,
        learn_direction_residual: bool = False,
        residual_scale: float = 0.05,
        gate_type: str = "anchored_linear",
        hidden_dim: int = 64,
    ):
        super().__init__()
        if init_direction.ndim != 2:
            raise ValueError("init_direction must have shape [layers, hidden]")
        gate_key = str(gate_type or "anchored_linear").strip().lower() or "anchored_linear"
        if gate_key not in {"anchored_linear", "anchored_mlp"}:
            raise ValueError(f"Unsupported gate_type: {gate_type!r}")
        self.register_buffer("base_direction", init_direction.detach().float().clone())
        self.bias = nn.Parameter(torch.tensor(-float(init_threshold), dtype=torch.float32))
        self.log_sharpness = nn.Parameter(torch.tensor(math.log(max(float(sharpness), 1e-6)), dtype=torch.float32))
        self.layer_logits = nn.Parameter(torch.zeros(init_direction.shape[0], dtype=torch.float32))
        self.learn_direction_residual = bool(learn_direction_residual)
        self.residual_scale = float(residual_scale)
        self.gate_type = gate_key
        self.hidden_dim = max(int(hidden_dim), 0)
        if self.learn_direction_residual:
            self.direction_residual = nn.Parameter(torch.zeros_like(self.base_direction))
        else:
            self.register_parameter("direction_residual", None)
        if self.gate_type == "anchored_mlp":
            if self.hidden_dim <= 0:
                raise ValueError("anchored_mlp gate_type requires hidden_dim > 0")
            self.mlp_in = nn.Linear(init_direction.shape[1], self.hidden_dim)
            self.mlp_out = nn.Linear(self.hidden_dim, 1, bias=False)
            nn.init.xavier_uniform_(self.mlp_in.weight)
            nn.init.zeros_(self.mlp_in.bias)
            # Start from the anchored linear gate and let the MLP learn only the
            # non-linear correction.
            nn.init.zeros_(self.mlp_out.weight)
        else:
            self.mlp_in = None
            self.mlp_out = None

    def current_direction(self) -> torch.Tensor:
        if self.direction_residual is None:
            direction = self.base_direction
        else:
            direction = self.base_direction + (self.residual_scale * self.direction_residual)
        return F.normalize(direction, dim=-1, eps=1e-6)

    def current_layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def raw_score(self, boundary_states: torch.Tensor) -> torch.Tensor:
        states_n = F.normalize(boundary_states.float(), dim=-1, eps=1e-6)
        dir_n = self.current_direction()
        per_layer_scores = (states_n * dir_n.unsqueeze(0)).sum(dim=-1)  # [B, L]
        layer_weights = self.current_layer_weights()
        score = (per_layer_scores * layer_weights.unsqueeze(0)).sum(dim=-1)
        if self.gate_type == "anchored_mlp":
            pooled = boundary_states.float().mean(dim=1)
            hidden = torch.tanh(self.mlp_in(pooled))
            score = score + self.mlp_out(hidden).squeeze(-1)
        return score + self.bias

    def forward(self, boundary_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.raw_score(boundary_states)
        sharpness = torch.clamp(self.log_sharpness.exp(), min=1.0, max=64.0)
        logits = sharpness * raw
        probs = torch.sigmoid(logits)
        return logits, probs

    def regularizer(
        self,
        residual_l2: float,
        layer_logit_l2: float,
        sharpness_center: float,
        sharpness_l2: float,
    ) -> torch.Tensor:
        reg = self.raw_score(self.base_direction.unsqueeze(0)).new_zeros(())
        if self.direction_residual is not None and residual_l2 > 0:
            reg = reg + residual_l2 * self.direction_residual.pow(2).mean()
        if layer_logit_l2 > 0:
            reg = reg + layer_logit_l2 * self.layer_logits.pow(2).mean()
            if self.gate_type == "anchored_mlp":
                reg = reg + layer_logit_l2 * (
                    self.mlp_in.weight.pow(2).mean()
                    + self.mlp_out.weight.pow(2).mean()
                    + self.mlp_in.bias.pow(2).mean()
                )
        if sharpness_l2 > 0:
            reg = reg + sharpness_l2 * (self.log_sharpness - math.log(max(sharpness_center, 1e-6))) ** 2
        return reg


@torch.no_grad()
def save_binary_state(path: str, state: BinaryTrainState):
    torch.save(asdict(state), path)
    print(f"Saved state to {path}")


@torch.no_grad()
def load_binary_state(path: str, device: torch.device) -> Dict:
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj.get("raw_vectors"), torch.Tensor):
        obj["raw_vectors"] = obj["raw_vectors"].to(device)
    if isinstance(obj.get("gate_base_direction"), torch.Tensor):
        obj["gate_base_direction"] = obj["gate_base_direction"].to(device)
    gate_state_dict = {}
    for key, value in obj["gate_state_dict"].items():
        gate_state_dict[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    obj["gate_state_dict"] = gate_state_dict
    if isinstance(obj.get("controller_state_dict"), dict):
        controller_state = {}
        for key, value in obj["controller_state_dict"].items():
            controller_state[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        obj["controller_state_dict"] = controller_state
    if isinstance(obj.get("router_state_dict"), dict):
        router_state = {}
        for key, value in obj["router_state_dict"].items():
            router_state[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        obj["router_state_dict"] = router_state
    return obj


@torch.no_grad()
def make_bucket_hard_negative_prompt(bucket: str, fr_prompt: str) -> str:
    bucket_key = (bucket or "general").strip().lower()
    templates = {
        "greeting_social_chat": "Reponds a ce message de facon naturelle, surtout en francais, avec un peu de franglais et un ton social detendu : {prompt}",
        "short_factual_qa": "Reponds brievement a cette question, surtout en francais, en glissant quelques mots anglais sans vraiment tout traduire : {prompt}",
        "long_explanation": "Explique ce sujet en detail, surtout en francais, avec quelques tournures anglaises melangees au milieu : {prompt}",
        "instruction_following": "Suis cette demande etape par etape, surtout en francais, sans faire une vraie traduction complete en anglais : {prompt}",
        "safety_refusal": "Refuse clairement cette demande dangereuse, mais fais-le surtout en francais naturel avec au plus un peu de franglais : {prompt}",
        "code_help": "Reponds comme un assistant technique surtout en francais, en gardant les termes de code en anglais et un peu de franglais : {prompt}",
        "opinion_open_ended": "Donne un avis nuance sur ce sujet, surtout en francais, avec quelques expressions anglaises melangees : {prompt}",
        "summarization_rewrite": "Resume ou reecris cette demande surtout en francais, en laissant un leger melange de termes anglais : {prompt}",
        "list_generation": "Reponds sous forme de liste, surtout en francais, avec un peu de franglais naturel : {prompt}",
        "ambiguous_slang_typo": "Reponds naturellement a ce message familier surtout en francais, en gardant le ton relache et un peu de franglais : {prompt}",
    }
    template = templates.get(
        bucket_key,
        "Reponds a cette demande surtout en francais, avec un peu de franglais et sans faire de vraie traduction complete : {prompt}",
    )
    return template.format(prompt=fr_prompt)


def _resolve_pad_token_id(tokenizer) -> int:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        return int(eos_token_id)
    raise ValueError("Tokenizer must define either pad_token_id or eos_token_id for batched prompt building.")


def _left_pad_batch(sequences: Sequence[torch.Tensor], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Expected at least one token sequence to batch.")
    max_len = max(int(seq.numel()) for seq in sequences)
    device = sequences[0].device
    batch = torch.full((len(sequences), max_len), int(pad_token_id), device=device, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), device=device, dtype=torch.long)
    for idx, seq in enumerate(sequences):
        flat = seq.view(-1).to(device=device, dtype=torch.long)
        width = int(flat.numel())
        batch[idx, -width:] = flat
        attention_mask[idx, -width:] = 1
    return batch, attention_mask


def _is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).strip().lower()
    return "out of memory" in message and "cuda" in message


def _run_batched_with_backoff(
    items: Sequence[Any],
    *,
    initial_batch_size: int,
    process_batch,
    on_backoff=None,
) -> None:
    if not items:
        return

    batch_size = max(1, int(initial_batch_size))
    start = 0
    while start < len(items):
        stop = min(start + batch_size, len(items))
        batch = list(items[start:stop])
        try:
            process_batch(batch)
            start = stop
            continue
        except Exception as exc:
            if not _is_cuda_oom_error(exc) or batch_size <= 1:
                raise
            new_batch_size = max(1, batch_size // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if on_backoff is not None:
                on_backoff(batch_size, new_batch_size, start, len(items))
            batch_size = new_batch_size


@torch.no_grad()
def capture_boundary_states_batch(
    model,
    paver: ResidualPaver,
    prompt_ids_batch: Sequence[torch.Tensor],
    *,
    pad_token_id: int,
) -> torch.Tensor:
    input_ids, attention_mask = _left_pad_batch(prompt_ids_batch, pad_token_id)
    with paver.intervene_and_capture(step_vectors=None, batch_scale=None) as captured:
        _ = model_forward(model, input_ids, attention_mask=attention_mask)
    states = [captured[i].float().clone() for i in range(len(paver.modules))]
    return torch.stack(states, dim=1)


@torch.no_grad()
def generate_ids_batch(
    model,
    tokenizer,
    base_ids_batch: Sequence[torch.Tensor],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
    stop_on_eos: bool = False,
    pad_token_id: Optional[int] = None,
) -> List[torch.Tensor]:
    if not base_ids_batch:
        return []

    resolved_pad_token_id = int(_resolve_pad_token_id(tokenizer) if pad_token_id is None else pad_token_id)
    eos_id = tokenizer.eos_token_id
    device = base_ids_batch[0].device
    generated: List[List[int]] = [[] for _ in base_ids_batch]
    finished = [False for _ in base_ids_batch]

    for _ in range(int(max_new_tokens)):
        curr_sequences: List[torch.Tensor] = []
        for base_ids, generated_ids in zip(base_ids_batch, generated):
            if generated_ids:
                gen_tensor = torch.tensor(generated_ids, device=device, dtype=torch.long)
                curr_sequences.append(torch.cat([base_ids.view(-1), gen_tensor], dim=0))
            else:
                curr_sequences.append(base_ids.view(-1))

        input_ids, attention_mask = _left_pad_batch(curr_sequences, resolved_pad_token_id)
        outputs = model_forward(model, input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :].float()
        next_ids = sample_next_token(logits, temperature=temperature, top_p=top_p).view(-1)

        for idx, token_id in enumerate(next_ids.tolist()):
            if finished[idx]:
                continue
            generated[idx].append(int(token_id))
            if stop_on_eos and eos_id is not None and int(token_id) == int(eos_id):
                finished[idx] = True

        if stop_on_eos and all(finished):
            break

    return [torch.tensor(tokens, device=device, dtype=torch.long) for tokens in generated]


@torch.no_grad()
def build_binary_examples(
    model,
    tokenizer,
    paver: ResidualPaver,
    device: torch.device,
    steer_steps: int,
    prompt_records: Optional[Sequence[Dict[str, str]]] = None,
    use_hard_negatives: bool = True,
    max_prompt_tokens: int = 0,
    build_batch_size: int = 1,
    progress_label: Optional[str] = None,
    progress_every: int = 0,
) -> Tuple[List[BinaryExample], List[BinaryExample]]:
    fr_examples: List[BinaryExample] = []
    en_examples: List[BinaryExample] = []

    records = prompt_records if prompt_records is not None else PARALLEL_LANGUAGE_PROMPTS
    total_records = len(records)
    resolved_batch_size = max(1, int(build_batch_size))
    resolved_progress_every = max(0, int(progress_every))
    resolved_pad_token_id = _resolve_pad_token_id(tokenizer)
    processed = 0
    next_progress = resolved_progress_every

    if progress_label:
        print(
            f"[build {progress_label}] total_records={total_records} "
            f"batch_size={resolved_batch_size} hard_negatives={'on' if use_hard_negatives else 'off'}"
        )

    enumerated_records = list(enumerate(records))

    def process_record_batch(record_batch: Sequence[Tuple[int, Dict[str, str]]]) -> None:
        nonlocal processed, next_progress
        batch_meta: List[Tuple[str, str, str, str, str, float, str]] = []
        fr_base_batch: List[torch.Tensor] = []
        en_base_batch: List[torch.Tensor] = []
        hard_negative_prompt_batch: List[torch.Tensor] = []

        for i, item in record_batch:
            fr_prompt = item["fr"]
            en_prompt = item["en"]
            sample_id = str(item.get("sample_id", "")).strip() or str(i)
            bucket = str(item.get("bucket", "general")).strip() or "general"
            answer_mode, refusal_target, length_bucket = derive_semantic_targets(item)

            fr_base_ids = encode_chat_prompt_with_cap(tokenizer, fr_prompt, None, device, max_prompt_tokens=max_prompt_tokens)
            en_base_ids = encode_chat_prompt_with_cap(tokenizer, en_prompt, None, device, max_prompt_tokens=max_prompt_tokens)
            fr_base_batch.append(fr_base_ids)
            en_base_batch.append(en_base_ids)
            batch_meta.append((sample_id, fr_prompt, en_prompt, bucket, answer_mode, refusal_target, length_bucket))

            if use_hard_negatives:
                hard_negative_prompt = str(item.get("fr_hard_negative_prompt", "")).strip()
                if not hard_negative_prompt:
                    hard_negative_prompt = make_bucket_hard_negative_prompt(bucket, fr_prompt)
                hard_negative_prompt_batch.append(
                    encode_chat_prompt_with_cap(
                        tokenizer,
                        hard_negative_prompt,
                        None,
                        device,
                        max_prompt_tokens=max_prompt_tokens,
                    )
                )

        en_anchor_batch = generate_ids_batch(
            model,
            tokenizer,
            en_base_batch,
            max_new_tokens=steer_steps,
            stop_on_eos=False,
            pad_token_id=resolved_pad_token_id,
        )
        fr_negative_batch = generate_ids_batch(
            model,
            tokenizer,
            fr_base_batch,
            max_new_tokens=steer_steps,
            stop_on_eos=False,
            pad_token_id=resolved_pad_token_id,
        )
        hard_negative_id_batch: List[Optional[torch.Tensor]] = [None for _ in batch_meta]
        if use_hard_negatives:
            hard_negative_generated = generate_ids_batch(
                model,
                tokenizer,
                hard_negative_prompt_batch,
                max_new_tokens=steer_steps,
                stop_on_eos=False,
                pad_token_id=resolved_pad_token_id,
            )
            hard_negative_id_batch = list(hard_negative_generated)

        fr_boundary_batch = capture_boundary_states_batch(
            model,
            paver,
            fr_base_batch,
            pad_token_id=resolved_pad_token_id,
        )
        en_boundary_batch = capture_boundary_states_batch(
            model,
            paver,
            en_base_batch,
            pad_token_id=resolved_pad_token_id,
        )

        for idx, meta in enumerate(batch_meta):
            sample_id, fr_prompt, en_prompt, bucket, answer_mode, refusal_target, length_bucket = meta
            fr_base_ids = fr_base_batch[idx]
            en_base_ids = en_base_batch[idx]
            en_anchor = en_anchor_batch[idx]
            fr_negative = fr_negative_batch[idx]
            hard_negative_ids = hard_negative_id_batch[idx]
            fr_boundary = fr_boundary_batch[idx]
            en_boundary = en_boundary_batch[idx]

            fr_examples.append(
                BinaryExample(
                    sample_id=f"fr_{sample_id}",
                    kind="fr",
                    prompt=fr_prompt,
                    paired_prompt=en_prompt,
                    base_ids=fr_base_ids,
                    anchor_ids=en_anchor,
                    boundary_states=fr_boundary,
                    gate_target=1.0,
                    teacher_ids=en_base_ids,
                    negative_ids=fr_negative,
                    bucket=bucket,
                    hard_negative_ids=hard_negative_ids,
                    answer_mode=answer_mode,
                    refusal_target=refusal_target,
                    length_bucket=length_bucket,
                )
            )
            en_examples.append(
                BinaryExample(
                    sample_id=f"en_{sample_id}",
                    kind="en",
                    prompt=en_prompt,
                    paired_prompt=fr_prompt,
                    base_ids=en_base_ids,
                    anchor_ids=en_anchor,
                    boundary_states=en_boundary,
                    gate_target=0.0,
                    teacher_ids=None,
                    negative_ids=None,
                    bucket=bucket,
                    hard_negative_ids=None,
                    answer_mode=answer_mode,
                    refusal_target=refusal_target,
                    length_bucket=length_bucket,
                )
            )

        processed += len(record_batch)
        if progress_label and (
            processed == total_records
            or (resolved_progress_every > 0 and processed >= next_progress)
        ):
            print(f"[build {progress_label}] processed={processed}/{total_records}")
            if resolved_progress_every > 0:
                while next_progress <= processed:
                    next_progress += resolved_progress_every

    def log_backoff(prev_batch_size: int, new_batch_size: int, start: int, total: int) -> None:
        label = progress_label or "binary_examples"
        print(
            f"[build {label}] CUDA OOM at batch_size={prev_batch_size} "
            f"for records {start}:{min(start + prev_batch_size, total)}; retrying with batch_size={new_batch_size}"
        )

    _run_batched_with_backoff(
        enumerated_records,
        initial_batch_size=resolved_batch_size,
        process_batch=process_record_batch,
        on_backoff=log_backoff,
    )

    return fr_examples, en_examples


@torch.no_grad()
def preview_teacher_pairs(tokenizer, examples: Sequence[BinaryExample], limit: int = 2) -> None:
    print("[teacher preview]")
    for ex in examples[:limit]:
        pos = tokenizer.decode(ex.anchor_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        neg = tokenizer.decode(ex.negative_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) if ex.negative_ids is not None else ""
        hard_neg = tokenizer.decode(ex.hard_negative_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) if ex.hard_negative_ids is not None else ""
        print(f"  {ex.sample_id} pos={pos[:160]!r}")
        print(f"  {ex.sample_id} neg={neg[:160]!r}")
        if hard_neg:
            print(f"  {ex.sample_id} hard_neg={hard_neg[:160]!r}")


@torch.no_grad()
def init_binary_gate_from_examples(
    fr_examples: Sequence[BinaryExample],
    en_examples: Sequence[BinaryExample],
) -> Tuple[torch.Tensor, float, Dict[str, float]]:
    fr_states = torch.stack([ex.boundary_states for ex in fr_examples], dim=0)
    en_states = torch.stack([ex.boundary_states for ex in en_examples], dim=0)

    direction = fr_states.mean(dim=0) - en_states.mean(dim=0)
    direction = F.normalize(direction, dim=-1, eps=1e-6)

    fr_scores = (F.normalize(fr_states, dim=-1, eps=1e-6) * direction.unsqueeze(0)).sum(dim=-1).mean(dim=-1)
    en_scores = (F.normalize(en_states, dim=-1, eps=1e-6) * direction.unsqueeze(0)).sum(dim=-1).mean(dim=-1)
    threshold, acc = choose_best_threshold(fr_scores, en_scores)

    stats = {
        "threshold": float(threshold),
        "train_acc": float(acc),
        "fr_mean": float(fr_scores.mean().item()),
        "en_mean": float(en_scores.mean().item()),
        "fr_std": float(fr_scores.std(unbiased=False).item()),
        "en_std": float(en_scores.std(unbiased=False).item()),
    }
    return direction.detach(), float(threshold), stats


@torch.no_grad()
def binary_gate_stats(gate: AnchoredBinaryGate, fr_examples: Sequence[BinaryExample], en_examples: Sequence[BinaryExample]) -> Dict[str, float]:
    fr_boundary = torch.stack([ex.boundary_states for ex in fr_examples], dim=0)
    en_boundary = torch.stack([ex.boundary_states for ex in en_examples], dim=0)
    fr_raw = gate.raw_score(fr_boundary)
    en_raw = gate.raw_score(en_boundary)
    _, fr_probs = gate(fr_boundary)
    _, en_probs = gate(en_boundary)
    pred_acc = torch.cat([(fr_raw > 0).float(), (en_raw <= 0).float()]).mean().item()
    return {
        "fr_gate_mean": float(fr_probs.mean().item()),
        "en_gate_mean": float(en_probs.mean().item()),
        "fr_raw_mean": float(fr_raw.mean().item()),
        "en_raw_mean": float(en_raw.mean().item()),
        "threshold_acc": float(pred_acc),
    }


@torch.no_grad()
def collect_fr_references(
    model,
    paver: ResidualPaver,
    examples: Sequence[BinaryExample],
    max_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_logits_list = []
    base_states_list = []
    pos_logits_list = []
    pos_states_list = []
    neg_logits_list = []
    neg_states_list = []

    for ex in examples:
        assert ex.teacher_ids is not None and ex.negative_ids is not None
        base_logits, base_states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.base_ids,
            anchor_ids=ex.anchor_ids[:max_steps] if max_steps is not None and int(max_steps) > 0 else ex.anchor_ids,
            controller_vectors=None,
            gate_strength=0.0,
        )
        pos_logits, pos_states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.teacher_ids,
            anchor_ids=ex.anchor_ids[:max_steps] if max_steps is not None and int(max_steps) > 0 else ex.anchor_ids,
            controller_vectors=None,
            gate_strength=0.0,
        )
        neg_logits, neg_states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.base_ids,
            anchor_ids=ex.negative_ids[:max_steps] if max_steps is not None and int(max_steps) > 0 else ex.negative_ids,
            controller_vectors=None,
            gate_strength=0.0,
        )
        base_logits_list.append(base_logits)
        base_states_list.append(base_states)
        pos_logits_list.append(pos_logits)
        pos_states_list.append(pos_states)
        neg_logits_list.append(neg_logits)
        neg_states_list.append(neg_states)

    return (
        torch.stack(base_logits_list, dim=0),
        torch.stack(base_states_list, dim=0),
        torch.stack(pos_logits_list, dim=0),
        torch.stack(pos_states_list, dim=0),
        torch.stack(neg_logits_list, dim=0),
        torch.stack(neg_states_list, dim=0),
    )


@torch.no_grad()
def collect_en_references(
    model,
    paver: ResidualPaver,
    examples: Sequence[BinaryExample],
    max_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    base_logits_list = []
    base_states_list = []
    for ex in examples:
        base_logits, base_states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.base_ids,
            anchor_ids=ex.anchor_ids[:max_steps] if max_steps is not None and int(max_steps) > 0 else ex.anchor_ids,
            controller_vectors=None,
            gate_strength=0.0,
        )
        base_logits_list.append(base_logits)
        base_states_list.append(base_states)
    return torch.stack(base_logits_list, dim=0), torch.stack(base_states_list, dim=0)


def gather_train_traces_with_strengths(
    model,
    paver: ResidualPaver,
    controller_vectors: torch.Tensor,
    examples: Sequence[BinaryExample],
    gate_strengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_list = []
    states_list = []
    for ex, gate_strength in zip(examples, gate_strengths):
        logits, states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.base_ids,
            anchor_ids=ex.anchor_ids,
            controller_vectors=controller_vectors,
            gate_strength=gate_strength,
        )
        logits_list.append(logits)
        states_list.append(states)
    return torch.stack(logits_list, dim=0), torch.stack(states_list, dim=0)


def gather_train_traces_with_gate(
    model,
    paver: ResidualPaver,
    controller_vectors: torch.Tensor,
    gate: AnchoredBinaryGate,
    examples: Sequence[BinaryExample],
    pos_floor: float,
    neg_cap: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boundary_batch = torch.stack([ex.boundary_states for ex in examples], dim=0)
    gate_logits, gate_probs = gate(boundary_batch)
    labels = torch.tensor([ex.gate_target for ex in examples], device=gate_logits.device, dtype=gate_logits.dtype)
    effective = torch.where(
        labels > 0.5,
        pos_floor + (1.0 - pos_floor) * gate_probs,
        neg_cap * gate_probs,
    )
    train_logits, train_states = gather_train_traces_with_strengths(
        model=model,
        paver=paver,
        controller_vectors=controller_vectors,
        examples=examples,
        gate_strengths=effective,
    )
    return train_logits, train_states, gate_logits, gate_probs, labels, effective


@torch.no_grad()
def decode_with_binary_gate(
    model,
    tokenizer,
    paver: ResidualPaver,
    gate: AnchoredBinaryGate,
    controller_vectors: torch.Tensor,
    device: torch.device,
    user_prompt: str,
    max_new_tokens: int,
    strength: float,
    temperature: float,
    top_p: float,
    hard_gate: bool,
) -> Tuple[float, float, str]:
    base_ids = encode_chat_prompt(tokenizer, user_prompt, None, device)
    boundary = capture_boundary_states(model, base_ids, paver).unsqueeze(0)
    gate_raw = float(gate.raw_score(boundary).item())
    _, gate_prob = gate(boundary)
    gate_prob_f = float(gate_prob.item())
    if hard_gate:
        gate_strength = float(gate_raw > 0.0) * float(strength)
    else:
        gate_strength = gate_prob_f * float(strength)

    generated: List[int] = []
    for step in range(max_new_tokens):
        if generated:
            gen_tensor = torch.tensor(generated, device=device, dtype=torch.long).view(1, -1)
            curr_ids = torch.cat([base_ids, gen_tensor], dim=1)
        else:
            curr_ids = base_ids

        use_edit = step < controller_vectors.shape[0] and abs(gate_strength) > 1e-8
        with paver.intervene_and_capture(
            step_vectors=controller_vectors[step] if use_edit else None,
            batch_scale=gate_strength if use_edit else None,
        ):
            outputs = model_forward(model, curr_ids)
        logits = outputs.logits[:, -1, :].float()
        next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
        generated.append(int(next_id.item()))
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    text = tokenizer.decode(torch.tensor(generated, device=device), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return gate_raw, gate_strength, text


def make_step_weights(
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    focus_steps: int,
    focus_boost: float,
    decay_tau: float,
) -> torch.Tensor:
    idx = torch.arange(max(int(steps), 1), device=device, dtype=dtype)
    if decay_tau > 0:
        weights = torch.exp(-idx / float(decay_tau))
    else:
        weights = torch.ones_like(idx)
    if focus_steps > 0:
        weights[: min(int(focus_steps), weights.numel())] *= float(max(focus_boost, 1.0))
    return weights / weights.sum().clamp_min(1e-6)


def make_piecewise_step_weights(
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    strong_steps: int,
    medium_steps: int,
    strong_weight: float,
    medium_weight: float,
    tail_weight: float,
) -> torch.Tensor:
    count = max(int(steps), 1)
    weights = torch.full((count,), float(max(tail_weight, 1e-6)), device=device, dtype=dtype)
    strong_end = max(0, min(int(strong_steps), count))
    medium_end = max(strong_end, min(strong_end + max(int(medium_steps), 0), count))
    if strong_end > 0:
        weights[:strong_end] = float(max(strong_weight, 1e-6))
    if medium_end > strong_end:
        weights[strong_end:medium_end] = float(max(medium_weight, 1e-6))
    return weights / weights.sum().clamp_min(1e-6)


def _dedupe_in_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _apply_typo_variant(text: str) -> str:
    result = text
    replacements = {
        "quoi": "kwa",
        "est-ce": "eske",
        "s'il te plait": "stp",
        "bonjour": "bjr",
        "pourquoi": "pk",
        "avec": "avc",
    }
    lowered = result.lower()
    for src, dst in replacements.items():
        if src in lowered:
            idx = lowered.index(src)
            result = result[:idx] + dst + result[idx + len(src) :]
            return result
    if len(result) > 6:
        return result[:-1]
    return result


def make_prompt_variants(prompt: str, bucket: str, max_variants: int = 2) -> List[str]:
    if max_variants <= 0:
        return []
    bucket_key = (bucket or "general").strip().lower()
    base = prompt.strip()
    variants: List[str] = []

    punctuation_variant = base
    if punctuation_variant.endswith("?"):
        punctuation_variant = punctuation_variant[:-1] + " ?"
    elif punctuation_variant.endswith("!"):
        punctuation_variant = punctuation_variant[:-1] + "..."
    else:
        punctuation_variant = punctuation_variant + " !"
    variants.append(punctuation_variant)

    if bucket_key == "instruction_following":
        variants.append(f"S'il te plait, suis bien chaque etape : {base}")
    elif bucket_key == "safety_refusal":
        variants.append(f"Contexte rapide : {base}\nReponds prudemment.")
    elif bucket_key == "code_help":
        variants.append(f"Petit contexte technique : {base}")
    elif bucket_key == "ambiguous_slang_typo":
        variants.append(_apply_typo_variant(base))
    else:
        variants.append(f"Contexte rapide : {base}")

    variants.append(_apply_typo_variant(base))
    variants.append(f"{base}\nReponds de facon concise.")
    return _dedupe_in_order(variants)[: int(max_variants)]


@torch.no_grad()
def build_prompt_variant_examples(
    model,
    tokenizer,
    paver: ResidualPaver,
    device: torch.device,
    examples: Sequence[BinaryExample],
    *,
    max_variants: int = 1,
    variant_index: int = 0,
    max_prompt_tokens: int = 0,
) -> List[BinaryExample]:
    variant_examples: List[BinaryExample] = []
    if max_variants <= 0:
        return variant_examples

    for ex in examples:
        variants = make_prompt_variants(ex.prompt, ex.bucket, max_variants=max(max_variants, variant_index + 1))
        if not variants:
            continue
        prompt_variant = variants[min(int(variant_index), len(variants) - 1)]
        base_ids = encode_chat_prompt_with_cap(
            tokenizer,
            prompt_variant,
            None,
            device,
            max_prompt_tokens=max_prompt_tokens,
        )
        boundary_states = capture_boundary_states(model, base_ids, paver)
        variant_examples.append(
            BinaryExample(
                sample_id=f"{ex.sample_id}_variant_{min(int(variant_index), len(variants) - 1)}",
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


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    k = min(values.shape[0], weights.shape[0])
    if k <= 0:
        return values.new_zeros(())
    w = weights[:k]
    return (values[:k] * w).sum() / w.sum().clamp_min(1e-6)


def step_weighted_ce(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    k = min(token_nll.shape[1], weights.shape[0])
    w = weights[:k].view(1, k)
    return (token_nll[:, :k] * w).sum() / token_nll.shape[0]


def step_weighted_kl(train_logits: torch.Tensor, ref_logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(train_logits.float(), dim=-1)
    q = torch.softmax(ref_logits.detach().float(), dim=-1)
    per_token = F.kl_div(log_p, q, reduction="none").sum(dim=-1)
    k = min(per_token.shape[1], weights.shape[0])
    w = weights[:k].view(1, k)
    return (per_token[:, :k] * w).sum() / per_token.shape[0]


def weighted_token_margin_loss(
    logits: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    weights: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    pos_logits = logits.gather(-1, pos_ids.unsqueeze(-1)).squeeze(-1)
    neg_logits = logits.gather(-1, neg_ids.unsqueeze(-1)).squeeze(-1)
    delta = pos_logits - neg_logits
    loss = F.softplus(float(margin) - delta)
    k = min(loss.shape[1], weights.shape[0])
    w = weights[:k].view(1, k)
    return (loss[:, :k] * w).sum() / loss.shape[0]


@torch.no_grad()
def english_fraction(logits: torch.Tensor, english_ids: torch.Tensor, french_ids: torch.Tensor, steps: int) -> float:
    k = min(int(steps), logits.shape[1], english_ids.shape[1], french_ids.shape[1])
    if k <= 0:
        return 0.0
    pred = logits[:, :k, :].argmax(dim=-1)
    pos = (pred == english_ids[:, :k]).float().mean().item()
    neg = (pred == french_ids[:, :k]).float().mean().item()
    return float(pos - neg)


def gate_pair_margin_loss(
    fr_logits: torch.Tensor,
    en_logits: torch.Tensor,
    margin: float,
    fr_weights: Optional[torch.Tensor] = None,
    en_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if fr_logits.numel() == 0 or en_logits.numel() == 0:
        return fr_logits.new_zeros(())
    pairwise = fr_logits.unsqueeze(1) - en_logits.unsqueeze(0)
    loss = F.softplus(float(margin) - pairwise)
    if fr_weights is None or en_weights is None:
        return loss.mean()
    weights = fr_weights.view(-1, 1) * en_weights.view(1, -1)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


def gate_saturation_penalty(probs: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    sat = probs * (1.0 - probs)
    if weights is None:
        return sat.mean()
    return (sat * weights).sum() / weights.sum().clamp_min(1e-6)


def fr_trajectory_terms(
    scorer: ProjectedTrajectoryScorer,
    train_logits: torch.Tensor,
    train_states: torch.Tensor,
    pos_logits: torch.Tensor,
    pos_states: torch.Tensor,
    base_logits: torch.Tensor,
    base_states: torch.Tensor,
    neg_logits: torch.Tensor,
    neg_states: torch.Tensor,
    step_weights: torch.Tensor,
    target_fraction: float,
    floor: float,
    sep_margin: float,
    sep_scale: float,
    mono_tol: float,
    eps: float,
) -> Dict[str, torch.Tensor]:
    d_pos_train = scorer.stepwise_distances(train_logits, pos_logits, train_states, pos_states)
    d_pos_base = scorer.stepwise_distances(base_logits, pos_logits, base_states, pos_states)
    d_neg_train = scorer.stepwise_distances(train_logits, neg_logits, train_states, neg_states)

    denom = torch.clamp(d_pos_base.detach(), min=float(floor))
    target = float(target_fraction) * denom
    close = F.softplus((d_pos_train - target) / (denom + float(eps)))
    sep = F.softplus((float(sep_margin) + d_pos_train - d_neg_train) / max(float(sep_scale), 1e-6))
    if d_pos_train.shape[0] > 1:
        mono = F.relu(d_pos_train[1:] - d_pos_train[:-1] - float(mono_tol))
    else:
        mono = d_pos_train.new_zeros((0,))

    return {
        "close": weighted_mean(close, step_weights),
        "sep": weighted_mean(sep, step_weights),
        "mono": weighted_mean(mono, step_weights[1:]) if mono.numel() > 0 else d_pos_train.new_zeros(()),
        "d_pos_train": d_pos_train.detach(),
        "d_neg_train": d_neg_train.detach(),
        "d_pos_base": d_pos_base.detach(),
    }


def en_trajectory_terms(
    scorer: ProjectedTrajectoryScorer,
    train_logits: torch.Tensor,
    train_states: torch.Tensor,
    base_logits: torch.Tensor,
    base_states: torch.Tensor,
    step_weights: torch.Tensor,
    tolerance: float,
    scale: float,
    mono_tol: float,
) -> Dict[str, torch.Tensor]:
    d_base = scorer.stepwise_distances(train_logits, base_logits, train_states, base_states)
    preserve = F.softplus((d_base - float(tolerance)) / max(float(scale), 1e-6))
    if d_base.shape[0] > 1:
        mono = F.relu(d_base[1:] - d_base[:-1] - float(mono_tol))
    else:
        mono = d_base.new_zeros((0,))
    return {
        "preserve": weighted_mean(preserve, step_weights),
        "mono": weighted_mean(mono, step_weights[1:]) if mono.numel() > 0 else d_base.new_zeros(()),
        "d_base": d_base.detach(),
    }


@torch.no_grad()
def compute_gate_utility_weights(
    model,
    paver: ResidualPaver,
    scorer: ProjectedTrajectoryScorer,
    controller_vectors: torch.Tensor,
    fr_examples: Sequence[BinaryExample],
    en_examples: Sequence[BinaryExample],
    step_weights: torch.Tensor,
    sep_reward: float,
    utility_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    fr_scores: List[float] = []
    en_scores: List[float] = []

    for ex in fr_examples:
        assert ex.teacher_ids is not None and ex.negative_ids is not None
        on_logits, on_states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.base_ids,
            anchor_ids=ex.anchor_ids,
            controller_vectors=controller_vectors,
            gate_strength=1.0,
        )
        off_logits, off_states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.base_ids,
            anchor_ids=ex.anchor_ids,
            controller_vectors=None,
            gate_strength=0.0,
        )
        pos_logits, pos_states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.teacher_ids,
            anchor_ids=ex.anchor_ids,
            controller_vectors=None,
            gate_strength=0.0,
        )
        neg_logits, neg_states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.base_ids,
            anchor_ids=ex.negative_ids,
            controller_vectors=None,
            gate_strength=0.0,
        )

        d_on_pos = scorer.stepwise_distances(on_logits.unsqueeze(0), pos_logits.unsqueeze(0), on_states.unsqueeze(0), pos_states.unsqueeze(0))
        d_off_pos = scorer.stepwise_distances(off_logits.unsqueeze(0), pos_logits.unsqueeze(0), off_states.unsqueeze(0), pos_states.unsqueeze(0))
        d_on_neg = scorer.stepwise_distances(on_logits.unsqueeze(0), neg_logits.unsqueeze(0), on_states.unsqueeze(0), neg_states.unsqueeze(0))

        benefit = weighted_mean(d_off_pos - d_on_pos, step_weights) + float(sep_reward) * weighted_mean(d_on_neg - d_on_pos, step_weights)
        fr_scores.append(float(torch.clamp(benefit, min=0.0).item()))

    for ex in en_examples:
        on_logits, on_states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.base_ids,
            anchor_ids=ex.anchor_ids,
            controller_vectors=controller_vectors,
            gate_strength=1.0,
        )
        off_logits, off_states = trace_forced_trajectory(
            model=model,
            paver=paver,
            base_ids=ex.base_ids,
            anchor_ids=ex.anchor_ids,
            controller_vectors=None,
            gate_strength=0.0,
        )
        d_harm = scorer.stepwise_distances(on_logits.unsqueeze(0), off_logits.unsqueeze(0), on_states.unsqueeze(0), off_states.unsqueeze(0))
        harm = weighted_mean(d_harm, step_weights)
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


@torch.no_grad()
def choose_weighted_threshold(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    pos_weights: Optional[torch.Tensor] = None,
    neg_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    pooled = torch.cat([pos_scores, neg_scores], dim=0).cpu().tolist()
    pooled = sorted(set(float(x) for x in pooled))
    if len(pooled) == 1:
        return float(pooled[0]), 0.5

    candidates = [pooled[0] - 1e-6]
    candidates += [(pooled[i] + pooled[i + 1]) / 2.0 for i in range(len(pooled) - 1)]
    candidates += [pooled[-1] + 1e-6]

    if pos_weights is None:
        pos_weights = torch.ones_like(pos_scores)
    if neg_weights is None:
        neg_weights = torch.ones_like(neg_scores)

    pos = pos_scores.detach()
    neg = neg_scores.detach()
    pw = pos_weights.detach().to(pos)
    nw = neg_weights.detach().to(neg)

    best_tau = candidates[0]
    best_acc = -1.0
    for tau in candidates:
        pos_hit = ((pos > tau).float() * pw).sum()
        neg_hit = ((neg <= tau).float() * nw).sum()
        total = pw.sum() + nw.sum()
        acc = float(((pos_hit + neg_hit) / total.clamp_min(1e-6)).item())
        if acc > best_acc:
            best_acc = acc
            best_tau = tau
    return float(best_tau), float(best_acc)


def train_binary_refined(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_id, device)
    total_layers = len(get_decoder_layers(model))
    layer_indices = parse_layer_indices(args.layers, total_layers)
    paver = ResidualPaver(model=model, layer_indices=layer_indices, intervention_positions="last_token")
    hidden_size = get_hidden_size(model)
    num_layers = len(layer_indices)
    vocab_size = int(model.lm_head.weight.shape[0])

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    loss_config = HybridLossConfig(
        ce_weight=args.ce_weight,
        fr_kl_weight=args.fr_kl_weight,
        nonfr_kl_weight=args.en_kl_weight,
        fr_traj_weight=args.fr_traj_weight,
        nonfr_traj_weight=args.en_traj_weight,
        traj_prob_weight=args.traj_prob_weight,
        traj_logit_weight=args.traj_logit_weight,
        traj_state_weight=args.traj_state_weight,
        fr_improvement_target=args.fr_improvement_target,
        fr_baseline_floor=args.fr_baseline_floor,
        preserve_tolerance=args.en_preserve_tolerance,
        preserve_scale=args.en_preserve_scale,
        gate_bce_weight=args.gate_bce_weight,
        gate_entropy_weight=0.0,
        l2=args.l2,
        layer_smooth=args.layer_smooth,
        time_smooth=args.time_smooth,
    )
    proj_config = ProjectionConfig(
        vocab_proj_dim=args.vocab_proj_dim,
        state_proj_dim=args.state_proj_dim,
        proj_seed=args.seed,
    )
    scorer = ProjectedTrajectoryScorer(
        vocab_size=vocab_size,
        flat_state_dim=num_layers * hidden_size,
        config=loss_config,
        proj_config=proj_config,
    ).to(device)

    build_batch_size = max(
        1,
        int(args.build_batch_size)
        if int(args.build_batch_size) > 0
        else min(8, int(args.fr_batch_size)),
    )

    print("Building French / English training examples ...")
    fr_examples, en_examples = build_binary_examples(
        model=model,
        tokenizer=tokenizer,
        paver=paver,
        device=device,
        steer_steps=args.steer_steps,
        build_batch_size=build_batch_size,
        progress_label="train",
        progress_every=args.build_progress_every,
    )
    print(f"French examples: {len(fr_examples)}")
    print(f"English examples: {len(en_examples)}")
    preview_teacher_pairs(tokenizer, fr_examples, limit=min(2, len(fr_examples)))

    init_direction, init_threshold, init_stats = init_binary_gate_from_examples(fr_examples, en_examples)
    gate = AnchoredBinaryGate(
        init_direction=init_direction,
        init_threshold=init_threshold,
        sharpness=args.gate_sharpness,
        learn_direction_residual=args.learn_direction_residual,
        residual_scale=args.gate_residual_scale,
    ).to(device)

    raw_vectors = nn.Parameter(torch.zeros(args.steer_steps, num_layers, hidden_size, device=device, dtype=torch.float32))
    opt_stage1 = torch.optim.Adam([raw_vectors], lr=args.lr_stage1)
    opt_stage2 = torch.optim.Adam([raw_vectors] + list(gate.parameters()), lr=args.lr_stage2)
    opt_stage3 = torch.optim.Adam(list(gate.parameters()), lr=args.lr_stage3)

    step_weights = make_step_weights(
        args.steer_steps,
        device=device,
        dtype=torch.float32,
        focus_steps=args.prefix_focus_steps,
        focus_boost=args.prefix_focus_boost,
        decay_tau=args.prefix_decay_tau,
    )

    print(
        "[gate init] "
        f"train_acc={init_stats['train_acc']:.3f} "
        f"fr_mean={init_stats['fr_mean']:.4f} "
        f"en_mean={init_stats['en_mean']:.4f} "
        f"threshold={init_stats['threshold']:.4f}"
    )

    def current_controller() -> torch.Tensor:
        return finalize_paving_vectors(
            raw_vectors,
            layer_start=args.layer_ramp_start,
            layer_end=args.layer_ramp_end,
            time_decay_tau=args.time_decay_tau,
            smooth_layer=args.post_smooth_layer,
            smooth_time=args.post_smooth_time,
        )

    preview_fr = args.preview_fr_prompt or PARALLEL_LANGUAGE_PROMPTS[0]["fr"]
    preview_en = args.preview_en_prompt or PARALLEL_LANGUAGE_PROMPTS[0]["en"]

    # -------------------- Stage 1: controller only --------------------
    for epoch in range(1, args.stage1_epochs + 1):
        fr_shuffled = list(fr_examples)
        random.shuffle(fr_shuffled)
        fr_batches = batched(fr_shuffled, args.fr_batch_size)
        epoch_records: List[Dict[str, float]] = []

        for fr_batch in fr_batches:
            opt_stage1.zero_grad(set_to_none=True)
            controller = current_controller()
            gate_strengths = torch.ones(len(fr_batch), device=device, dtype=torch.float32) * float(args.stage1_fr_gate)
            fr_train_logits, fr_train_states = gather_train_traces_with_strengths(
                model=model,
                paver=paver,
                controller_vectors=controller,
                examples=fr_batch,
                gate_strengths=gate_strengths,
            )
            (
                fr_base_logits,
                fr_base_states,
                fr_pos_logits,
                fr_pos_states,
                fr_neg_logits,
                fr_neg_states,
            ) = collect_fr_references(model=model, paver=paver, examples=fr_batch)

            fr_targets = torch.stack([ex.anchor_ids for ex in fr_batch], dim=0)
            fr_negative_ids = torch.stack([ex.negative_ids for ex in fr_batch], dim=0)

            fr_ce = step_weighted_ce(fr_train_logits, fr_targets, step_weights)
            fr_kl = step_weighted_kl(fr_train_logits, fr_pos_logits, step_weights)
            fr_token_margin = weighted_token_margin_loss(
                fr_train_logits,
                fr_targets,
                fr_negative_ids,
                step_weights,
                margin=args.token_margin,
            )
            fr_traj = fr_trajectory_terms(
                scorer=scorer,
                train_logits=fr_train_logits,
                train_states=fr_train_states,
                pos_logits=fr_pos_logits,
                pos_states=fr_pos_states,
                base_logits=fr_base_logits,
                base_states=fr_base_states,
                neg_logits=fr_neg_logits,
                neg_states=fr_neg_states,
                step_weights=step_weights,
                target_fraction=loss_config.fr_improvement_target,
                floor=loss_config.fr_baseline_floor,
                sep_margin=args.traj_sep_margin,
                sep_scale=args.traj_sep_scale,
                mono_tol=args.traj_mono_tol,
                eps=loss_config.eps,
            )
            reg = paving_regularizer(
                vectors=raw_vectors,
                l2=loss_config.l2,
                layer_smooth=loss_config.layer_smooth,
                time_smooth=loss_config.time_smooth,
            )

            total = (
                loss_config.ce_weight * fr_ce
                + loss_config.fr_kl_weight * fr_kl
                + args.fr_token_margin_weight * fr_token_margin
                + args.fr_traj_close_weight * fr_traj["close"]
                + args.fr_traj_sep_weight * fr_traj["sep"]
                + args.fr_traj_mono_weight * fr_traj["mono"]
                + reg
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_([raw_vectors], max_norm=1.0)
            opt_stage1.step()

            epoch_records.append(
                {
                    "total": float(total.detach().cpu()),
                    "fr_ce": float(fr_ce.detach().cpu()),
                    "fr_kl": float(fr_kl.detach().cpu()),
                    "fr_tok_margin": float(fr_token_margin.detach().cpu()),
                    "fr_traj_close": float(fr_traj["close"].detach().cpu()),
                    "fr_traj_sep": float(fr_traj["sep"].detach().cpu()),
                    "fr_traj_mono": float(fr_traj["mono"].detach().cpu()),
                    "eng_minus_fr": english_fraction(fr_train_logits.detach(), fr_targets, fr_negative_ids, args.metric_steps),
                }
            )

        avg = average_dicts(epoch_records)
        print(
            f"[stage1 epoch {epoch:02d}] "
            f"total={avg.get('total', 0.0):.4f} "
            f"fr_ce={avg.get('fr_ce', 0.0):.4f} "
            f"fr_kl={avg.get('fr_kl', 0.0):.4f} "
            f"fr_tok_margin={avg.get('fr_tok_margin', 0.0):.4f} "
            f"fr_traj_close={avg.get('fr_traj_close', 0.0):.4f} "
            f"fr_traj_sep={avg.get('fr_traj_sep', 0.0):.4f} "
            f"fr_traj_mono={avg.get('fr_traj_mono', 0.0):.4f} "
            f"eng_minus_fr={avg.get('eng_minus_fr', 0.0):.4f}"
        )
        if args.preview_fr_prompt or args.preview_en_prompt:
            controller_eval = current_controller().detach()
            fr_raw, fr_strength, fr_text = decode_with_binary_gate(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                gate=gate,
                controller_vectors=controller_eval,
                device=device,
                user_prompt=preview_fr,
                max_new_tokens=args.max_new_tokens,
                strength=1.0,
                temperature=args.temperature,
                top_p=args.top_p,
                hard_gate=True,
            )
            en_raw, en_strength, en_text = decode_with_binary_gate(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                gate=gate,
                controller_vectors=controller_eval,
                device=device,
                user_prompt=preview_en,
                max_new_tokens=args.max_new_tokens,
                strength=1.0,
                temperature=args.temperature,
                top_p=args.top_p,
                hard_gate=True,
            )
            print(f"  stage1_fr_gate_raw={fr_raw:.3f} strength={fr_strength:.3f} preview={fr_text[:220]!r}")
            print(f"  stage1_en_gate_raw={en_raw:.3f} strength={en_strength:.3f} preview={en_text[:220]!r}")

    # -------------------- Stage 2: controller + anchored gate --------------------
    for epoch in range(1, args.stage2_epochs + 1):
        fr_shuffled = list(fr_examples)
        en_shuffled = list(en_examples)
        random.shuffle(fr_shuffled)
        random.shuffle(en_shuffled)

        fr_batches = batched(fr_shuffled, args.fr_batch_size)
        en_batches = batched(en_shuffled, args.en_batch_size)
        fr_cycle = itertools.cycle(fr_batches)
        en_cycle = itertools.cycle(en_batches)
        num_updates = max(len(fr_batches), len(en_batches))
        epoch_records: List[Dict[str, float]] = []

        for _ in range(num_updates):
            opt_stage2.zero_grad(set_to_none=True)
            controller = current_controller()

            fr_batch = next(fr_cycle)
            en_batch = next(en_cycle)

            (
                fr_train_logits,
                fr_train_states,
                fr_gate_logits,
                fr_gate_probs,
                fr_gate_targets,
                fr_gate_effective,
            ) = gather_train_traces_with_gate(
                model=model,
                paver=paver,
                controller_vectors=controller,
                gate=gate,
                examples=fr_batch,
                pos_floor=args.stage2_pos_floor,
                neg_cap=args.stage2_neg_cap,
            )
            (
                fr_base_logits,
                fr_base_states,
                fr_pos_logits,
                fr_pos_states,
                fr_neg_logits,
                fr_neg_states,
            ) = collect_fr_references(model=model, paver=paver, examples=fr_batch)

            fr_targets = torch.stack([ex.anchor_ids for ex in fr_batch], dim=0)
            fr_negative_ids = torch.stack([ex.negative_ids for ex in fr_batch], dim=0)

            fr_ce = step_weighted_ce(fr_train_logits, fr_targets, step_weights)
            fr_kl = step_weighted_kl(fr_train_logits, fr_pos_logits, step_weights)
            fr_token_margin = weighted_token_margin_loss(
                fr_train_logits,
                fr_targets,
                fr_negative_ids,
                step_weights,
                margin=args.token_margin,
            )
            fr_traj = fr_trajectory_terms(
                scorer=scorer,
                train_logits=fr_train_logits,
                train_states=fr_train_states,
                pos_logits=fr_pos_logits,
                pos_states=fr_pos_states,
                base_logits=fr_base_logits,
                base_states=fr_base_states,
                neg_logits=fr_neg_logits,
                neg_states=fr_neg_states,
                step_weights=step_weights,
                target_fraction=loss_config.fr_improvement_target,
                floor=loss_config.fr_baseline_floor,
                sep_margin=args.traj_sep_margin,
                sep_scale=args.traj_sep_scale,
                mono_tol=args.traj_mono_tol,
                eps=loss_config.eps,
            )
            fr_gate_bce = F.binary_cross_entropy_with_logits(fr_gate_logits, fr_gate_targets)

            (
                en_train_logits,
                en_train_states,
                en_gate_logits,
                en_gate_probs,
                en_gate_targets,
                en_gate_effective,
            ) = gather_train_traces_with_gate(
                model=model,
                paver=paver,
                controller_vectors=controller,
                gate=gate,
                examples=en_batch,
                pos_floor=args.stage2_pos_floor,
                neg_cap=args.stage2_neg_cap,
            )
            en_base_logits, en_base_states = collect_en_references(model=model, paver=paver, examples=en_batch)
            en_kl = step_weighted_kl(en_train_logits, en_base_logits, step_weights)
            en_traj = en_trajectory_terms(
                scorer=scorer,
                train_logits=en_train_logits,
                train_states=en_train_states,
                base_logits=en_base_logits,
                base_states=en_base_states,
                step_weights=step_weights,
                tolerance=args.en_preserve_tolerance,
                scale=args.en_preserve_scale,
                mono_tol=args.traj_mono_tol,
            )
            en_gate_bce = F.binary_cross_entropy_with_logits(en_gate_logits, en_gate_targets)
            gate_rank = gate_pair_margin_loss(fr_gate_logits, en_gate_logits, args.gate_rank_margin)
            gate_sat = gate_saturation_penalty(torch.cat([fr_gate_probs, en_gate_probs], dim=0))
            gate_reg = gate.regularizer(
                residual_l2=args.gate_residual_l2,
                layer_logit_l2=args.gate_layer_l2,
                sharpness_center=args.gate_sharpness,
                sharpness_l2=args.gate_sharpness_l2,
            )
            reg = paving_regularizer(
                vectors=raw_vectors,
                l2=loss_config.l2,
                layer_smooth=loss_config.layer_smooth,
                time_smooth=loss_config.time_smooth,
            )

            total = (
                loss_config.ce_weight * fr_ce
                + loss_config.fr_kl_weight * fr_kl
                + args.fr_token_margin_weight * fr_token_margin
                + args.fr_traj_close_weight * fr_traj["close"]
                + args.fr_traj_sep_weight * fr_traj["sep"]
                + args.fr_traj_mono_weight * fr_traj["mono"]
                + loss_config.nonfr_kl_weight * en_kl
                + args.en_traj_preserve_weight * en_traj["preserve"]
                + args.en_traj_mono_weight * en_traj["mono"]
                + loss_config.gate_bce_weight * (fr_gate_bce + en_gate_bce)
                + args.gate_rank_weight * gate_rank
                + args.gate_sat_weight * gate_sat
                + args.gate_reg_weight * gate_reg
                + reg
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_([raw_vectors] + list(gate.parameters()), max_norm=1.0)
            opt_stage2.step()

            epoch_records.append(
                {
                    "total": float(total.detach().cpu()),
                    "fr_ce": float(fr_ce.detach().cpu()),
                    "fr_kl": float(fr_kl.detach().cpu()),
                    "fr_tok_margin": float(fr_token_margin.detach().cpu()),
                    "fr_traj_close": float(fr_traj["close"].detach().cpu()),
                    "fr_traj_sep": float(fr_traj["sep"].detach().cpu()),
                    "fr_traj_mono": float(fr_traj["mono"].detach().cpu()),
                    "en_kl": float(en_kl.detach().cpu()),
                    "en_traj_preserve": float(en_traj["preserve"].detach().cpu()),
                    "en_traj_mono": float(en_traj["mono"].detach().cpu()),
                    "gate_rank": float(gate_rank.detach().cpu()),
                    "gate_sat": float(gate_sat.detach().cpu()),
                    "fr_gate_raw": float(fr_gate_probs.detach().mean().cpu()),
                    "en_gate_raw": float(en_gate_probs.detach().mean().cpu()),
                    "eng_minus_fr": english_fraction(fr_train_logits.detach(), fr_targets, fr_negative_ids, args.metric_steps),
                }
            )

        avg = average_dicts(epoch_records)
        gstats = binary_gate_stats(gate, fr_examples, en_examples)
        print(
            f"[stage2 epoch {epoch:02d}] "
            f"total={avg.get('total', 0.0):.4f} "
            f"fr_ce={avg.get('fr_ce', 0.0):.4f} "
            f"fr_kl={avg.get('fr_kl', 0.0):.4f} "
            f"fr_tok_margin={avg.get('fr_tok_margin', 0.0):.4f} "
            f"fr_traj_close={avg.get('fr_traj_close', 0.0):.4f} "
            f"fr_traj_sep={avg.get('fr_traj_sep', 0.0):.4f} "
            f"fr_traj_mono={avg.get('fr_traj_mono', 0.0):.4f} "
            f"en_kl={avg.get('en_kl', 0.0):.4f} "
            f"en_traj_preserve={avg.get('en_traj_preserve', 0.0):.4f} "
            f"gate_rank={avg.get('gate_rank', 0.0):.4f} "
            f"eng_minus_fr={avg.get('eng_minus_fr', 0.0):.4f} "
            f"fr_gate_mean={gstats['fr_gate_mean']:.3f} "
            f"en_gate_mean={gstats['en_gate_mean']:.3f} "
            f"thr_acc={gstats['threshold_acc']:.3f}"
        )
        if args.preview_fr_prompt or args.preview_en_prompt:
            controller_eval = current_controller().detach()
            fr_raw, fr_strength, fr_text = decode_with_binary_gate(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                gate=gate,
                controller_vectors=controller_eval,
                device=device,
                user_prompt=preview_fr,
                max_new_tokens=args.max_new_tokens,
                strength=args.strength,
                temperature=args.temperature,
                top_p=args.top_p,
                hard_gate=args.preview_hard_gate,
            )
            en_raw, en_strength, en_text = decode_with_binary_gate(
                model=model,
                tokenizer=tokenizer,
                paver=paver,
                gate=gate,
                controller_vectors=controller_eval,
                device=device,
                user_prompt=preview_en,
                max_new_tokens=args.max_new_tokens,
                strength=args.strength,
                temperature=args.temperature,
                top_p=args.top_p,
                hard_gate=args.preview_hard_gate,
            )
            print(f"  stage2_fr_gate_raw={fr_raw:.3f} strength={fr_strength:.3f} preview={fr_text[:220]!r}")
            print(f"  stage2_en_gate_raw={en_raw:.3f} strength={en_strength:.3f} preview={en_text[:220]!r}")

    # -------------------- Stage 3: gate-only calibration --------------------
    controller_frozen = current_controller().detach()
    if args.stage3_epochs > 0:
        fr_util_w, en_util_w, util_stats = compute_gate_utility_weights(
            model=model,
            paver=paver,
            scorer=scorer,
            controller_vectors=controller_frozen,
            fr_examples=fr_examples,
            en_examples=en_examples,
            step_weights=step_weights,
            sep_reward=args.calib_sep_reward,
            utility_scale=args.calib_utility_scale,
        )
        print(
            "[stage3 utility] "
            f"fr_benefit_mean={util_stats['fr_benefit_mean']:.4f} "
            f"fr_benefit_max={util_stats['fr_benefit_max']:.4f} "
            f"en_harm_mean={util_stats['en_harm_mean']:.4f} "
            f"en_harm_max={util_stats['en_harm_max']:.4f}"
        )

        fr_boundary_all = torch.stack([ex.boundary_states for ex in fr_examples], dim=0)
        en_boundary_all = torch.stack([ex.boundary_states for ex in en_examples], dim=0)
        fr_targets_all = torch.ones(len(fr_examples), device=device, dtype=torch.float32)
        en_targets_all = torch.zeros(len(en_examples), device=device, dtype=torch.float32)

        for epoch in range(1, args.stage3_epochs + 1):
            opt_stage3.zero_grad(set_to_none=True)
            fr_logits, fr_probs = gate(fr_boundary_all)
            en_logits, en_probs = gate(en_boundary_all)
            fr_bce = F.binary_cross_entropy_with_logits(fr_logits, fr_targets_all, reduction="none")
            en_bce = F.binary_cross_entropy_with_logits(en_logits, en_targets_all, reduction="none")
            fr_bce = (fr_bce * fr_util_w).sum() / fr_util_w.sum().clamp_min(1e-6)
            en_bce = (en_bce * en_util_w).sum() / en_util_w.sum().clamp_min(1e-6)
            gate_rank = gate_pair_margin_loss(fr_logits, en_logits, args.calib_gate_margin, fr_util_w, en_util_w)
            gate_sat = gate_saturation_penalty(torch.cat([fr_probs, en_probs], dim=0))
            gate_reg = gate.regularizer(
                residual_l2=args.gate_residual_l2,
                layer_logit_l2=args.gate_layer_l2,
                sharpness_center=args.gate_sharpness,
                sharpness_l2=args.gate_sharpness_l2,
            )
            total = fr_bce + en_bce + args.calib_gate_rank_weight * gate_rank + args.calib_gate_sat_weight * gate_sat + args.gate_reg_weight * gate_reg
            total.backward()
            torch.nn.utils.clip_grad_norm_(list(gate.parameters()), max_norm=1.0)
            opt_stage3.step()

            with torch.no_grad():
                fr_raw = gate.raw_score(fr_boundary_all)
                en_raw = gate.raw_score(en_boundary_all)
                tau, wacc = choose_weighted_threshold(fr_raw, en_raw, fr_util_w, en_util_w)
                pred_acc = torch.cat([(fr_raw > tau).float(), (en_raw <= tau).float()]).mean().item()
            print(
                f"[stage3 epoch {epoch:02d}] "
                f"total={float(total.detach().cpu()):.4f} "
                f"fr_bce={float(fr_bce.detach().cpu()):.4f} "
                f"en_bce={float(en_bce.detach().cpu()):.4f} "
                f"gate_rank={float(gate_rank.detach().cpu()):.4f} "
                f"gate_sat={float(gate_sat.detach().cpu()):.4f} "
                f"tau={tau:.4f} "
                f"weighted_acc={wacc:.4f} "
                f"raw_acc={pred_acc:.4f}"
            )

        # Final calibration by folding the weighted threshold into the learned bias.
        with torch.no_grad():
            fr_raw = gate.raw_score(fr_boundary_all)
            en_raw = gate.raw_score(en_boundary_all)
            tau, wacc = choose_weighted_threshold(fr_raw, en_raw, fr_util_w, en_util_w)
            gate.bias.sub_(torch.tensor(tau, device=gate.bias.device, dtype=gate.bias.dtype))
            print(f"[stage3 calibrate] folded_threshold={tau:.4f} weighted_acc={wacc:.4f}")

    final_controller = controller_frozen
    final_gstats = binary_gate_stats(gate, fr_examples, en_examples)
    print(
        "[final gate] "
        f"fr_gate_mean={final_gstats['fr_gate_mean']:.3f} "
        f"en_gate_mean={final_gstats['en_gate_mean']:.3f} "
        f"fr_raw_mean={final_gstats['fr_raw_mean']:.3f} "
        f"en_raw_mean={final_gstats['en_raw_mean']:.3f} "
        f"thr_acc={final_gstats['threshold_acc']:.3f}"
    )

    if args.save_path:
        save_binary_state(
            args.save_path,
            BinaryTrainState(
                raw_vectors=raw_vectors.detach().cpu(),
                layer_indices=layer_indices,
                gate_state_dict={k: v.detach().cpu() for k, v in gate.state_dict().items()},
                gate_base_direction=gate.base_direction.detach().cpu(),
                gate_threshold=0.0,
                gate_residual_scale=args.gate_residual_scale,
                gate_sharpness_init=args.gate_sharpness,
                learn_direction_residual=args.learn_direction_residual,
                model_id=args.model_id,
                steer_steps=args.steer_steps,
                layer_ramp_start=args.layer_ramp_start,
                layer_ramp_end=args.layer_ramp_end,
                time_decay_tau=args.time_decay_tau,
                post_smooth_layer=args.post_smooth_layer,
                post_smooth_time=args.post_smooth_time,
                loss_config=asdict(loss_config),
                projection_config=asdict(proj_config),
            ),
        )

    if args.preview_fr_prompt or args.preview_en_prompt:
        print("\n===== FINAL PREVIEWS =====")
        fr_raw, fr_strength, fr_text = decode_with_binary_gate(
            model=model,
            tokenizer=tokenizer,
            paver=paver,
            gate=gate,
            controller_vectors=final_controller,
            device=device,
            user_prompt=preview_fr,
            max_new_tokens=args.max_new_tokens,
            strength=args.strength,
            temperature=args.temperature,
            top_p=args.top_p,
            hard_gate=args.preview_hard_gate,
        )
        en_raw, en_strength, en_text = decode_with_binary_gate(
            model=model,
            tokenizer=tokenizer,
            paver=paver,
            gate=gate,
            controller_vectors=final_controller,
            device=device,
            user_prompt=preview_en,
            max_new_tokens=args.max_new_tokens,
            strength=args.strength,
            temperature=args.temperature,
            top_p=args.top_p,
            hard_gate=args.preview_hard_gate,
        )
        print(f"[FR raw={fr_raw:.4f} strength={fr_strength:.4f}]\n{fr_text}")
        print(f"\n[EN raw={en_raw:.4f} strength={en_strength:.4f}]\n{en_text}")


@torch.no_grad()
def infer_binary_refined(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_binary_state(args.load_path, device)

    model_id = state.get("model_id", args.model_id or DEFAULT_MODEL_ID)
    model, tokenizer = load_model_and_tokenizer(model_id, device)
    paver = ResidualPaver(
        model=model,
        layer_indices=[int(x) for x in state["layer_indices"]],
        intervention_positions=str(state.get("intervention_positions", "last_token")),
    )
    gate_config = state.get("gate_config") or {}
    gate = AnchoredBinaryGate(
        init_direction=state["gate_base_direction"],
        init_threshold=0.0,
        sharpness=float(state["gate_sharpness_init"]),
        learn_direction_residual=bool(state.get("learn_direction_residual", False)),
        residual_scale=float(state.get("gate_residual_scale", 0.05)),
        gate_type=str(gate_config.get("gate_type", "anchored_linear")),
        hidden_dim=int(gate_config.get("hidden_dim", 64)),
    ).to(device)
    gate.load_state_dict(state["gate_state_dict"])
    gate.eval()

    controller = finalize_paving_vectors(
        state["raw_vectors"].to(device),
        layer_start=float(state["layer_ramp_start"]),
        layer_end=float(state["layer_ramp_end"]),
        time_decay_tau=float(state["time_decay_tau"]),
        smooth_layer=float(state["post_smooth_layer"]),
        smooth_time=float(state["post_smooth_time"]),
    )

    gate_raw, gate_strength, text = decode_with_binary_gate(
        model=model,
        tokenizer=tokenizer,
        paver=paver,
        gate=gate,
        controller_vectors=controller,
        device=device,
        user_prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        strength=args.strength,
        temperature=args.temperature,
        top_p=args.top_p,
        hard_gate=args.hard_gate,
    )
    print(text)
    print(f"\n[gate_raw={gate_raw:.4f} gate_strength={gate_strength:.4f}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p_train.add_argument("--layers", type=str, default="-4,-3,-2,-1")
    p_train.add_argument("--steer_steps", type=int, default=24)
    p_train.add_argument("--stage1_epochs", type=int, default=12)
    p_train.add_argument("--stage2_epochs", type=int, default=16)
    p_train.add_argument("--stage3_epochs", type=int, default=12)
    p_train.add_argument("--fr_batch_size", type=int, default=4)
    p_train.add_argument("--en_batch_size", type=int, default=4)
    p_train.add_argument("--build_batch_size", type=int, default=0, help="Prompt-building batch size. Use 0 to auto-pick from train batch sizes.")
    p_train.add_argument("--build_progress_every", type=int, default=64, help="Print prompt-building progress every N records. Use 0 to only print the final count.")
    p_train.add_argument("--lr_stage1", type=float, default=0.10)
    p_train.add_argument("--lr_stage2", type=float, default=0.04)
    p_train.add_argument("--lr_stage3", type=float, default=0.03)

    p_train.add_argument("--ce_weight", type=float, default=1.0)
    p_train.add_argument("--fr_kl_weight", type=float, default=0.50)
    p_train.add_argument("--en_kl_weight", type=float, default=1.10)
    p_train.add_argument("--fr_token_margin_weight", type=float, default=0.90)
    p_train.add_argument("--fr_traj_weight", type=float, default=1.0)  # saved only
    p_train.add_argument("--en_traj_weight", type=float, default=1.0)  # saved only
    p_train.add_argument("--fr_traj_close_weight", type=float, default=0.90)
    p_train.add_argument("--fr_traj_sep_weight", type=float, default=0.85)
    p_train.add_argument("--fr_traj_mono_weight", type=float, default=0.25)
    p_train.add_argument("--en_traj_preserve_weight", type=float, default=1.35)
    p_train.add_argument("--en_traj_mono_weight", type=float, default=0.20)

    p_train.add_argument("--traj_prob_weight", type=float, default=1.0)
    p_train.add_argument("--traj_logit_weight", type=float, default=0.35)
    p_train.add_argument("--traj_state_weight", type=float, default=0.45)
    p_train.add_argument("--fr_improvement_target", type=float, default=0.35)
    p_train.add_argument("--fr_baseline_floor", type=float, default=0.05)
    p_train.add_argument("--en_preserve_tolerance", type=float, default=0.04)
    p_train.add_argument("--en_preserve_scale", type=float, default=0.22)
    p_train.add_argument("--traj_sep_margin", type=float, default=0.12)
    p_train.add_argument("--traj_sep_scale", type=float, default=0.20)
    p_train.add_argument("--traj_mono_tol", type=float, default=0.02)

    p_train.add_argument("--prefix_focus_steps", type=int, default=8)
    p_train.add_argument("--prefix_focus_boost", type=float, default=2.5)
    p_train.add_argument("--prefix_decay_tau", type=float, default=8.0)
    p_train.add_argument("--token_margin", type=float, default=1.0)
    p_train.add_argument("--metric_steps", type=int, default=8)

    p_train.add_argument("--gate_bce_weight", type=float, default=0.75)
    p_train.add_argument("--gate_rank_weight", type=float, default=0.80)
    p_train.add_argument("--gate_rank_margin", type=float, default=2.0)
    p_train.add_argument("--gate_sat_weight", type=float, default=0.15)
    p_train.add_argument("--gate_sharpness", type=float, default=12.0)
    p_train.add_argument("--learn_direction_residual", action="store_true")
    p_train.add_argument("--gate_residual_scale", type=float, default=0.05)
    p_train.add_argument("--gate_residual_l2", type=float, default=1e-3)
    p_train.add_argument("--gate_layer_l2", type=float, default=1e-3)
    p_train.add_argument("--gate_sharpness_l2", type=float, default=1e-3)
    p_train.add_argument("--gate_reg_weight", type=float, default=1.0)
    p_train.add_argument("--stage1_fr_gate", type=float, default=1.0)
    p_train.add_argument("--stage2_pos_floor", type=float, default=0.70)
    p_train.add_argument("--stage2_neg_cap", type=float, default=0.10)

    p_train.add_argument("--calib_sep_reward", type=float, default=0.50)
    p_train.add_argument("--calib_utility_scale", type=float, default=1.50)
    p_train.add_argument("--calib_gate_margin", type=float, default=2.0)
    p_train.add_argument("--calib_gate_rank_weight", type=float, default=0.90)
    p_train.add_argument("--calib_gate_sat_weight", type=float, default=0.30)

    p_train.add_argument("--vocab_proj_dim", type=int, default=64)
    p_train.add_argument("--state_proj_dim", type=int, default=64)
    p_train.add_argument("--l2", type=float, default=1e-4)
    p_train.add_argument("--layer_smooth", type=float, default=5e-4)
    p_train.add_argument("--time_smooth", type=float, default=5e-4)
    p_train.add_argument("--layer_ramp_start", type=float, default=0.65)
    p_train.add_argument("--layer_ramp_end", type=float, default=1.10)
    p_train.add_argument("--time_decay_tau", type=float, default=12.0)
    p_train.add_argument("--post_smooth_layer", type=float, default=0.15)
    p_train.add_argument("--post_smooth_time", type=float, default=0.10)

    p_train.add_argument("--preview_fr_prompt", type=str, default=PARALLEL_LANGUAGE_PROMPTS[0]["fr"])
    p_train.add_argument("--preview_en_prompt", type=str, default=PARALLEL_LANGUAGE_PROMPTS[0]["en"])
    p_train.add_argument("--preview_hard_gate", action="store_true")
    p_train.add_argument("--max_new_tokens", type=int, default=128)
    p_train.add_argument("--strength", type=float, default=1.0)
    p_train.add_argument("--temperature", type=float, default=0.0)
    p_train.add_argument("--top_p", type=float, default=0.95)
    p_train.add_argument("--save_path", type=str, default="gemma3_fr_en_binary_refined.pt")
    p_train.add_argument("--seed", type=int, default=0)

    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--load_path", type=str, required=True)
    p_infer.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p_infer.add_argument("--prompt", type=str, required=True)
    p_infer.add_argument("--hard_gate", action="store_true")
    p_infer.add_argument("--max_new_tokens", type=int, default=128)
    p_infer.add_argument("--strength", type=float, default=1.0)
    p_infer.add_argument("--temperature", type=float, default=0.0)
    p_infer.add_argument("--top_p", type=float, default=0.95)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train_binary_refined(args)
    elif args.command == "infer":
        infer_binary_refined(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
