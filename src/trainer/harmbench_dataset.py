from __future__ import annotations

import csv
import io
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HARMBENCH_SPLIT_URLS: Dict[str, str] = {
    "val": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_val.csv",
    "test": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv",
}
SALADBENCH_DEFAULT_REPO = "walledai/SaladBench"
SALADBENCH_PROMPTS_CONFIG = "prompts"
SALADBENCH_SPLITS: Tuple[str, ...] = ("base", "attackEnhanced", "defenseEnhanced")
ALPACA_DEFAULT_SOURCE = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
DEFAULT_HARMBENCH_COMBINE_SEPARATOR = " and also "
HARMBENCH_CACHE_FIELDNAMES: Tuple[str, ...] = (
    "sample_id",
    "bucket",
    "fr",
    "en",
    "behavior_id",
    "functional_category",
    "semantic_category",
    "tags",
)


def _snake(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "general"


def format_harmbench_prompt(row: Mapping[str, str]) -> str:
    behavior = str(row.get("Behavior", "")).strip()
    context = str(row.get("ContextString", "")).strip()
    if context:
        return f"Context: {context}\n\nRequest: {behavior}"
    return behavior


def format_alpaca_prompt(row: Mapping[str, str]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if input_text:
        return f"Instruction: {instruction}\n\nInput: {input_text}\n\nResponse:"
    return f"Instruction: {instruction}\n\nResponse:"


def format_alpaca_match_text(row: Mapping[str, str]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def format_saladbench_prompt(row: Mapping[str, Any]) -> str:
    return str(row.get("prompt", "")).strip()


def default_harmbench_source(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in HARMBENCH_SPLIT_URLS:
        raise ValueError(f"Unsupported HarmBench split: {split!r}")
    return HARMBENCH_SPLIT_URLS[normalized]


def _read_text_source(source: str) -> str:
    if source.startswith("https://") or source.startswith("http://"):
        with urllib.request.urlopen(source) as response:
            data = response.read()
        return data.decode("utf-8")

    resolved = Path(source).expanduser()
    return resolved.read_text(encoding="utf-8")


def load_harmbench_rows(source: str) -> List[Dict[str, str]]:
    text = _read_text_source(source)
    reader = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, str]] = []
    for row in reader:
        cleaned = {
            str(key).strip(): (value or "").strip()
            for key, value in row.items()
            if key is not None
        }
        if cleaned:
            rows.append(cleaned)
    if not rows:
        raise ValueError(f"No HarmBench rows loaded from {source}")
    return rows


def load_alpaca_rows(source: Optional[str] = None) -> List[Dict[str, str]]:
    resolved_source = source or ALPACA_DEFAULT_SOURCE
    text = _read_text_source(resolved_source)
    obj = json.loads(text)
    if not isinstance(obj, list):
        raise ValueError(f"Expected Alpaca JSON list from {resolved_source}")
    rows: List[Dict[str, str]] = []
    for row in obj:
        if not isinstance(row, Mapping):
            continue
        cleaned = {
            str(key).strip(): ("" if value is None else str(value).strip())
            for key, value in row.items()
            if key is not None
        }
        if cleaned.get("instruction"):
            rows.append(cleaned)
    if not rows:
        raise ValueError(f"No Alpaca rows loaded from {resolved_source}")
    return rows


def _normalize_saladbench_categories(raw_categories: Any) -> List[str]:
    if raw_categories is None:
        return []
    if isinstance(raw_categories, str):
        value = raw_categories.strip()
        return [value] if value else []

    categories: List[str] = []
    for item in raw_categories:
        value = str(item).strip()
        if value:
            categories.append(value)
    return categories


def normalize_saladbench_bucket(
    categories: Sequence[str],
    *,
    level: str = "top",
) -> str:
    items = [str(item).strip() for item in categories if str(item).strip()]
    if not items:
        return "general"

    level_key = str(level).strip().lower() or "top"
    index_map = {"top": 0, "mid": 1, "leaf": 2}
    if level_key not in index_map:
        raise ValueError(f"Unsupported SaladBench bucket level: {level!r}")
    idx = min(index_map[level_key], len(items) - 1)
    target = items[idx]
    match = re.match(r"^(o\d+):\s*(.+)$", target, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1).lower()}_{_snake(match.group(2))}"
    return _snake(target)


def load_saladbench_rows(
    source: Optional[str] = None,
    *,
    split: str = "base",
) -> List[Dict[str, Any]]:
    normalized_split = str(split).strip() or "base"
    if normalized_split not in SALADBENCH_SPLITS:
        raise ValueError(f"Unsupported SaladBench split: {split!r}")

    if source is None:
        from datasets import load_dataset

        ds = load_dataset(SALADBENCH_DEFAULT_REPO, SALADBENCH_PROMPTS_CONFIG, split=normalized_split)
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(ds):
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            rows.append(
                {
                    "sample_id": f"saladbench_{normalized_split}_{index}",
                    "prompt": prompt,
                    "categories": _normalize_saladbench_categories(row.get("categories")),
                    "source": str(row.get("source", "")).strip(),
                }
            )
        if not rows:
            raise ValueError(f"No SaladBench rows loaded for split {normalized_split}")
        return rows

    text = _read_text_source(source)
    obj = json.loads(text)
    if isinstance(obj, Mapping):
        if isinstance(obj.get("rows"), list):
            obj = obj["rows"]
        elif isinstance(obj.get("data"), list):
            obj = obj["data"]
        else:
            obj = [obj]
    if not isinstance(obj, list):
        raise ValueError(f"Expected SaladBench JSON list from {source}")

    rows = []
    for index, row in enumerate(obj):
        if not isinstance(row, Mapping):
            continue
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        rows.append(
            {
                "sample_id": str(row.get("sample_id", "")).strip() or f"saladbench_{normalized_split}_{index}",
                "prompt": prompt,
                "categories": _normalize_saladbench_categories(row.get("categories")),
                "source": str(row.get("source", "")).strip(),
            }
        )
    if not rows:
        raise ValueError(f"No SaladBench rows loaded from {source}")
    return rows


def filter_harmbench_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    exclude_copyright: bool = True,
    exclude_contextual: bool = True,
) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []
    for row in rows:
        functional_category = str(row.get("FunctionalCategory", "")).strip().lower()
        semantic_category = str(row.get("SemanticCategory", "")).strip().lower()
        if exclude_copyright and (functional_category == "copyright" or semantic_category == "copyright"):
            continue
        if exclude_contextual and functional_category == "contextual":
            continue
        filtered.append({str(key): str(value) for key, value in row.items()})
    if not filtered:
        raise ValueError("All HarmBench rows were filtered out.")
    return filtered


def load_cached_parallel_records(
    cache_path: str | Path,
    *,
    max_records: Optional[int] = None,
) -> List[Dict[str, str]]:
    resolved = Path(cache_path).expanduser()
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Cached HarmBench prompt file is missing a header row: {resolved}")
        required = {"fr", "en", "bucket"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Cached HarmBench prompt file must include {sorted(required)}: {resolved}")

        records: List[Dict[str, str]] = []
        for row in reader:
            cleaned = {
                str(key).strip(): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            cleaned = {key: value for key, value in cleaned.items() if value}
            if not cleaned:
                continue
            records.append(cleaned)
            if max_records is not None and len(records) >= int(max_records):
                break
    if not records:
        raise ValueError(f"No cached HarmBench prompt records were loaded from {resolved}")
    return records


def write_parallel_records_cache(
    cache_path: str | Path,
    records: Sequence[Mapping[str, str]],
) -> Path:
    resolved = Path(cache_path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HARMBENCH_CACHE_FIELDNAMES), delimiter="\t")
        writer.writeheader()
        for row in records:
            writer.writerow({field: str(row.get(field, "") or "") for field in HARMBENCH_CACHE_FIELDNAMES})
    return resolved


def build_harmbench_parallel_records(
    rows: Sequence[Mapping[str, str]],
    *,
    translator: Callable[[str], str],
    bucket_field: str = "SemanticCategory",
    max_records: Optional[int] = None,
) -> List[Dict[str, str]]:
    if max_records is not None and int(max_records) <= 0:
        raise ValueError("--max_prompt_records must be positive when provided.")

    normalized_bucket_field = str(bucket_field).strip() or "SemanticCategory"
    records: List[Dict[str, str]] = []
    for index, row in enumerate(rows):
        en_prompt = format_harmbench_prompt(row)
        fr_prompt = translator(en_prompt).strip()
        if not fr_prompt:
            raise ValueError(f"Translator returned an empty French prompt for HarmBench row {index}.")
        bucket = str(row.get(normalized_bucket_field, "")).strip() or "general"
        behavior_id = str(row.get("BehaviorID", "")).strip() or f"harmbench_{index}"
        records.append(
            {
                "sample_id": behavior_id,
                "bucket": bucket,
                "fr": fr_prompt,
                "en": en_prompt.strip(),
                "behavior_id": behavior_id,
                "functional_category": str(row.get("FunctionalCategory", "")).strip(),
                "semantic_category": str(row.get("SemanticCategory", "")).strip(),
                "tags": str(row.get("Tags", "")).strip(),
            }
        )
        if max_records is not None and len(records) >= int(max_records):
            break
    if not records:
        raise ValueError("No HarmBench prompt records were built.")
    return records


def split_alpaca_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    train_count: int,
    test_count: int,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    shuffled = [{str(key): str(value) for key, value in row.items()} for row in rows]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    if train_count < 0 or test_count < 0:
        raise ValueError("train_count and test_count must be non-negative.")
    total_needed = int(train_count) + int(test_count)
    if total_needed > len(shuffled):
        raise ValueError(
            f"Requested {total_needed} Alpaca prompts, but only {len(shuffled)} rows are available."
        )
    train_rows = shuffled[: int(train_count)]
    test_rows = shuffled[int(train_count) : int(train_count) + int(test_count)]
    return train_rows, test_rows


def build_harmbench_alpaca_prompt_records(
    harmbench_rows: Sequence[Mapping[str, str]],
    alpaca_rows: Sequence[Mapping[str, str]],
    *,
    bucket_field: str = "SemanticCategory",
) -> List[Dict[str, str]]:
    pair_count = min(len(harmbench_rows), len(alpaca_rows))
    if pair_count <= 0:
        raise ValueError("No HarmBench/Alpaca prompt pairs were created.")

    normalized_bucket_field = str(bucket_field).strip() or "SemanticCategory"
    records: List[Dict[str, str]] = []
    for index in range(pair_count):
        hb_row = harmbench_rows[index]
        alpaca_row = alpaca_rows[index]
        behavior_id = str(hb_row.get("BehaviorID", "")).strip() or f"harmbench_{index}"
        bucket = str(hb_row.get(normalized_bucket_field, "")).strip() or "general"
        records.append(
            {
                "sample_id": behavior_id,
                "bucket": bucket,
                "fr": format_harmbench_prompt(hb_row),
                "en": format_alpaca_prompt(alpaca_row),
                "behavior_id": behavior_id,
                "functional_category": str(hb_row.get("FunctionalCategory", "")).strip(),
                "semantic_category": str(hb_row.get("SemanticCategory", "")).strip(),
                "tags": str(hb_row.get("Tags", "")).strip(),
            }
        )
    return records


def build_harmbench_two_questions_prompt_records(
    harmbench_rows: Sequence[Mapping[str, str]],
    alpaca_rows: Sequence[Mapping[str, str]],
    *,
    bucket_field: str = "SemanticCategory",
    separator: str = DEFAULT_HARMBENCH_COMBINE_SEPARATOR,
) -> List[Dict[str, str]]:
    pair_count = min(len(harmbench_rows), len(alpaca_rows))
    if pair_count <= 0:
        raise ValueError("No HarmBench/Alpaca prompt pairs were created.")

    normalized_bucket_field = str(bucket_field).strip() or "SemanticCategory"
    joiner = str(separator)
    records: List[Dict[str, str]] = []
    for index in range(pair_count):
        hb_row = harmbench_rows[index]
        alpaca_row = alpaca_rows[index]
        behavior_id = str(hb_row.get("BehaviorID", "")).strip() or f"harmbench_{index}"
        bucket = str(hb_row.get(normalized_bucket_field, "")).strip() or "general"
        keep_prompt = format_alpaca_prompt(alpaca_row)
        harmbench_prompt = format_harmbench_prompt(hb_row)
        records.append(
            {
                "sample_id": behavior_id,
                "bucket": bucket,
                "fr": f"{keep_prompt}{joiner}{harmbench_prompt}",
                "en": keep_prompt,
                "behavior_id": behavior_id,
                "functional_category": str(hb_row.get("FunctionalCategory", "")).strip(),
                "semantic_category": str(hb_row.get("SemanticCategory", "")).strip(),
                "tags": str(hb_row.get("Tags", "")).strip(),
            }
        )
    return records


def build_saladbench_alpaca_prompt_records(
    saladbench_rows: Sequence[Mapping[str, Any]],
    alpaca_rows: Sequence[Mapping[str, str]],
    *,
    bucket_level: str = "top",
) -> List[Dict[str, str]]:
    pair_count = min(len(saladbench_rows), len(alpaca_rows))
    if pair_count <= 0:
        raise ValueError("No SaladBench/Alpaca prompt pairs were created.")

    records: List[Dict[str, str]] = []
    for index in range(pair_count):
        saladbench_row = saladbench_rows[index]
        alpaca_row = alpaca_rows[index]
        categories = _normalize_saladbench_categories(saladbench_row.get("categories"))
        records.append(
            {
                "sample_id": str(saladbench_row.get("sample_id", "")).strip() or f"saladbench_{index}",
                "bucket": normalize_saladbench_bucket(categories, level=bucket_level),
                "fr": format_saladbench_prompt(saladbench_row),
                "en": format_alpaca_prompt(alpaca_row),
                "top_category": categories[0] if len(categories) > 0 else "",
                "mid_category": categories[1] if len(categories) > 1 else "",
                "leaf_category": categories[2] if len(categories) > 2 else "",
                "source": str(saladbench_row.get("source", "")).strip(),
            }
        )
    return records


def build_saladbench_two_questions_prompt_records(
    saladbench_rows: Sequence[Mapping[str, Any]],
    alpaca_rows: Sequence[Mapping[str, str]],
    *,
    bucket_level: str = "top",
    separator: str = DEFAULT_HARMBENCH_COMBINE_SEPARATOR,
) -> List[Dict[str, str]]:
    pair_count = min(len(saladbench_rows), len(alpaca_rows))
    if pair_count <= 0:
        raise ValueError("No SaladBench/Alpaca prompt pairs were created.")

    joiner = str(separator)
    records: List[Dict[str, str]] = []
    for index in range(pair_count):
        saladbench_row = saladbench_rows[index]
        alpaca_row = alpaca_rows[index]
        categories = _normalize_saladbench_categories(saladbench_row.get("categories"))
        keep_prompt = format_alpaca_prompt(alpaca_row)
        saladbench_prompt = format_saladbench_prompt(saladbench_row)
        records.append(
            {
                "sample_id": str(saladbench_row.get("sample_id", "")).strip() or f"saladbench_{index}",
                "bucket": normalize_saladbench_bucket(categories, level=bucket_level),
                "fr": f"{keep_prompt}{joiner}{saladbench_prompt}",
                "en": keep_prompt,
                "top_category": categories[0] if len(categories) > 0 else "",
                "mid_category": categories[1] if len(categories) > 1 else "",
                "leaf_category": categories[2] if len(categories) > 2 else "",
                "source": str(saladbench_row.get("source", "")).strip(),
            }
        )
    return records


def select_top_unique_semantic_matches(
    candidate_pairs: Sequence[Tuple[float, int, int]],
    *,
    pair_limit: int,
    allow_alpaca_reuse: bool = False,
) -> List[Dict[str, Any]]:
    target = max(int(pair_limit), 0)
    if target <= 0:
        return []

    normalized_candidates: List[Tuple[float, int, int]] = []
    for score, saladbench_index, alpaca_index in candidate_pairs:
        normalized_candidates.append((float(score), int(saladbench_index), int(alpaca_index)))
    normalized_candidates.sort(key=lambda item: item[0], reverse=True)

    used_saladbench: set[int] = set()
    used_alpaca: set[int] = set()
    matches: List[Dict[str, Any]] = []
    for score, saladbench_index, alpaca_index in normalized_candidates:
        if saladbench_index in used_saladbench:
            continue
        if not allow_alpaca_reuse and alpaca_index in used_alpaca:
            continue

        used_saladbench.add(saladbench_index)
        if not allow_alpaca_reuse:
            used_alpaca.add(alpaca_index)
        matches.append(
            {
                "pair_rank": len(matches) + 1,
                "score": float(score),
                "saladbench_index": int(saladbench_index),
                "alpaca_index": int(alpaca_index),
            }
        )
        if len(matches) >= target:
            break
    return matches


def allocate_balanced_bucket_counts(
    bucket_sizes: Mapping[str, int],
    *,
    pair_limit: int,
) -> Dict[str, int]:
    positive_items = {
        str(bucket): max(int(size), 0)
        for bucket, size in bucket_sizes.items()
        if int(size) > 0
    }
    target = max(0, min(int(pair_limit), sum(positive_items.values())))
    if target <= 0:
        return {str(bucket): 0 for bucket in bucket_sizes}

    ordered_buckets = sorted(positive_items)
    counts = {str(bucket): 0 for bucket in bucket_sizes}
    remaining = target

    base = target // len(ordered_buckets)
    remainder = target % len(ordered_buckets)
    for idx, bucket in enumerate(ordered_buckets):
        counts[bucket] = min(positive_items[bucket], base + (1 if idx < remainder else 0))
        remaining -= counts[bucket]

    while remaining > 0:
        progressed = False
        for bucket in ordered_buckets:
            if counts[bucket] >= positive_items[bucket]:
                continue
            counts[bucket] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
    return counts


def select_balanced_semantic_matches(
    candidate_pairs: Sequence[Tuple[float, int, int]],
    *,
    saladbench_buckets: Sequence[str],
    pair_limit: int,
    allow_alpaca_reuse: bool = False,
) -> List[Dict[str, Any]]:
    if len(saladbench_buckets) <= 0:
        return []

    normalized_candidates: List[Tuple[float, int, int]] = []
    bucket_sizes: Dict[str, int] = {}
    for saladbench_index, bucket in enumerate(saladbench_buckets):
        bucket_key = str(bucket).strip() or "general"
        bucket_sizes[bucket_key] = bucket_sizes.get(bucket_key, 0) + 1
    target_counts = allocate_balanced_bucket_counts(bucket_sizes, pair_limit=pair_limit)

    for score, saladbench_index, alpaca_index in candidate_pairs:
        normalized_candidates.append((float(score), int(saladbench_index), int(alpaca_index)))
    normalized_candidates.sort(key=lambda item: item[0], reverse=True)

    per_bucket_selected: Dict[str, int] = {bucket: 0 for bucket in target_counts}
    selected_saladbench: set[int] = set()
    selected_alpaca: set[int] = set()
    matches: List[Dict[str, Any]] = []

    for score, saladbench_index, alpaca_index in normalized_candidates:
        if saladbench_index < 0 or saladbench_index >= len(saladbench_buckets):
            continue
        bucket = str(saladbench_buckets[saladbench_index]).strip() or "general"
        if per_bucket_selected.get(bucket, 0) >= target_counts.get(bucket, 0):
            continue
        if saladbench_index in selected_saladbench:
            continue
        if not allow_alpaca_reuse and alpaca_index in selected_alpaca:
            continue

        selected_saladbench.add(saladbench_index)
        if not allow_alpaca_reuse:
            selected_alpaca.add(alpaca_index)
        per_bucket_selected[bucket] = per_bucket_selected.get(bucket, 0) + 1
        matches.append(
            {
                "pair_rank": len(matches) + 1,
                "score": float(score),
                "saladbench_index": int(saladbench_index),
                "alpaca_index": int(alpaca_index),
                "bucket": bucket,
            }
        )
        if len(matches) >= min(int(pair_limit), len(saladbench_buckets)):
            break

    return matches


def build_saladbench_semantic_prompt_records(
    saladbench_rows: Sequence[Mapping[str, Any]],
    alpaca_rows: Sequence[Mapping[str, str]],
    matches: Sequence[Mapping[str, Any]],
    *,
    bucket_level: str = "top",
) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for fallback_rank, match in enumerate(matches, start=1):
        saladbench_index = int(match.get("saladbench_index", -1))
        alpaca_index = int(match.get("alpaca_index", -1))
        if saladbench_index < 0 or saladbench_index >= len(saladbench_rows):
            raise IndexError(f"saladbench_index out of range: {saladbench_index}")
        if alpaca_index < 0 or alpaca_index >= len(alpaca_rows):
            raise IndexError(f"alpaca_index out of range: {alpaca_index}")

        saladbench_row = saladbench_rows[saladbench_index]
        alpaca_row = alpaca_rows[alpaca_index]
        categories = _normalize_saladbench_categories(saladbench_row.get("categories"))
        score = float(match.get("score", 0.0))
        pair_rank = int(match.get("pair_rank", fallback_rank))
        records.append(
            {
                "sample_id": str(saladbench_row.get("sample_id", "")).strip() or f"saladbench_{saladbench_index}",
                "bucket": normalize_saladbench_bucket(categories, level=bucket_level),
                "fr": format_saladbench_prompt(saladbench_row),
                "en": format_alpaca_prompt(alpaca_row),
                "match_score": f"{score:.6f}",
                "pair_rank": str(pair_rank),
                "saladbench_index": str(saladbench_index),
                "alpaca_index": str(alpaca_index),
                "alpaca_instruction": str(alpaca_row.get("instruction", "")).strip(),
                "alpaca_input": str(alpaca_row.get("input", "")).strip(),
                "top_category": categories[0] if len(categories) > 0 else "",
                "mid_category": categories[1] if len(categories) > 1 else "",
                "leaf_category": categories[2] if len(categories) > 2 else "",
                "source": str(saladbench_row.get("source", "")).strip(),
            }
        )
    return records


def load_or_build_harmbench_parallel_records(
    *,
    source: str,
    cache_path: Optional[str],
    translator: Callable[[str], str],
    bucket_field: str = "SemanticCategory",
    max_records: Optional[int] = None,
    exclude_copyright: bool = True,
    exclude_contextual: bool = True,
    force_refresh: bool = False,
) -> Tuple[List[Dict[str, str]], str]:
    if cache_path and Path(cache_path).expanduser().exists() and not force_refresh:
        records = load_cached_parallel_records(cache_path, max_records=max_records)
        return records, f"cache:{Path(cache_path).expanduser().resolve()}"

    rows = load_harmbench_rows(source)
    rows = filter_harmbench_rows(
        rows,
        exclude_copyright=exclude_copyright,
        exclude_contextual=exclude_contextual,
    )
    records = build_harmbench_parallel_records(
        rows,
        translator=translator,
        bucket_field=bucket_field,
        max_records=max_records,
    )
    if cache_path:
        resolved_cache = write_parallel_records_cache(cache_path, records)
        return records, f"{source} -> {resolved_cache}"
    return records, source
