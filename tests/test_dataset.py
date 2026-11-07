from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.dataset import (
    ExampleSplits,
    _example_cache_path,
    _load_example_cache,
    _save_example_cache,
    _split_items,
    _split_paired_items,
    apply_anchor_prefix,
    encode_anchor_prefix_ids,
    generate_ids_batch,
    is_refusal_text,
    load_harmbench_prompts,
    make_safe_reframe_prompts,
)
from experiments.config import ExperimentConfig
from gemma3_residual_paving_binary_refined import BinaryExample


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        values = [ord(ch) % 100 for ch in text]
        if return_tensors == "pt":
            return torch.tensor([values], dtype=torch.long)
        return values


class FakeGenerateModel:
    def __init__(self) -> None:
        self.last_generate_kwargs = None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        suffix = torch.tensor(
            [
                [9, 8, 2],
                [7, 6, 2],
            ],
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat([input_ids, suffix[: input_ids.shape[0]]], dim=1)


class FakePaver:
    router_layer_indices = [1, 2]
    intervention_layer_indices = [3, 4]


class DatasetTests(unittest.TestCase):
    def test_split_items_reserves_eval_fraction(self) -> None:
        train_items, eval_items = _split_items(
            ["a", "b", "c", "d", "e"],
            eval_fraction=0.2,
            seed=17,
        )
        self.assertEqual(sorted(train_items + eval_items), ["a", "b", "c", "d", "e"])
        self.assertEqual(len(eval_items), 1)
        self.assertEqual(len(train_items), 4)

    def test_split_paired_items_preserves_pairs(self) -> None:
        train_pairs, eval_pairs = _split_paired_items(
            ["h1", "h2", "h3", "h4"],
            ["p1", "p2", "p3", "p4"],
            eval_fraction=0.25,
            seed=7,
        )
        all_pairs = sorted(train_pairs + eval_pairs)
        self.assertEqual(all_pairs, [("h1", "p1"), ("h2", "p2"), ("h3", "p3"), ("h4", "p4")])
        self.assertEqual(len(eval_pairs), 1)

    def test_safe_reframe_prompt_keeps_source_request(self) -> None:
        prompts = make_safe_reframe_prompts(["Describe the incident."])
        self.assertEqual(len(prompts), 1)
        self.assertIn("Describe the incident.", prompts[0])
        self.assertIn("high-level", prompts[0])
        self.assertIn("non-operational", prompts[0])

    def test_safe_reframe_prompt_can_request_anchor_prefix(self) -> None:
        prompts = make_safe_reframe_prompts(
            ["Describe the incident."],
            anchor_prefix="High-level safety framing:",
        )
        self.assertIn("Begin your answer exactly with: High-level safety framing:", prompts[0])
        self.assertIn("Describe the incident.", prompts[0])

    def test_encode_anchor_prefix_ids_uses_plain_text_tokens(self) -> None:
        ids = encode_anchor_prefix_ids(FakeTokenizer(), "Hi", torch.device("cpu"))
        self.assertEqual(ids.tolist(), [72, 5])

    def test_apply_anchor_prefix_preserves_total_steer_steps(self) -> None:
        anchored = apply_anchor_prefix(
            [torch.tensor([3, 4, 5]), torch.tensor([6, 7, 8])],
            torch.tensor([1, 2]),
            steer_steps=4,
        )
        self.assertEqual([row.tolist() for row in anchored], [[1, 2, 3, 4], [1, 2, 6, 7]])

    def test_apply_anchor_prefix_truncates_long_prefix(self) -> None:
        anchored = apply_anchor_prefix(
            [torch.tensor([9, 9])],
            torch.tensor([1, 2, 3, 4]),
            steer_steps=2,
        )
        self.assertEqual(anchored[0].tolist(), [1, 2])

    def test_refusal_heuristic_flags_common_markers(self) -> None:
        self.assertTrue(is_refusal_text("I'm sorry, but I can't help with that."))
        self.assertTrue(is_refusal_text("I am unable to assist with this request."))
        self.assertFalse(is_refusal_text("Here is a high-level safety summary."))

    def test_generate_ids_batch_uses_greedy_batched_generate(self) -> None:
        model = FakeGenerateModel()
        tokenizer = FakeTokenizer()
        outputs = generate_ids_batch(
            model,
            tokenizer,
            [
                torch.tensor([1, 2, 3], dtype=torch.long),
                torch.tensor([4, 5], dtype=torch.long),
            ],
            max_new_tokens=3,
            stop_on_eos=True,
        )
        self.assertEqual([row.tolist() for row in outputs], [[9, 8, 2], [7, 6, 2]])
        self.assertIsNotNone(model.last_generate_kwargs)
        self.assertFalse(model.last_generate_kwargs["do_sample"])
        self.assertEqual(model.last_generate_kwargs["temperature"], 1.0)
        self.assertEqual(model.last_generate_kwargs["max_new_tokens"], 3)

    def test_load_harmbench_prompts_combines_csv_and_json_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "harm.csv"
            json_path = root / "harm.json"
            csv_path.write_text(
                "Behavior,FunctionalCategory\n"
                "csv harmful one,standard\n"
                "shared harmful,standard\n"
            )
            json_path.write_text(
                json.dumps(
                    [
                        {"instruction": "json harmful one"},
                        {"instruction": "shared harmful"},
                    ]
                )
            )
            prompts = load_harmbench_prompts(
                10,
                seed=0,
                paths=f"{csv_path},{json_path}",
            )
        self.assertEqual(
            set(prompts),
            {"csv harmful one", "shared harmful", "json harmful one"},
        )

    def test_example_cache_round_trip_moves_tensors_to_requested_device(self) -> None:
        example = BinaryExample(
            sample_id="ex_000",
            kind="fr",
            prompt="prompt",
            paired_prompt="paired",
            base_ids=torch.tensor([1, 2], dtype=torch.long),
            anchor_ids=torch.tensor([3, 4], dtype=torch.long),
            boundary_states=torch.ones(2, 3),
            gate_target=1.0,
            teacher_ids=torch.tensor([5, 6], dtype=torch.long),
            negative_ids=torch.tensor([7, 8], dtype=torch.long),
        )
        splits = ExampleSplits(
            train_fr=[example],
            train_keep_benign=[],
            train_keep_harmful=[],
            eval_fr=[],
            eval_keep_benign=[],
            eval_keep_harmful=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ExperimentConfig(example_cache_dir=tmp)
            cache_info = _example_cache_path(cfg, FakePaver())
            self.assertIsNotNone(cache_info)
            cache_path, cache_key, key_config = cache_info
            _save_example_cache(
                cache_path,
                key=cache_key,
                key_config=key_config,
                splits=splits,
            )
            loaded = _load_example_cache(
                cache_path,
                expected_key=cache_key,
                device=torch.device("cpu"),
            )
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.train_fr[0].sample_id, "ex_000")
        self.assertEqual(loaded.train_fr[0].base_ids.device.type, "cpu")
        self.assertTrue(torch.equal(loaded.train_fr[0].boundary_states, torch.ones(2, 3)))


if __name__ == "__main__":
    unittest.main()
