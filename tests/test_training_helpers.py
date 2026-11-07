from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path
import sys

import torch

_ROOT = Path(__file__).resolve().parent.parent
for _d in [str(_ROOT / "experiments"), str(_ROOT / "src" / "trainer"), str(_ROOT / "src" / "optimized")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from experiments.harness import (
    build_boundary_warm_start,
    compute_mean_boundary_direction,
    stepwise_topk_kl_distance,
    topk_kl_loss,
)
from src.trainer.gemma3_residual_paving_binary_merged import (
    BoundaryExpertRouter,
    PavingControllerBank,
    _run_stage1_batches_with_oom_backoff,
    _run_stage2_batches_with_oom_backoff,
    maybe_warm_start_gate,
    maybe_warm_start_router,
    maybe_warm_start_free_controller,
    resolve_stage_train_steps,
    stack_negative_id_sets,
)
from src.trainer.gemma3_residual_paving_binary_refined import AnchoredBinaryGate, _run_batched_with_backoff
import src.trainer.gemma3_residual_paving_hybrid as hybrid


class TrainingHelperTests(unittest.TestCase):
    def test_text_only_wrapper_preserves_resolved_model_revision(self) -> None:
        class FakeLanguageModel(torch.nn.Module):
            pass

        top_config = SimpleNamespace(
            text_config=SimpleNamespace(),
            _commit_hash="deadbeef",
        )
        wrapped = SimpleNamespace(
            model=SimpleNamespace(language_model=FakeLanguageModel()),
            lm_head=torch.nn.Linear(2, 2, bias=False),
            config=top_config,
            generation_config=None,
        )

        model = hybrid.Gemma3TextOnlyCausalLM(wrapped)

        self.assertEqual(model.config._commit_hash, "deadbeef")

    def test_model_loader_forwards_explicit_revision_to_all_hf_loaders(self) -> None:
        calls = []

        class Config:
            model_type = "text"

        class ConfigLoader:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                calls.append(("config", model_id, kwargs))
                return Config()

        class TokenizerLoader:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                calls.append(("tokenizer", model_id, kwargs))
                return SimpleNamespace(
                    pad_token_id=0,
                    eos_token_id=1,
                    padding_side="left",
                )

        class FakeModel:
            def eval(self):
                return self

            def to(self, _device):
                return self

        class ModelLoader:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                calls.append(("model", model_id, kwargs))
                return FakeModel()

        originals = hybrid.AutoConfig, hybrid.AutoTokenizer, hybrid.AutoModelForCausalLM
        original_pick_dtype = hybrid.pick_dtype
        hybrid.AutoConfig = ConfigLoader
        hybrid.AutoTokenizer = TokenizerLoader
        hybrid.AutoModelForCausalLM = ModelLoader
        hybrid.pick_dtype = lambda: torch.float32
        try:
            hybrid.load_model_and_tokenizer(
                "model",
                torch.device("cpu"),
                revision="deadbeef",
            )
        finally:
            hybrid.AutoConfig, hybrid.AutoTokenizer, hybrid.AutoModelForCausalLM = originals
            hybrid.pick_dtype = original_pick_dtype

        self.assertEqual([name for name, _, _ in calls], ["config", "tokenizer", "model"])
        self.assertTrue(all(kwargs["revision"] == "deadbeef" for _, _, kwargs in calls))

    def test_boundary_warm_start_repeats_mean_direction_across_steps(self) -> None:
        fr_examples = [
            SimpleNamespace(boundary_states=torch.tensor([[3.0, 1.0], [2.0, 4.0]], dtype=torch.float32)),
            SimpleNamespace(boundary_states=torch.tensor([[1.0, 5.0], [0.0, 2.0]], dtype=torch.float32)),
        ]
        keep_examples = [
            SimpleNamespace(boundary_states=torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)),
        ]

        mean_direction = compute_mean_boundary_direction(fr_examples, keep_examples)
        init = build_boundary_warm_start(
            mean_direction,
            steer_steps=3,
            scale=0.1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        expected_mean = torch.tensor([[1.0, 2.0], [0.0, 2.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(mean_direction, expected_mean))
        self.assertEqual(tuple(init.shape), (3, 2, 2))
        self.assertTrue(torch.allclose(init[0], expected_mean * 0.1))
        self.assertTrue(torch.allclose(init[1], init[0]))
        self.assertTrue(torch.allclose(init[2], init[0]))

    def test_free_controller_warm_start_initializes_all_experts(self) -> None:
        controller_bank = PavingControllerBank(
            num_experts=2,
            steer_steps=2,
            num_layers=2,
            hidden_size=2,
            controller_mode="free",
        )

        fr_examples = [
            SimpleNamespace(boundary_states=torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)),
        ]
        en_examples = [
            SimpleNamespace(boundary_states=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)),
        ]

        changed = maybe_warm_start_free_controller(
            controller_bank,
            fr_examples=fr_examples,
            en_examples=en_examples,
            init_mode="mean_boundary",
            init_scale=0.5,
        )

        self.assertTrue(changed)
        expected = torch.tensor([[1.0, 0.0], [-0.5, 0.5]], dtype=torch.float32).unsqueeze(0).expand(2, -1, -1)
        self.assertTrue(torch.allclose(controller_bank.expert_residual_vectors[0], expected))
        self.assertTrue(torch.allclose(controller_bank.expert_residual_vectors[1], expected))

    def test_run_batched_with_backoff_retries_smaller_batches_after_oom(self) -> None:
        seen = []
        backoffs = []

        def process_batch(batch) -> None:
            seen.append(list(batch))
            if len(batch) > 1:
                raise torch.OutOfMemoryError("CUDA out of memory")

        def on_backoff(prev_batch_size: int, new_batch_size: int, start: int, total: int) -> None:
            backoffs.append((prev_batch_size, new_batch_size, start, total))

        _run_batched_with_backoff(
            [0, 1, 2, 3],
            initial_batch_size=4,
            process_batch=process_batch,
            on_backoff=on_backoff,
        )

        self.assertEqual(backoffs, [(4, 2, 0, 4), (2, 1, 0, 4)])
        self.assertEqual(seen, [[0, 1, 2, 3], [0, 1], [0], [1], [2], [3]])

    def test_stepwise_topk_kl_distance_matches_mean_topk_kl_loss(self) -> None:
        train_logits = torch.tensor(
            [
                [2.0, 0.0, -1.0],
                [0.0, 1.5, -0.5],
            ],
            dtype=torch.float32,
        )
        teacher_topk_ids = torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.long,
        )
        teacher_topk_probs = torch.tensor(
            [
                [0.7, 0.2],
                [0.6, 0.3],
            ],
            dtype=torch.float32,
        )

        stepwise = stepwise_topk_kl_distance(train_logits, teacher_topk_ids, teacher_topk_probs)
        scalar = topk_kl_loss(train_logits, teacher_topk_ids, teacher_topk_probs)

        self.assertEqual(tuple(stepwise.shape), (2,))
        self.assertTrue(torch.all(stepwise >= 0))
        self.assertAlmostEqual(float(stepwise.mean().item()), float(scalar.item()), places=6)

    def test_stage1_oom_backoff_splits_training_batches(self) -> None:
        seen = []

        def compute_metrics(batch):
            seen.append(len(batch))
            if len(batch) > 2:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return torch.tensor(0.0), {"batch_size": float(len(batch))}

        avg = _run_stage1_batches_with_oom_backoff(
            batches=[[0, 1, 2, 3], [4, 5]],
            label="stage1 unit",
            compute_metrics=compute_metrics,
        )

        self.assertEqual(seen, [4, 2, 2, 2])
        self.assertAlmostEqual(avg["batch_size"], 2.0, places=6)

    def test_stage2_oom_backoff_splits_paired_batches(self) -> None:
        seen = []

        def compute_metrics(fr_batch, en_batch):
            seen.append((len(fr_batch), len(en_batch)))
            if max(len(fr_batch), len(en_batch)) > 2:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return torch.tensor(0.0), {"pair_size": float(len(fr_batch) + len(en_batch))}

        avg = _run_stage2_batches_with_oom_backoff(
            batch_pairs=[([0, 1, 2, 3], [10, 11, 12, 13]), ([4, 5], [14, 15])],
            label="stage2 unit",
            compute_metrics=compute_metrics,
        )

        self.assertEqual(seen, [(4, 4), (2, 2), (2, 2), (2, 2)])
        self.assertAlmostEqual(avg["pair_size"], 4.0, places=6)

    def test_resolve_stage_train_steps_keeps_eval_full_and_caps_train(self) -> None:
        args = SimpleNamespace(steer_steps=8, stage1_train_steps=1, stage2_train_steps=4)

        self.assertEqual(resolve_stage_train_steps(args, stage=1, for_eval=False), 1)
        self.assertEqual(resolve_stage_train_steps(args, stage=2, for_eval=False), 4)
        self.assertEqual(resolve_stage_train_steps(args, stage=1, for_eval=True), 8)
        self.assertEqual(resolve_stage_train_steps(args, stage=2, for_eval=True), 8)

    def test_stack_negative_id_sets_truncates_positive_and_hard_negatives(self) -> None:
        batch = [
            SimpleNamespace(
                negative_ids=torch.tensor([3, 4, 5, 6], dtype=torch.long),
                hard_negative_ids=torch.tensor([7, 8, 9, 10], dtype=torch.long),
            ),
            SimpleNamespace(
                negative_ids=torch.tensor([13, 14, 15, 16], dtype=torch.long),
                hard_negative_ids=torch.tensor([17, 18, 19, 20], dtype=torch.long),
            ),
        ]

        negative_sets = stack_negative_id_sets(batch, max_steps=2)

        self.assertEqual(len(negative_sets), 2)
        self.assertTrue(torch.equal(negative_sets[0], torch.tensor([[3, 4], [13, 14]], dtype=torch.long)))
        self.assertTrue(torch.equal(negative_sets[1], torch.tensor([[7, 8], [17, 18]], dtype=torch.long)))

    def test_router_warm_start_matches_boundary_kmeans_targets(self) -> None:
        examples = [
            SimpleNamespace(boundary_states=torch.tensor([[3.0, 0.0], [3.0, 0.0]], dtype=torch.float32), bucket="a"),
            SimpleNamespace(boundary_states=torch.tensor([[2.0, 0.1], [2.0, 0.1]], dtype=torch.float32), bucket="a"),
            SimpleNamespace(boundary_states=torch.tensor([[0.0, 3.0], [0.0, 3.0]], dtype=torch.float32), bucket="b"),
            SimpleNamespace(boundary_states=torch.tensor([[0.1, 2.0], [0.1, 2.0]], dtype=torch.float32), bucket="b"),
        ]
        router = BoundaryExpertRouter(
            num_layers=2,
            hidden_size=2,
            num_experts=2,
            hidden_dim=4,
            temperature=1.0,
            topk=0,
        )

        changed = maybe_warm_start_router(
            router,
            examples=examples,
            num_experts=2,
            bucket_smoothing=0.0,
            fallback_mode="boundary_kmeans",
            fallback_centroids=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            steps=120,
            lr=0.1,
        )

        self.assertTrue(changed)
        boundary_batch = torch.stack([ex.boundary_states for ex in examples], dim=0)
        _, weights = router(boundary_batch)
        self.assertTrue(torch.equal(weights.argmax(dim=-1), torch.tensor([0, 0, 1, 1], dtype=torch.long)))

    def test_gate_warm_start_learns_easy_boundary_split(self) -> None:
        fr_examples = [
            SimpleNamespace(boundary_states=torch.tensor([[2.0, 0.0], [2.0, 0.0]], dtype=torch.float32)),
            SimpleNamespace(boundary_states=torch.tensor([[1.5, 0.2], [1.5, 0.2]], dtype=torch.float32)),
        ]
        en_examples = [
            SimpleNamespace(boundary_states=torch.tensor([[-2.0, 0.0], [-2.0, 0.0]], dtype=torch.float32)),
            SimpleNamespace(boundary_states=torch.tensor([[-1.5, -0.2], [-1.5, -0.2]], dtype=torch.float32)),
        ]
        gate = AnchoredBinaryGate(
            init_direction=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            init_threshold=0.0,
            sharpness=8.0,
            gate_type="anchored_mlp",
            hidden_dim=8,
        )

        changed = maybe_warm_start_gate(
            gate,
            fr_examples=fr_examples,
            en_examples=en_examples,
            steps=80,
            lr=0.1,
        )

        self.assertTrue(changed)
        boundary_batch = torch.stack([ex.boundary_states for ex in fr_examples + en_examples], dim=0)
        logits, _ = gate(boundary_batch)
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        acc = ((logits > 0.0).float() == labels).float().mean().item()
        self.assertGreaterEqual(acc, 0.99)


if __name__ == "__main__":
    unittest.main()
