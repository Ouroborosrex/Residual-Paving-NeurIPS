from __future__ import annotations

import unittest
import os
from types import SimpleNamespace

import torch

from src.optimized.optimized_binary_runtime import trace_forced_trajectory_batch_optimized
from src.trainer.gemma3_residual_paving_hybrid import ResidualPaver, trace_forced_trajectory


class DummyLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, offset: float) -> None:
        super().__init__()
        self.register_buffer("offset", torch.full((hidden_size,), float(offset), dtype=torch.float32))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.offset.view(1, 1, -1)


class DummyCacheModel(torch.nn.Module):
    def __init__(self, *, support_cache: bool) -> None:
        super().__init__()
        self.support_cache = bool(support_cache)
        self.layers = torch.nn.ModuleList([DummyLayer(3, 0.5), DummyLayer(3, -0.25)])
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=self.layers))
        self.forward_widths = []
        self.forward_cache_flags = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
        past_key_values=None,
        token_type_ids=None,
    ):
        self.forward_widths.append(int(input_ids.shape[1]))
        self.forward_cache_flags.append(bool(use_cache))

        ids = input_ids.float()
        hidden = torch.stack(
            [
                ids,
                ids * 2.0,
                torch.ones_like(ids),
            ],
            dim=-1,
        )
        hidden = hidden * attention_mask.unsqueeze(-1).float()

        for layer in self.layers:
            hidden = layer(hidden)

        vocab_basis = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits = torch.einsum("bsh,vh->bsv", hidden, vocab_basis)
        cache = (torch.tensor([1], device=input_ids.device),) if self.support_cache and use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


class DummyContextModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([DummyLayer(3, 0.0)])
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=self.layers))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
        past_key_values=None,
        token_type_ids=None,
    ):
        ids = input_ids.float()
        hidden = torch.stack([ids, ids * 0.5, torch.ones_like(ids)], dim=-1)
        hidden = hidden * attention_mask.unsqueeze(-1).float()

        for layer in self.layers:
            hidden = layer(hidden)

        # Make the final-token logits depend on every position so all-position
        # interventions differ from last-token-only interventions.
        context = hidden.sum(dim=1, keepdim=True)
        hidden = hidden.clone()
        hidden[:, -1:, :] = hidden[:, -1:, :] + 0.5 * context

        vocab_basis = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits = torch.einsum("bsh,vh->bsv", hidden, vocab_basis)
        return SimpleNamespace(logits=logits, past_key_values=None)


class OptimizedRuntimeTests(unittest.TestCase):
    def test_trace_uses_single_token_updates_when_cache_is_available(self) -> None:
        cached_model = DummyCacheModel(support_cache=True)
        fallback_model = DummyCacheModel(support_cache=False)
        cached_model.eval()
        fallback_model.eval()
        paver = SimpleNamespace(modules=list(cached_model.layers))
        fallback_paver = SimpleNamespace(modules=list(fallback_model.layers))

        base_ids_batch = [
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([3, 4, 1], dtype=torch.long),
        ]
        anchor_ids_batch = [
            torch.tensor([2, 3, 4], dtype=torch.long),
            torch.tensor([1, 2, 3], dtype=torch.long),
        ]
        controller_vectors = torch.tensor(
            [
                [
                    [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
                    [[0.0, 0.0, 0.3], [0.1, 0.0, 0.0]],
                    [[0.0, 0.4, 0.0], [0.0, 0.0, 0.5]],
                ],
                [
                    [[0.2, 0.0, 0.0], [0.0, 0.1, 0.0]],
                    [[0.0, 0.3, 0.0], [0.0, 0.0, 0.2]],
                    [[0.1, 0.0, 0.4], [0.0, 0.2, 0.0]],
                ],
            ],
            dtype=torch.float32,
        )
        gate_strengths = torch.tensor([1.0, 0.5], dtype=torch.float32)

        cached_logits, cached_states = trace_forced_trajectory_batch_optimized(
            model=cached_model,
            paver=paver,
            base_ids_batch=base_ids_batch,
            anchor_ids_batch=anchor_ids_batch,
            controller_vectors=controller_vectors,
            gate_strengths=gate_strengths,
        )
        fallback_logits, fallback_states = trace_forced_trajectory_batch_optimized(
            model=fallback_model,
            paver=fallback_paver,
            base_ids_batch=base_ids_batch,
            anchor_ids_batch=anchor_ids_batch,
            controller_vectors=controller_vectors,
            gate_strengths=gate_strengths,
        )

        self.assertEqual(cached_model.forward_widths, [3, 1, 1])
        self.assertEqual(fallback_model.forward_widths, [3, 3, 4, 5])
        self.assertTrue(all(cached_model.forward_cache_flags))
        self.assertTrue(all(fallback_model.forward_cache_flags[:1]))
        self.assertTrue(torch.allclose(cached_logits, fallback_logits))
        self.assertTrue(torch.allclose(cached_states, fallback_states))

    def test_trace_disables_incremental_cache_for_gemma3_by_default(self) -> None:
        model = DummyCacheModel(support_cache=True)
        model.config = SimpleNamespace(model_type="gemma3")
        model.eval()
        paver = SimpleNamespace(modules=list(model.layers))
        base_ids_batch = [torch.tensor([1, 2, 3], dtype=torch.long)]
        anchor_ids_batch = [torch.tensor([4, 3, 2], dtype=torch.long)]

        prev = os.environ.pop("RESIDUAL_PAVING_ENABLE_UNSAFE_KV_TRACE", None)
        try:
            _ = trace_forced_trajectory_batch_optimized(
                model=model,
                paver=paver,
                base_ids_batch=base_ids_batch,
                anchor_ids_batch=anchor_ids_batch,
                controller_vectors=None,
                gate_strengths=0.0,
            )
        finally:
            if prev is not None:
                os.environ["RESIDUAL_PAVING_ENABLE_UNSAFE_KV_TRACE"] = prev

        self.assertEqual(model.forward_widths, [3, 4, 5])

    def test_hybrid_trace_supports_all_position_intervention(self) -> None:
        model = DummyContextModel()
        model.eval()
        paver_last = ResidualPaver(model=model, layer_indices=[0], intervention_positions="last_token")
        paver_all = ResidualPaver(model=model, layer_indices=[0], intervention_positions="all_positions")
        base_ids = torch.tensor([[1, 2]], dtype=torch.long)
        anchor_ids = torch.tensor([3, 4], dtype=torch.long)
        controller = torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=torch.float32)

        last_logits, _ = trace_forced_trajectory(
            model=model,
            paver=paver_last,
            base_ids=base_ids,
            anchor_ids=anchor_ids,
            controller_vectors=controller,
            gate_strength=1.0,
        )
        all_logits, _ = trace_forced_trajectory(
            model=model,
            paver=paver_all,
            base_ids=base_ids,
            anchor_ids=anchor_ids,
            controller_vectors=controller,
            gate_strength=1.0,
        )

        self.assertFalse(torch.allclose(last_logits, all_logits))

    def test_optimized_trace_supports_all_position_intervention(self) -> None:
        model = DummyContextModel()
        model.eval()
        base_ids_batch = [torch.tensor([1, 2], dtype=torch.long)]
        anchor_ids_batch = [torch.tensor([3, 4], dtype=torch.long)]
        controller_vectors = torch.tensor([[[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]], dtype=torch.float32)

        last_logits, _ = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=SimpleNamespace(modules=list(model.layers), intervention_positions="last_token"),
            base_ids_batch=base_ids_batch,
            anchor_ids_batch=anchor_ids_batch,
            controller_vectors=controller_vectors,
            gate_strengths=torch.tensor([1.0], dtype=torch.float32),
        )
        all_logits, _ = trace_forced_trajectory_batch_optimized(
            model=model,
            paver=SimpleNamespace(modules=list(model.layers), intervention_positions="all_positions"),
            base_ids_batch=base_ids_batch,
            anchor_ids_batch=anchor_ids_batch,
            controller_vectors=controller_vectors,
            gate_strengths=torch.tensor([1.0], dtype=torch.float32),
        )

        self.assertFalse(torch.allclose(last_logits, all_logits))


if __name__ == "__main__":
    unittest.main()
