from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from experiments.moe_harness import (
    EarlyLayerRouter,
    MoEPavingController,
    MoEResidualPavingExperiment,
    apply_harmful_veto_to_gate_strengths,
    _accuracy_threshold,
    _binary_gate_calibration,
    _compact_preview,
    _encode_prefill_prefix_ids,
    _gate_activation_rate,
    _is_operational_compliance_moe,
    _parse_opening_refusal_phrases,
    _sequence_nll_from_logits,
    _target_token_probability,
    _target_token_rank,
    moe_expert_weights_for_policy,
    moe_gate_strengths_for_policy,
)
from experiments.moe_gate_policy_smoke import (
    _parse_scale_candidates,
    _scale_calibration_score,
    _select_scale_candidate,
    _split_examples_for_calibration,
)


class MoEHarnessTests(unittest.TestCase):
    class _TinyTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            del add_special_tokens
            return [ord(ch) for ch in text]

    def test_gate_policy_soft_uses_sigmoid(self) -> None:
        logits = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        strengths = moe_gate_strengths_for_policy(logits, 0.25, "soft")
        self.assertTrue(torch.allclose(strengths, torch.sigmoid(logits)))

    def test_gate_policy_thresholded_soft_zeroes_below_threshold(self) -> None:
        logits = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        strengths = moe_gate_strengths_for_policy(logits, 0.25, "thresholded_soft")
        expected = torch.tensor([0.0, 0.0, torch.sigmoid(logits[-1]).item()])
        self.assertTrue(torch.allclose(strengths, expected))

    def test_gate_policy_hard_uses_threshold(self) -> None:
        logits = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        strengths = moe_gate_strengths_for_policy(logits, 0.25, "hard")
        self.assertTrue(torch.equal(strengths, torch.tensor([0.0, 0.0, 1.0])))

    def test_gate_policy_hard_tie_stays_off(self) -> None:
        logits = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        strengths = moe_gate_strengths_for_policy(logits, 0.0, "hard")
        self.assertTrue(torch.equal(strengths, torch.tensor([0.0, 0.0, 1.0])))

    def test_gate_policy_oracle_uses_target(self) -> None:
        logits = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        strengths = moe_gate_strengths_for_policy(logits, 0.25, "oracle", oracle_target=0.0)
        self.assertTrue(torch.equal(strengths, torch.zeros_like(logits)))

    def test_gate_policy_oracle_requires_target(self) -> None:
        logits = torch.tensor([0.0], dtype=torch.float32)
        with self.assertRaises(ValueError):
            moe_gate_strengths_for_policy(logits, 0.0, "oracle")

    def test_gate_selectivity_helpers_report_rates_and_calibration(self) -> None:
        self.assertEqual(_gate_activation_rate([-1.0, 0.5, 2.0], 0.0), 2 / 3)
        brier, ece = _binary_gate_calibration([4.0, -4.0], [1.0, 0.0], bins=2)
        self.assertLess(brier, 0.001)
        self.assertLess(ece, 0.02)

    def test_compact_preview_collapses_whitespace_and_truncates(self) -> None:
        self.assertEqual(_compact_preview("a\n\n b\tc", limit=20), "a b c")
        self.assertEqual(_compact_preview("abcdefghij", limit=8), "abcde...")

    def test_eval_preview_examples_prefers_eval_and_falls_back_to_train(self) -> None:
        exp = object.__new__(MoEResidualPavingExperiment)
        eval_edit = SimpleNamespace(name="eval_edit")
        train_edit = SimpleNamespace(name="train_edit")
        train_benign = SimpleNamespace(name="train_benign")
        eval_harmful = SimpleNamespace(name="eval_harmful")
        train_harmful = SimpleNamespace(name="train_harmful")
        exp.eval_fr_examples = [eval_edit]
        exp.fr_examples = [train_edit]
        exp.eval_keep_benign = []
        exp.keep_benign = [train_benign]
        exp.eval_keep_harmful = [eval_harmful]
        exp.keep_harmful = [train_harmful]

        rows = MoEResidualPavingExperiment._eval_preview_examples(exp)

        self.assertEqual([row[0] for row in rows], ["edit", "benign_keep", "harmful_keep"])
        self.assertIs(rows[0][2], eval_edit)
        self.assertIs(rows[1][2], train_benign)
        self.assertIs(rows[2][2], eval_harmful)

    def test_harmful_veto_zeroes_non_oracle_gate_strengths(self) -> None:
        class DummyExp:
            cfg = SimpleNamespace(harmful_veto_enabled=True)
            harmful_veto_threshold = 0.0

            def _harmful_veto_logits(self, boundary_states):
                del boundary_states
                return torch.tensor([-1.0, 2.0], dtype=torch.float32)

        strengths, logits = apply_harmful_veto_to_gate_strengths(
            DummyExp(),
            torch.zeros(2, 1, 1),
            torch.tensor([3.0, 3.0], dtype=torch.float32),
            "soft",
        )
        self.assertTrue(torch.equal(strengths, torch.tensor([3.0, 0.0])))
        self.assertIsNotNone(logits)

    def test_soft_harmful_veto_attenuates_non_oracle_gate_strengths(self) -> None:
        class DummyExp:
            cfg = SimpleNamespace(
                harmful_veto_enabled=True,
                harmful_veto_mode="soft",
                harmful_veto_temperature=1.0,
                harmful_veto_min_multiplier=0.0,
            )
            harmful_veto_threshold = 0.0

            def _harmful_veto_logits(self, boundary_states):
                del boundary_states
                return torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)

        strengths, logits = apply_harmful_veto_to_gate_strengths(
            DummyExp(),
            torch.zeros(3, 1, 1),
            torch.full((3,), 4.0, dtype=torch.float32),
            "soft",
        )
        expected = 4.0 * torch.sigmoid(torch.tensor([1.0, 0.0, -1.0]))
        self.assertTrue(torch.allclose(strengths, expected))
        self.assertIsNotNone(logits)

    def test_harmful_veto_ensemble_logits_use_max_margin(self) -> None:
        class DummyRouter:
            input_dim = 1

            def _flatten_and_normalize(self, boundary_states):
                return boundary_states.reshape(boundary_states.shape[0], -1).float()

        head_a = torch.nn.Linear(1, 1)
        head_b = torch.nn.Linear(1, 1)
        with torch.no_grad():
            head_a.weight.fill_(1.0)
            head_a.bias.zero_()
            head_b.weight.fill_(-1.0)
            head_b.bias.zero_()

        exp = object.__new__(MoEResidualPavingExperiment)
        exp.device = torch.device("cpu")
        exp.moe_router = DummyRouter()
        exp.harmful_veto_head = torch.nn.ModuleList([head_a, head_b])
        exp.harmful_veto_thresholds = [0.5, 1.0]

        logits = MoEResidualPavingExperiment._harmful_veto_logits(
            exp,
            torch.tensor([[[2.0]], [[-2.0]]], dtype=torch.float32),
        )
        self.assertTrue(torch.allclose(logits, torch.tensor([1.5, 1.0])))

    def test_harmful_veto_leaves_oracle_gate_strengths_unchanged(self) -> None:
        class DummyExp:
            cfg = SimpleNamespace(harmful_veto_enabled=True)
            harmful_veto_threshold = 0.0

            def _harmful_veto_logits(self, boundary_states):
                del boundary_states
                return torch.tensor([2.0], dtype=torch.float32)

        strengths, logits = apply_harmful_veto_to_gate_strengths(
            DummyExp(),
            torch.zeros(1, 1, 1),
            torch.tensor([3.0], dtype=torch.float32),
            "oracle",
        )
        self.assertTrue(torch.equal(strengths, torch.tensor([3.0])))
        self.assertIsNone(logits)

    def test_accuracy_threshold_can_use_high_tie_break(self) -> None:
        logits = torch.tensor([-2.0, 0.0, 10.0], dtype=torch.float32)
        targets = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

        low_threshold, low_acc = _accuracy_threshold(logits, targets, tie_break="low")
        high_threshold, high_acc = _accuracy_threshold(logits, targets, tie_break="high")

        self.assertEqual(low_acc, high_acc)
        self.assertGreater(high_threshold, low_threshold)
        self.assertAlmostEqual(low_threshold, 5.0)
        self.assertGreater(high_threshold, 9.99)
        self.assertLess(high_threshold, 10.0)

    def test_expert_policy_router_preserves_weights(self) -> None:
        weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
        routed = moe_expert_weights_for_policy(weights, "router")
        self.assertTrue(torch.equal(routed, weights))

    def test_expert_policy_uniform_spreads_weights(self) -> None:
        weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
        routed = moe_expert_weights_for_policy(weights, "uniform")
        self.assertTrue(torch.equal(routed, torch.full_like(weights, 1.0 / 3.0)))

    def test_expert_policy_forces_single_expert(self) -> None:
        weights = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]], dtype=torch.float32)
        routed = moe_expert_weights_for_policy(weights, "expert0")
        expected = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(routed, expected))

    def test_expert_policy_rejects_out_of_range_expert(self) -> None:
        weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
        with self.assertRaises(ValueError):
            moe_expert_weights_for_policy(weights, "expert3")

    def test_controller_expert_gain_scales_intervention(self) -> None:
        controller = MoEPavingController(
            num_experts=1,
            hidden_size=1,
            bottleneck_dim=1,
            num_intervention_layers=1,
            expert_center_fracs=[0.0],
            expert_sigma_fracs=[1.0],
            expert_gain_inits=[4.0],
        )
        with torch.no_grad():
            controller.experts[0].down.weight.fill_(1.0)
            controller.experts[0].up.weight.fill_(1.0)

        h = torch.tensor([[[0.1]]], dtype=torch.float32)
        out = controller.apply_at_layer(
            h,
            layer_slot=0,
            expert_weights=torch.ones(1, 1),
            gate_strengths=torch.ones(1),
        )
        expected = h + 4.0 * torch.nn.functional.gelu(h)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_router_normalizes_inputs_before_gate(self) -> None:
        router = EarlyLayerRouter(
            num_router_layers=1,
            hidden_size=2,
            num_experts=1,
            router_hidden_dim=1,
        )
        router.set_input_normalization(
            mean=torch.tensor([10.0, 20.0]),
            scale=torch.tensor([2.0, 4.0]),
        )
        with torch.no_grad():
            router.gate_proj[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
            router.gate_proj[0].bias.zero_()
            router.gate_proj[2].weight.copy_(torch.tensor([[1.0]]))
            router.gate_proj[2].bias.zero_()

        logits, expert_weights = router(torch.tensor([[[12.0, 24.0]]]))
        self.assertTrue(torch.allclose(logits, torch.tanh(torch.tensor([1.0]))))
        self.assertTrue(torch.equal(expert_weights, torch.ones(1, 1)))

    def test_router_normalization_rejects_wrong_shape(self) -> None:
        router = EarlyLayerRouter(
            num_router_layers=1,
            hidden_size=2,
            num_experts=1,
            router_hidden_dim=1,
        )
        with self.assertRaises(ValueError):
            router.set_input_normalization(
                mean=torch.tensor([0.0]),
                scale=torch.tensor([1.0]),
            )

    def test_paired_fr_gate_logits_keep_router_gradient(self) -> None:
        class DummyRouter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0, -0.5]))

            def forward(self, boundary_states: torch.Tensor):
                flat = boundary_states.reshape(boundary_states.shape[0], -1).float()
                logits = flat.matmul(self.weight)
                weights = torch.ones(boundary_states.shape[0], 1)
                return logits, weights

        router = DummyRouter()
        exp = object.__new__(MoEResidualPavingExperiment)
        exp.device = torch.device("cpu")
        exp.moe_router = router
        exp.fr_examples = [
            SimpleNamespace(boundary_states=torch.tensor([[1.0, 0.0]])),
            SimpleNamespace(boundary_states=torch.tensor([[0.0, 2.0]])),
        ]

        logits = MoEResidualPavingExperiment._paired_fr_gate_logits(exp, 3, offset=1)
        self.assertTrue(torch.allclose(logits, torch.tensor([-1.0, 1.0, -1.0])))

        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(router.weight.grad)
        self.assertGreater(float(router.weight.grad.abs().sum().item()), 0.0)

    def test_paired_fr_gate_logits_can_boundary_match_harmful_examples(self) -> None:
        class DummyRouter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([10.0, 1.0]))

            def forward(self, boundary_states: torch.Tensor):
                flat = boundary_states.reshape(boundary_states.shape[0], -1).float()
                logits = flat.matmul(self.weight)
                weights = torch.ones(boundary_states.shape[0], 1)
                return logits, weights

        exp = object.__new__(MoEResidualPavingExperiment)
        exp.cfg = SimpleNamespace(harmful_gate_pair_match="boundary_cosine")
        exp.device = torch.device("cpu")
        exp.moe_router = DummyRouter()
        exp.fr_examples = [
            SimpleNamespace(boundary_states=torch.tensor([[1.0, 0.0]])),
            SimpleNamespace(boundary_states=torch.tensor([[0.0, 1.0]])),
        ]
        harmful_examples = [
            SimpleNamespace(boundary_states=torch.tensor([[0.9, 0.1]])),
            SimpleNamespace(boundary_states=torch.tensor([[0.1, 0.9]])),
        ]

        logits = MoEResidualPavingExperiment._paired_fr_gate_logits(
            exp,
            2,
            offset=1,
            match_examples=harmful_examples,
        )
        self.assertTrue(torch.allclose(logits, torch.tensor([10.0, 1.0])))

    def test_gate_boundary_targets_marks_harmful_keep_kind(self) -> None:
        exp = object.__new__(MoEResidualPavingExperiment)
        exp.device = torch.device("cpu")
        exp.fr_examples = [
            SimpleNamespace(
                kind="fr",
                gate_target=1.0,
                boundary_states=torch.tensor([[1.0, 0.0]]),
            )
        ]
        exp.keep_benign = [
            SimpleNamespace(
                kind="benign_keep",
                gate_target=0.0,
                boundary_states=torch.tensor([[0.0, 1.0]]),
            )
        ]
        exp.keep_harmful = [
            SimpleNamespace(
                kind="harmful_keep",
                gate_target=0.0,
                boundary_states=torch.tensor([[0.5, 0.5]]),
            )
        ]

        boundaries, targets, kinds = MoEResidualPavingExperiment._gate_boundary_targets(exp)
        self.assertEqual(tuple(boundaries.shape), (3, 1, 2))
        self.assertTrue(torch.equal(targets, torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(kinds, torch.tensor([0, 1, 2])))

    def test_harmful_pair_margin_loss_is_default_off(self) -> None:
        exp = object.__new__(MoEResidualPavingExperiment)
        exp.cfg = SimpleNamespace(harmful_gate_pair_margin_weight=0.0, gate_margin=1.0)
        loss = MoEResidualPavingExperiment._gate_harmful_pair_margin_loss(
            exp,
            torch.tensor([1.0, 0.0, 2.0]),
            torch.tensor([0, 1, 2]),
        )
        self.assertEqual(float(loss.item()), 0.0)

    def test_harmful_pair_margin_loss_penalizes_close_harmful_logit(self) -> None:
        exp = object.__new__(MoEResidualPavingExperiment)
        exp.cfg = SimpleNamespace(harmful_gate_pair_margin_weight=2.0, gate_margin=1.0)
        loss = MoEResidualPavingExperiment._gate_harmful_pair_margin_loss(
            exp,
            torch.tensor([3.0, 0.0, 2.5]),
            torch.tensor([0, 1, 2]),
        )
        self.assertAlmostEqual(float(loss.item()), 1.0)

    def test_harmful_pair_margin_loss_can_use_boundary_match(self) -> None:
        exp = object.__new__(MoEResidualPavingExperiment)
        exp.device = torch.device("cpu")
        exp.cfg = SimpleNamespace(
            harmful_gate_pair_margin_weight=1.0,
            harmful_gate_pair_match="boundary_cosine",
            gate_margin=2.0,
        )
        exp.fr_examples = [
            SimpleNamespace(boundary_states=torch.tensor([[1.0, 0.0]])),
            SimpleNamespace(boundary_states=torch.tensor([[0.0, 1.0]])),
        ]
        exp.keep_harmful = [
            SimpleNamespace(boundary_states=torch.tensor([[0.1, 0.9]]))
        ]

        loss = MoEResidualPavingExperiment._gate_harmful_pair_margin_loss(
            exp,
            torch.tensor([10.0, 1.0, 0.0]),
            torch.tensor([0, 0, 2]),
        )
        self.assertAlmostEqual(float(loss.item()), 1.0)

    def test_harmful_pair_margin_loss_can_use_hard_topk_violations(self) -> None:
        exp = object.__new__(MoEResidualPavingExperiment)
        exp.cfg = SimpleNamespace(
            harmful_gate_pair_margin_weight=1.0,
            harmful_gate_pair_match="hard_topk",
            harmful_gate_hard_neg_k=2,
            gate_margin=1.0,
        )
        loss = MoEResidualPavingExperiment._gate_harmful_pair_margin_loss(
            exp,
            torch.tensor([4.0, 1.0, 3.5, -2.0]),
            torch.tensor([0, 0, 2, 2]),
        )
        self.assertAlmostEqual(float(loss.item()), 2.0)

    def test_harmful_expert0_avoidance_loss_is_default_off(self) -> None:
        exp = object.__new__(MoEResidualPavingExperiment)
        exp.cfg = SimpleNamespace(harmful_expert0_avoid_weight=0.0)
        weights = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
        loss = MoEResidualPavingExperiment._harmful_expert0_avoidance_loss(exp, weights)
        self.assertEqual(float(loss.item()), 0.0)

    def test_harmful_expert0_avoidance_loss_penalizes_expert0_mass(self) -> None:
        exp = object.__new__(MoEResidualPavingExperiment)
        exp.cfg = SimpleNamespace(harmful_expert0_avoid_weight=2.0)
        weights = torch.tensor(
            [[0.8, 0.2], [0.4, 0.6]],
            dtype=torch.float32,
            requires_grad=True,
        )
        loss = MoEResidualPavingExperiment._harmful_expert0_avoidance_loss(exp, weights)
        self.assertAlmostEqual(float(loss.item()), 1.2)

    def test_harmful_expert0_avoidance_loss_ignores_forced_weights(self) -> None:
        exp = object.__new__(MoEResidualPavingExperiment)
        exp.cfg = SimpleNamespace(harmful_expert0_avoid_weight=2.0)
        weights = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        loss = MoEResidualPavingExperiment._harmful_expert0_avoidance_loss(exp, weights)
        self.assertEqual(float(loss.item()), 0.0)

    def test_edit_expert0_attraction_loss_penalizes_non_expert0_mass(self) -> None:
        exp = object.__new__(MoEResidualPavingExperiment)
        exp.cfg = SimpleNamespace(edit_expert0_attract_weight=2.0)
        weights = torch.tensor(
            [[0.8, 0.2], [0.4, 0.6]],
            dtype=torch.float32,
            requires_grad=True,
        )
        loss = MoEResidualPavingExperiment._edit_expert0_attraction_loss(exp, weights)
        self.assertAlmostEqual(float(loss.item()), 0.8)

    def test_edit_expert0_attraction_loss_ignores_forced_weights(self) -> None:
        exp = object.__new__(MoEResidualPavingExperiment)
        exp.cfg = SimpleNamespace(edit_expert0_attract_weight=2.0)
        weights = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        loss = MoEResidualPavingExperiment._edit_expert0_attraction_loss(exp, weights)
        self.assertEqual(float(loss.item()), 0.0)

    def test_gate_boundary_training_can_apply_harmful_logit_margin(self) -> None:
        class TinyRouter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = torch.nn.Linear(1, 1, bias=False)
                torch.nn.init.constant_(self.gate_proj.weight, 1.0)

            def forward(self, boundaries):
                logits = self.gate_proj(boundaries.view(-1, 1)).view(-1)
                return logits, None

        exp = object.__new__(MoEResidualPavingExperiment)
        exp.device = torch.device("cpu")
        exp.cfg = SimpleNamespace(
            build_batch_size=3,
            trace_batch_size=3,
            data_seed=0,
            harmful_gate_margin_weight=1.0,
            harmful_gate_logit_target=-4.0,
            harmful_gate_pair_margin_weight=0.0,
        )
        exp.moe_router = TinyRouter()
        exp.fr_examples = [
            SimpleNamespace(kind="fr", gate_target=1.0, boundary_states=torch.tensor([[1.0]]))
        ]
        exp.keep_benign = [
            SimpleNamespace(
                kind="benign_keep", gate_target=0.0, boundary_states=torch.tensor([[0.0]])
            )
        ]
        exp.keep_harmful = [
            SimpleNamespace(
                kind="harmful_keep", gate_target=0.0, boundary_states=torch.tensor([[2.0]])
            )
        ]

        calls = []

        def fake_margin(logits):
            calls.append(logits.detach().clone())
            return logits.sum() * 0.0

        exp._harmful_gate_margin_loss = fake_margin

        MoEResidualPavingExperiment._train_gate_on_boundaries(
            exp,
            n_epochs=1,
            lr=0.0,
            label="test",
            params=list(exp.moe_router.gate_proj.parameters()),
            seed_offset=0,
            apply_harmful_gate_margin=False,
        )
        self.assertEqual(calls, [])

        MoEResidualPavingExperiment._train_gate_on_boundaries(
            exp,
            n_epochs=1,
            lr=0.0,
            label="test",
            params=list(exp.moe_router.gate_proj.parameters()),
            seed_offset=0,
            apply_harmful_gate_margin=True,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(calls[0].shape), (1,))
        self.assertAlmostEqual(float(calls[0].item()), 2.0)

    def test_parse_opening_refusal_phrases_strips_empty_parts(self) -> None:
        self.assertEqual(
            _parse_opening_refusal_phrases(" I'm sorry | | I understand "),
            ("I'm sorry", "I understand"),
        )

    def test_encode_prefill_prefix_ids_rejects_empty_prefix(self) -> None:
        self.assertIsNone(
            _encode_prefill_prefix_ids(
                self._TinyTokenizer(),
                "   ",
                device=torch.device("cpu"),
            )
        )

    def test_encode_prefill_prefix_ids_keeps_full_prefix(self) -> None:
        encoded = _encode_prefill_prefix_ids(
            self._TinyTokenizer(),
            "AB",
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.equal(encoded, torch.tensor([65, 66], dtype=torch.long)))

    def test_sequence_nll_from_logits_scores_target_sequence(self) -> None:
        logits = torch.full((2, 4), -5.0, dtype=torch.float32)
        logits[0, 1] = 5.0
        logits[1, 2] = 5.0
        good = _sequence_nll_from_logits(logits, torch.tensor([1, 2]))
        bad = _sequence_nll_from_logits(logits, torch.tensor([2, 1]))
        self.assertLess(float(good.item()), 0.001)
        self.assertGreater(float(bad.item()), 9.0)

    def test_target_token_rank_and_probability_score_target(self) -> None:
        logits = torch.tensor([0.0, 3.0, 1.0], dtype=torch.float32)
        self.assertEqual(_target_token_rank(logits, 1), 1)
        self.assertEqual(_target_token_rank(logits, 2), 2)
        self.assertGreater(_target_token_probability(logits, 1), 0.8)

    def test_parse_scale_candidates_rejects_negative_scales(self) -> None:
        with self.assertRaises(ValueError):
            _parse_scale_candidates("0.5,-1,2")

    def test_split_examples_for_calibration_keeps_test_slice_when_possible(self) -> None:
        calib, test, reused = _split_examples_for_calibration([1, 2, 3, 4], 0.5)
        self.assertEqual(calib, [1, 2])
        self.assertEqual(test, [3, 4])
        self.assertFalse(reused)

    def test_split_examples_for_calibration_reuses_singleton(self) -> None:
        calib, test, reused = _split_examples_for_calibration([1], 0.5)
        self.assertEqual(calib, [1])
        self.assertEqual(test, [1])
        self.assertTrue(reused)

    def test_select_scale_candidate_prefers_lower_scale_on_score_tie(self) -> None:
        selected, score = _select_scale_candidate(
            {
                "0.5": {
                    "edit_refusal_rate": 0.0,
                    "keep_preservation": 0.9,
                    "harmful_keep_refusal_rate": 0.0,
                },
                "1": {
                    "edit_refusal_rate": 0.0,
                    "keep_preservation": 0.9,
                    "harmful_keep_refusal_rate": 0.0,
                },
            }
        )
        self.assertEqual(selected, 0.5)
        self.assertAlmostEqual(score, 1.09)

    def test_operational_compliance_heuristic_ignores_refusals(self) -> None:
        self.assertFalse(
            _is_operational_compliance_moe("I can't help with step-by-step instructions.")
        )

    def test_operational_compliance_heuristic_catches_operational_nonrefusal(self) -> None:
        self.assertTrue(
            _is_operational_compliance_moe("Okay, here's how to do it step-by-step.")
        )

    def test_operational_compliance_heuristic_allows_safe_breakdown(self) -> None:
        self.assertFalse(
            _is_operational_compliance_moe(
                "Let's break down why this request is problematic and risky."
            )
        )

    def test_operational_compliance_heuristic_catches_outline_how(self) -> None:
        self.assertTrue(
            _is_operational_compliance_moe("Okay, I will outline *how* someone might do it.")
        )

    def test_scale_calibration_score_penalizes_operational_compliance(self) -> None:
        score = _scale_calibration_score(
            {
                "edit_refusal_rate": 0.0,
                "edit_operational_compliance_heuristic_rate": 1.0,
                "keep_preservation": 1.0,
                "harmful_keep_refusal_rate": 0.0,
            }
        )
        self.assertAlmostEqual(score, 0.1)

    def test_scale_calibration_score_rewards_target_alignment_as_tiebreaker(self) -> None:
        low = _scale_calibration_score(
            {
                "edit_refusal_rate": 0.0,
                "edit_operational_compliance_heuristic_rate": 0.0,
                "edit_target_alignment": 0.25,
                "keep_preservation": 1.0,
                "harmful_keep_refusal_rate": 0.0,
            }
        )
        high = _scale_calibration_score(
            {
                "edit_refusal_rate": 0.0,
                "edit_operational_compliance_heuristic_rate": 0.0,
                "edit_target_alignment": 0.75,
                "keep_preservation": 1.0,
                "harmful_keep_refusal_rate": 0.0,
            }
        )
        self.assertGreater(high, low)

    def test_scale_calibration_score_uses_harmful_delta_when_available(self) -> None:
        score = _scale_calibration_score(
            {
                "edit_refusal_rate": 0.0,
                "edit_operational_compliance_heuristic_rate": 0.0,
                "keep_preservation": 1.0,
                "harmful_keep_refusal_rate": 1.0,
                "harmful_keep_refusal_delta_vs_base": -0.25,
            }
        )
        self.assertAlmostEqual(score, 1.175)


if __name__ == "__main__":
    unittest.main()
