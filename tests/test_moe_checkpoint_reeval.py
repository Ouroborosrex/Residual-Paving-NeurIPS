from __future__ import annotations

import argparse
import unittest

from experiments.moe_checkpoint_reeval import (
    _attach_target_control_score,
    _checkpoint_overrides,
    _parse_eval_path_spec,
    _resolve_eval_paths,
)


class MoECheckpointReevalTests(unittest.TestCase):
    def test_default_eval_paths_use_selected_scale_where_requested(self) -> None:
        paths = _resolve_eval_paths([], selected_scale=2.0)
        self.assertEqual(
            paths,
            [
                (
                    "thresholded_soft_router_calibrated",
                    "thresholded_soft",
                    "router",
                    2.0,
                    None,
                ),
                ("soft_router_calibrated", "soft", "router", 2.0, None),
                ("oracle_router_calibrated", "oracle", "router", 2.0, None),
                ("oracle_expert0_x4", "oracle", "expert0", 4.0, None),
            ],
        )

    def test_parse_eval_path_spec_accepts_selected_scale(self) -> None:
        self.assertEqual(
            _parse_eval_path_spec("soft_custom:soft:router:selected"),
            ("soft_custom", "soft", "router", "selected", None),
        )

    def test_parse_eval_path_spec_accepts_thresholded_soft_gate(self) -> None:
        self.assertEqual(
            _parse_eval_path_spec("thresholded:thresholded_soft:router:8"),
            ("thresholded", "thresholded_soft", "router", "8", None),
        )

    def test_parse_eval_path_spec_accepts_threshold_override(self) -> None:
        self.assertEqual(
            _parse_eval_path_spec("hard_custom:hard:router:8:2.5"),
            ("hard_custom", "hard", "router", "8", "2.5"),
        )

    def test_resolve_eval_paths_resolves_threshold_override(self) -> None:
        self.assertEqual(
            _resolve_eval_paths(["hard_custom:hard:router:8:2.5"], selected_scale=1.0),
            [("hard_custom", "hard", "router", 8.0, 2.5)],
        )

    def test_parse_eval_path_spec_rejects_bad_gate_policy(self) -> None:
        with self.assertRaises(ValueError):
            _parse_eval_path_spec("bad:nope:router:1")

    def test_parse_eval_path_spec_rejects_bad_scale(self) -> None:
        with self.assertRaises(ValueError):
            _parse_eval_path_spec("bad:soft:router:not-a-scale")

    def test_attach_target_control_score_does_not_mutate_input(self) -> None:
        result = {
            "edit_refusal_rate": 0.0,
            "edit_operational_compliance_heuristic_rate": 0.0,
            "edit_target_alignment": 0.5,
            "keep_preservation": 1.0,
            "harmful_keep_refusal_rate": 1.0,
        }
        scored = _attach_target_control_score(result)
        self.assertNotIn("target_control_score", result)
        self.assertAlmostEqual(scored["target_control_score"], 1.25)

    def test_checkpoint_overrides_support_independent_bucket_counts(self) -> None:
        args = argparse.Namespace(
            name="holdout",
            model_quantization=None,
            model_quant_compute_dtype=None,
            n=100,
            n_edit=40,
            n_keep_benign=50,
            n_keep_harmful=320,
            eval_fraction=0.25,
            data_seed=None,
            eval_seed=None,
            eval_max_new_tokens=None,
            trace_batch_size=None,
            build_batch_size=None,
            edit_refusal_filter_max_new_tokens=None,
            edit_prompts_path=None,
            edit_prompts_prefiltered_base_refusal=False,
            example_cache_dir=None,
            refresh_example_cache=False,
            eval_prefill_anchor_prefix=False,
            harmful_veto=False,
            harmful_veto_l2_weight=None,
            harmful_veto_threshold_tie_break=None,
            harmful_veto_mode=None,
            harmful_veto_temperature=None,
            harmful_veto_min_multiplier=None,
            harmful_veto_ensemble_l2_values=None,
            harmbench_paths="",
        )
        overrides = _checkpoint_overrides({"config": {}}, args)
        self.assertEqual(overrides["n_edit"], 40)
        self.assertEqual(overrides["n_keep_benign"], 50)
        self.assertEqual(overrides["n_keep_harmful"], 320)
        self.assertEqual(overrides["eval_fraction"], 0.25)


if __name__ == "__main__":
    unittest.main()
