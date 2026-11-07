from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from experiments.moe_export_completions import _checkpoint_overrides, _parse_args


class CompletionExportArgumentsTests(unittest.TestCase):
    def test_export_defaults_work_with_shared_checkpoint_loader(self):
        with patch.object(sys, "argv", ["export", "--checkpoint", "example.pt"]):
            args = _parse_args()
        overrides = _checkpoint_overrides({"config": {"n_edit": 40}}, args)
        self.assertEqual(overrides["n_edit"], 40)

    def test_export_can_select_explicit_prompt_source_and_bucket_counts(self):
        with patch.object(sys, "argv", ["export", "--checkpoint", "example.pt",
                "--n", "100", "--n-edit", "40", "--eval-fraction", "0.25",
                "--edit-prompts-path", "edits.json", "--edit-prompts-prefiltered-base-refusal"]):
            args = _parse_args()
        overrides = _checkpoint_overrides({"config": {}}, args)
        self.assertEqual(overrides["n_edit"], 40)
        self.assertEqual(overrides["n_keep_harmful"], 100)
        self.assertEqual(overrides["eval_fraction"], 0.25)
        self.assertEqual(overrides["edit_prompts_path"], "edits.json")
        self.assertTrue(overrides["edit_prompts_prefiltered_base_refusal"])


if __name__ == "__main__":
    unittest.main()
