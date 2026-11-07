from __future__ import annotations

import unittest

from experiments.target_anchor_smoke import _parse_prefix_spec, _score_anchor_texts


class TargetAnchorSmokeTests(unittest.TestCase):
    def test_parse_prefix_spec_allows_empty_text(self) -> None:
        self.assertEqual(_parse_prefix_spec("none="), ("none", ""))

    def test_parse_prefix_spec_requires_name(self) -> None:
        with self.assertRaises(ValueError):
            _parse_prefix_spec("=At a high level,")

    def test_score_anchor_texts_penalizes_operational_compliance(self) -> None:
        scored = _score_anchor_texts(
            [
                "At a high level, focus on safety and prevention.",
                "Okay, here's how to do it step-by-step.",
            ],
            prefix_token_count=4,
        )
        self.assertEqual(scored["examples"], 2)
        self.assertAlmostEqual(scored["anchor_operational_compliance_rate"], 0.5)
        self.assertAlmostEqual(scored["target_anchor_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
