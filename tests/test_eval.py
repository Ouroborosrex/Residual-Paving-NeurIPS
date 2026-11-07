from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from experiments.eval import _fit_linear_probe_accuracy, _is_refusal


def _ex(vec):
    return SimpleNamespace(boundary_states=torch.tensor(vec, dtype=torch.float32))


class EvalTests(unittest.TestCase):
    def test_refusal_heuristic_flags_common_markers(self) -> None:
        self.assertTrue(_is_refusal("I'm sorry, but I can't help with that."))
        self.assertTrue(_is_refusal("Je ne peux pas vous aider avec cela."))
        self.assertFalse(_is_refusal("Here is a short explanation in English."))

    def test_linear_probe_accuracy_runs_on_cpu_features(self) -> None:
        exp = SimpleNamespace(
            fr_examples=[_ex([[2.0, 2.0]]), _ex([[1.5, 1.0]])],
            keep_benign=[_ex([[-2.0, -2.0]])],
            keep_harmful=[_ex([[-1.0, -1.5]])],
            eval_fr_examples=[_ex([[2.5, 1.5]])],
            eval_keep_benign=[_ex([[-2.5, -1.5]])],
            eval_keep_harmful=[_ex([[-1.5, -2.5]])],
        )
        acc = _fit_linear_probe_accuracy(exp)
        self.assertGreaterEqual(acc, 0.99)


if __name__ == "__main__":
    unittest.main()
