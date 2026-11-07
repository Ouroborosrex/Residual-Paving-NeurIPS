from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

_ROOT = Path(__file__).resolve().parent.parent
for _d in [str(_ROOT / "src" / "optimized"), str(_ROOT / "src" / "trainer")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from src.optimized.frozen_reference_cache import CachedTeacherTopK, approximate_topk_union_kl


def _reference_approximate_topk_union_kl(
    student_logits: torch.Tensor,
    teacher: CachedTeacherTopK,
    *,
    student_topk: int,
    extra_ids: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    logits_f = student_logits.float()
    batch_size, steps, vocab_size = logits_f.shape
    k = max(0, min(int(student_topk), vocab_size))
    student_top_ids = None
    if k > 0:
        student_top_ids = torch.topk(logits_f, k=k, dim=-1).indices
    log_z = torch.logsumexp(logits_f, dim=-1)
    losses = logits_f.new_zeros((batch_size, steps), dtype=torch.float32)

    for batch_idx in range(batch_size):
        for step_idx in range(steps):
            union = []
            seen = set()

            def add_ids(values) -> None:
                for item in values:
                    value = int(item)
                    if value not in seen:
                        seen.add(value)
                        union.append(value)

            add_ids(teacher.ids[batch_idx, step_idx].tolist())
            if student_top_ids is not None:
                add_ids(student_top_ids[batch_idx, step_idx].tolist())
            if extra_ids is not None:
                if extra_ids.ndim == 2:
                    add_ids([int(extra_ids[batch_idx, step_idx].item())])
                elif extra_ids.ndim == 3:
                    add_ids(extra_ids[batch_idx, step_idx].tolist())
                else:
                    raise ValueError(f"Unsupported extra_ids rank: {extra_ids.ndim}")

            union_ids = torch.tensor(union, device=logits_f.device, dtype=torch.long)
            student_log_probs = logits_f[batch_idx, step_idx].index_select(0, union_ids) - log_z[batch_idx, step_idx]
            teacher_probs = logits_f.new_zeros((len(union),), dtype=torch.float32)
            teacher_index = {
                int(token_id): idx for idx, token_id in enumerate(teacher.ids[batch_idx, step_idx].tolist())
            }
            for pos, token_id in enumerate(union):
                cache_idx = teacher_index.get(int(token_id))
                if cache_idx is not None:
                    teacher_probs[pos] = teacher.probs[batch_idx, step_idx, cache_idx].float()
            nonzero_mask = teacher_probs > float(eps)
            step_loss = student_log_probs.new_zeros(())
            if nonzero_mask.any():
                step_loss = (
                    teacher_probs[nonzero_mask]
                    * (
                        torch.log(teacher_probs[nonzero_mask].clamp_min(float(eps)))
                        - student_log_probs[nonzero_mask]
                    )
                ).sum()
            teacher_rest = teacher.rest[batch_idx, step_idx].float()
            if float(teacher_rest.item()) > float(eps):
                student_mass = student_log_probs.exp().sum().clamp(max=1.0)
                student_rest = (1.0 - student_mass).clamp_min(float(eps))
                step_loss = step_loss + teacher_rest * (
                    torch.log(teacher_rest.clamp_min(float(eps))) - torch.log(student_rest)
                )
            losses[batch_idx, step_idx] = torch.clamp(step_loss, min=0.0)
    return losses


class FrozenReferenceCacheTests(unittest.TestCase):
    def test_vectorized_union_kl_matches_reference_for_supported_extra_id_shapes(self) -> None:
        torch.manual_seed(7)
        student_logits = torch.randn(2, 3, 11, dtype=torch.float32)
        teacher_ids = torch.tensor(
            [
                [[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                [[1, 2, 3], [3, 4, 5], [5, 6, 7]],
            ],
            dtype=torch.long,
        )
        teacher_probs = torch.tensor(
            [
                [[0.30, 0.20, 0.10], [0.25, 0.20, 0.10], [0.20, 0.18, 0.12]],
                [[0.28, 0.17, 0.15], [0.26, 0.19, 0.11], [0.18, 0.17, 0.15]],
            ],
            dtype=torch.float32,
        )
        teacher_rest = 1.0 - teacher_probs.sum(dim=-1)
        teacher = CachedTeacherTopK(ids=teacher_ids, probs=teacher_probs, rest=teacher_rest)

        for extra_ids in (
            None,
            torch.tensor([[1, 5, 4], [2, 6, 8]], dtype=torch.long),
            torch.tensor(
                [
                    [[0, 9], [3, 10], [4, 6]],
                    [[2, 8], [3, 7], [1, 9]],
                ],
                dtype=torch.long,
            ),
        ):
            with self.subTest(extra_rank=None if extra_ids is None else extra_ids.ndim):
                actual = approximate_topk_union_kl(
                    student_logits,
                    teacher,
                    student_topk=4,
                    extra_ids=extra_ids,
                    eps=1e-8,
                )
                expected = _reference_approximate_topk_union_kl(
                    student_logits,
                    teacher,
                    student_topk=4,
                    extra_ids=extra_ids,
                    eps=1e-8,
                )
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
