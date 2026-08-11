"""Regression tests for exact four-modality masking budgets."""

import math
import unittest
from types import SimpleNamespace

import torch

from embodied_mae_4m import (
    EmbodiedMAE4M,
    _bounded_proportional_allocation,
    _visible_token_counts,
)


class MaskAllocationTests(unittest.TestCase):

    LENGTHS = (196, 196, 196, 25)
    WEIGHTS = (0.55, 0.25, 0.15, 0.05)

    def test_requested_global_ratios_are_honored(self):
        total = sum(self.LENGTHS)
        for ratio in (0.15, 0.25, 0.75):
            with self.subTest(ratio=ratio):
                visible = _visible_token_counts(
                    self.LENGTHS, self.WEIGHTS, ratio, min_mask_ratio=0.25
                )
                expected_visible = int(math.floor(total * (1.0 - ratio)))
                self.assertEqual(sum(visible), expected_visible)
                self.assertTrue(all(count >= 1 for count in visible))
                self.assertTrue(all(count < length for count, length in zip(
                    visible, self.LENGTHS
                )))

    def test_minimum_mask_caps_are_relaxed_only_when_unavoidable(self):
        caps = tuple(int(math.floor(length * 0.75)) for length in self.LENGTHS)
        visible = _visible_token_counts(
            self.LENGTHS, self.WEIGHTS, 0.15, min_mask_ratio=0.25
        )

        self.assertTrue(all(count >= cap for count, cap in zip(visible, caps)))
        self.assertEqual(sum(count - cap for count, cap in zip(visible, caps)), 62)

    def test_capped_allocation_redistributes_saturated_share(self):
        allocation = _bounded_proportional_allocation(
            total=12,
            weights=(0.9, 0.05, 0.05),
            lower=(1, 1, 1),
            upper=(4, 10, 10),
        )

        self.assertEqual(sum(allocation), 12)
        self.assertEqual(allocation[0], 4)
        self.assertTrue(all(lo <= value <= hi for value, lo, hi in zip(
            allocation, (1, 1, 1), (4, 10, 10)
        )))

    def test_masks_match_the_exact_global_budget_per_sample(self):
        batch = 3
        dim = 8
        tensors = tuple(torch.randn(batch, length, dim) for length in self.LENGTHS)
        owner = SimpleNamespace(dirichlet_alpha=1.0)

        outputs = EmbodiedMAE4M.random_masking_dirichlet(
            owner, *tensors, mask_ratio_total=0.15
        )
        visible = outputs[:4]
        masks = outputs[4:8]

        self.assertEqual(sum(tensor.shape[1] for tensor in visible), 521)
        total_masked = torch.stack([mask.sum(dim=1) for mask in masks]).sum(dim=0)
        torch.testing.assert_close(total_masked, torch.full((batch,), 92.0))

    def test_impossible_all_masked_request_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least 4'):
            _visible_token_counts(
                self.LENGTHS, self.WEIGHTS, 1.0, min_mask_ratio=0.25
            )


if __name__ == '__main__':
    unittest.main()
