"""Regression tests for training/validation mask-ratio plumbing."""

import unittest

import torch
import torch.nn as nn

from embodied_mae import earth_movers_distance
from train_sorghum_4m import evaluate, train_one_epoch


class _RecordingModel(nn.Module):
    patch_size = 1

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.mask_ratios = []
        self.last_pc_base_loss = torch.tensor(0.0)
        self.last_sinkhorn_loss = torch.tensor(0.0)

    def forward(self, rgb, depth, pc, params, text_valid, *, mask_ratio,
                compute_sinkhorn=True):
        del text_valid, compute_sinkhorn
        self.mask_ratios.append(float(mask_ratio))
        batch, _, height, width = rgb.shape
        pred_rgb = rgb.permute(0, 2, 3, 1).reshape(batch, height * width, 3)
        pred_depth = depth.permute(0, 2, 3, 1).reshape(batch, height * width, 1)
        pred_pc = pc + torch.rand_like(pc) * 1e-3
        pred_params = params.clone()

        mask_rgb = torch.ones(batch, height * width)
        mask_depth = torch.ones(batch, height * width)
        mask_pc = torch.ones(batch, 1)
        mask_text = (torch.rand(batch, params.shape[1]) > 0.5).float()
        mask_text[:, 0] = 1.0

        component = self.weight.square()
        self.last_pc_base_loss = component.detach()
        self.last_sinkhorn_loss = component.detach().new_zeros(())
        return (
            component,
            (component, component, component, component),
            (pred_rgb, pred_depth, pred_pc, pred_params),
            (mask_rgb, mask_depth, mask_pc, mask_text),
        )


def _batch():
    generator = torch.Generator().manual_seed(7)
    batch = 2
    return (
        torch.randn(batch, 3, 2, 2, generator=generator),
        torch.randn(batch, 1, 2, 2, generator=generator),
        torch.randn(batch, 8, 3, generator=generator),
        torch.rand(batch, 3, 9, generator=generator),
        torch.ones(batch, 3),
        ('sample-a', 'sample-b'),
    )


class TrainingMaskConfigTests(unittest.TestCase):

    def test_train_forwards_the_configured_mask_ratio(self):
        model = _RecordingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        train_one_epoch(
            model, [_batch()], optimizer, torch.device('cpu'), epoch=1,
            mask_ratio=0.15,
        )

        self.assertEqual(model.mask_ratios, [0.15])

    def test_evaluation_is_repeatable_and_restores_rng_state(self):
        model = _RecordingModel()
        dataloader = [_batch()]

        torch.manual_seed(123)
        state_before = torch.random.get_rng_state()
        expected_next = torch.rand(4)
        torch.random.set_rng_state(state_before)

        _, first = evaluate(
            model, dataloader, torch.device('cpu'), mask_ratio=0.15,
            val_mask_seed=42, pc_metric_thresholds=(0.01,),
        )
        actual_next = torch.rand(4)
        torch.testing.assert_close(actual_next, expected_next)

        torch.manual_seed(999)
        _, second = evaluate(
            model, dataloader, torch.device('cpu'), mask_ratio=0.15,
            val_mask_seed=42, pc_metric_thresholds=(0.01,),
        )

        for key in ('pc_chamfer', 'pc_precision@0.01',
                    'pc_recall@0.01', 'pc_f1@0.01'):
            self.assertEqual(first[key], second[key])
        self.assertEqual(model.mask_ratios, [0.15, 0.15])

    def test_emd_subsampling_does_not_consume_global_rng(self):
        generator = torch.Generator().manual_seed(7)
        pred = torch.randn(1, 12, 3, generator=generator)
        target = torch.randn(1, 12, 3, generator=generator)

        torch.manual_seed(123)
        state_before = torch.random.get_rng_state()
        first = earth_movers_distance(pred, target, num_samples=4)
        state_after = torch.random.get_rng_state()
        second = earth_movers_distance(pred, target, num_samples=4)

        self.assertTrue(torch.equal(state_after, state_before))
        torch.testing.assert_close(first, second)


if __name__ == '__main__':
    unittest.main()
