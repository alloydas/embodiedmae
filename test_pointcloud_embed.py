"""Regression tests for stable, spatially-aware point-cloud tokens."""

import unittest

import torch

from embodied_mae import PointCloudEmbed


class PointCloudEmbedTests(unittest.TestCase):

    def setUp(self):
        generator = torch.Generator().manual_seed(123)
        self.xyz = torch.randn(2, 32, 3, generator=generator)

    def test_deterministic_fps_does_not_depend_on_rng_state(self):
        embed = PointCloudEmbed(
            num_tokens=8, group_size=4, embed_dim=12,
            deterministic_fps=True,
        )

        torch.manual_seed(1)
        first = embed.fps(self.xyz, 8)
        torch.manual_seed(999)
        second = embed.fps(self.xyz, 8)

        torch.testing.assert_close(first, second)

    def test_deterministic_fps_centers_survive_input_permutation(self):
        embed = PointCloudEmbed(
            num_tokens=8, group_size=4, embed_dim=12,
            deterministic_fps=True,
        )
        permutation = torch.randperm(self.xyz.shape[1], generator=torch.Generator().manual_seed(7))
        permuted = self.xyz[:, permutation]

        original_idx = embed.fps(self.xyz, 8)
        permuted_idx = embed.fps(permuted, 8)
        batch = torch.arange(self.xyz.shape[0])[:, None]
        original_centers = self.xyz[batch, original_idx]
        permuted_centers = permuted[batch, permuted_idx]

        torch.testing.assert_close(original_centers, permuted_centers)

    def test_center_coordinates_are_added_without_shape_change(self):
        embed = PointCloudEmbed(
            num_tokens=8, group_size=4, embed_dim=12,
            deterministic_fps=True,
            add_center_coordinates=True,
        ).eval()
        with torch.no_grad():
            for parameter in embed.parameters():
                parameter.zero_()

        tokens = embed(self.xyz)
        fps_idx = embed.fps(self.xyz, 8)
        batch = torch.arange(self.xyz.shape[0])[:, None]
        centers = self.xyz[batch, fps_idx]

        self.assertEqual(tokens.shape, (2, 8, 12))
        self.assertEqual(tokens.dtype, self.xyz.dtype)
        torch.testing.assert_close(tokens[..., :3], centers)
        torch.testing.assert_close(tokens[..., 3:], torch.zeros_like(tokens[..., 3:]))

    def test_feature_flags_do_not_change_state_dict_schema(self):
        legacy = PointCloudEmbed(
            num_tokens=8, group_size=4, embed_dim=12,
            deterministic_fps=False,
            add_center_coordinates=False,
        )
        corrected = PointCloudEmbed(
            num_tokens=8, group_size=4, embed_dim=12,
            deterministic_fps=True,
            add_center_coordinates=True,
        )

        self.assertEqual(set(legacy.state_dict()), set(corrected.state_dict()))
        corrected.load_state_dict(legacy.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
