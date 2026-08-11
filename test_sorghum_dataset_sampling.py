"""Focused regression tests for point-cloud sampling and normalization."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from sorghum_dataset import SorghumDataset


class _FixedRng:
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)

    def choice(self, population, size, replace):
        del population, replace
        if size != len(self.indices):
            raise AssertionError(f'expected choice size {len(self.indices)}, got {size}')
        return self.indices.copy()


def _loader(num_points, deterministic):
    dataset = SorghumDataset.__new__(SorghumDataset)
    dataset.num_points = num_points
    dataset._deterministic_point_sampling = deterministic
    return dataset


def _normalise_full(points):
    points = points - points.mean(axis=0)
    radius = np.linalg.norm(points, axis=1).max()
    return points / radius if radius > 0 else points


class PointCloudSamplingTests(unittest.TestCase):

    def setUp(self):
        self.points = np.array([
            [0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 2.0],
            [2.0, 1.0, 0.5],
            [5.0, 3.0, 1.5],
        ], dtype=np.float64)
        self.read_patch = mock.patch(
            'sorghum_dataset.o3d.io.read_point_cloud',
            return_value=SimpleNamespace(points=self.points),
        )

    def test_full_cloud_is_normalized_before_subsampling(self):
        dataset = _loader(num_points=3, deterministic=False)
        selected = np.array([0, 2, 4])

        with self.read_patch, mock.patch.object(
            dataset, '_pointcloud_rng', return_value=_FixedRng(selected)
        ):
            actual = dataset.load_pointcloud(Path('/data/train/sample/cloud.ply'))

        expected = _normalise_full(self.points)[selected].astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        self.assertEqual(actual.dtype, np.float32)

    def test_full_cloud_is_normalized_before_padding(self):
        dataset = _loader(num_points=9, deterministic=False)
        padding_indices = np.array([0, 1, 1])

        with self.read_patch, mock.patch.object(
            dataset, '_pointcloud_rng', return_value=_FixedRng(padding_indices)
        ):
            actual = dataset.load_pointcloud(Path('/data/train/sample/cloud.ply'))

        normalized = _normalise_full(self.points).astype(np.float32)
        expected = np.vstack([normalized, normalized[padding_indices]])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)

    def test_validation_sampling_is_stable_per_file(self):
        dataset = _loader(num_points=3, deterministic=True)

        with self.read_patch:
            first = dataset.load_pointcloud(Path('/data/val/sample_a/cloud.ply'))
            second = dataset.load_pointcloud(Path('/data/val/sample_a/cloud.ply'))
            other = dataset.load_pointcloud(Path('/data/val/sample_b/cloud.ply'))

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))

    def test_training_sampling_remains_stochastic(self):
        dataset = _loader(num_points=3, deterministic=False)
        np.random.seed(1234)

        with self.read_patch:
            first = dataset.load_pointcloud(Path('/data/train/sample/cloud.ply'))
            second = dataset.load_pointcloud(Path('/data/train/sample/cloud.ply'))

        self.assertFalse(np.array_equal(first, second))

    def test_legacy_subclass_without_sampling_policy_does_not_crash(self):
        dataset = SorghumDataset.__new__(SorghumDataset)

        rng = dataset._pointcloud_rng(Path('/data/train/sample/cloud.ply'))

        self.assertIs(rng, np.random)


if __name__ == '__main__':
    unittest.main()
