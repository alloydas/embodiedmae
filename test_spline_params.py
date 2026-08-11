"""Regression tests for spline-parameter YAML loading."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from embodied_mae_4m import load_spline_params


def _valid_leaf(index=1):
    return {
        'Leaf Index': index,
        'starting_point': 0.5,
        'length': 0.25,
        'roll_angle': 180.0,
        'branching_angle': 90.0,
        'waviness_frequency': 0.05,
        'waviness_period_start': [90.0, 180.0],
    }


def _spline_document(leaves):
    return {
        'Sorghums': [{
            'Parameters': {
                'stem_length': 1.5,
                'stem_direction': [0.0, 1.0, -1.0],
                'panicle_size': [0.25, 0.5, 0.75],
                'panicle_seed_amount': 25,
                'panicle_seed_radius': 0.01,
            },
            'Leaves': leaves,
        }],
    }


class LoadSplineParamsTests(unittest.TestCase):

    def _load(self, document, max_leaves):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'sample_spline.yml'
            with path.open('w') as handle:
                yaml.safe_dump(document, handle)
            return load_spline_params(path, max_leaves=max_leaves)

    def test_geometry_only_leaf_is_filtered_before_padding(self):
        geometry_only_leaf = {
            'Leaf Index': 2,
            'Center Points': [[0.0, 0.0, 0.0]],
            'Left Points': [[0.0, 0.0, 0.0]],
            'Right Points': [[0.0, 0.0, 0.0]],
        }

        valid, params = self._load(
            _spline_document([geometry_only_leaf, _valid_leaf()]),
            max_leaves=2,
        )

        np.testing.assert_array_equal(valid.numpy(), [1.0, 1.0, 0.0])
        np.testing.assert_allclose(
            params[1].numpy(),
            [0.5, 0.25, 0.5, 0.5, 0.5, 0.25, 0.5, 0.0, 0.0],
        )
        np.testing.assert_array_equal(params[2].numpy(), np.zeros(9))

    def test_partially_parameterized_leaf_is_also_filtered(self):
        partial_leaf = _valid_leaf(index=7)
        partial_leaf.pop('waviness_period_start')

        valid, params = self._load(
            _spline_document([partial_leaf, _valid_leaf(index=8)]),
            max_leaves=2,
        )

        np.testing.assert_array_equal(valid.numpy(), [1.0, 1.0, 0.0])
        np.testing.assert_allclose(
            params[1].numpy(),
            [0.5, 0.25, 0.5, 0.5, 0.5, 0.25, 0.5, 0.0, 0.0],
        )
        np.testing.assert_array_equal(params[2].numpy(), np.zeros(9))


if __name__ == '__main__':
    unittest.main()
