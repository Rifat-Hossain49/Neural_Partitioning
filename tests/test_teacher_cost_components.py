from __future__ import annotations

import unittest

import numpy as np

from realtrain.teacher import action_cost, action_cost_components


class TeacherCostComponentsTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(29)
        lower = rng.uniform(0.0, 0.8, size=(270, 2))
        upper = np.minimum(1.0, lower + rng.uniform(0.001, 0.08, size=(270, 2)))
        self.entries = np.column_stack(
            [lower[:, 0], lower[:, 1], upper[:, 0], upper[:, 1]]
        ).astype(np.float32)
        query_lower = rng.uniform(0.0, 0.9, size=(31, 2))
        query_upper = np.minimum(
            1.0, query_lower + rng.uniform(0.02, 0.2, size=(31, 2))
        )
        self.queries = np.column_stack(
            [
                query_lower[:, 0],
                query_lower[:, 1],
                query_upper[:, 0],
                query_upper[:, 1],
            ]
        ).astype(np.float32)

    def test_components_reconstruct_existing_action_cost(self) -> None:
        overlap_weight = 1e-4
        margin_weight = 2e-5
        for action_id in (0, 17, 42, 79):
            hit, overlap, margin = action_cost_components(
                self.entries, self.queries, action_id, 128
            )
            expected = action_cost(
                self.entries,
                self.queries,
                action_id,
                128,
                overlap_weight,
                margin_weight,
            )
            self.assertAlmostEqual(
                expected,
                hit + overlap_weight * overlap + margin_weight * margin,
                places=12,
            )

    def test_components_are_finite_and_nonnegative(self) -> None:
        for action_id in (0, 31, 79):
            components = action_cost_components(
                self.entries, self.queries, action_id, 128
            )
            self.assertTrue(np.isfinite(components).all())
            self.assertTrue(all(value >= 0.0 for value in components))


if __name__ == "__main__":
    unittest.main()
