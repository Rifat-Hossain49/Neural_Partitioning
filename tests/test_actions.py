from __future__ import annotations

import unittest

import numpy as np

from realtrain.actions import ACTION_COUNT, ACTION_TABLE, page_count, split_positions


class ActionSpaceTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        low = rng.uniform(0.0, 0.95, size=(385, 2))
        high = np.minimum(low + 0.01, 1.0)
        self.rects = np.column_stack([low, high]).astype(np.float32)

    def test_action_table_has_expected_size(self) -> None:
        self.assertEqual(ACTION_COUNT, 80)
        self.assertEqual(len(set(ACTION_TABLE)), 80)

    def test_every_action_preserves_entries_and_feasible_pages(self) -> None:
        capacity = 128
        expected_pages = page_count(len(self.rects), capacity)
        for action_id in range(ACTION_COUNT):
            left, right = split_positions(self.rects, action_id, capacity)
            merged = np.concatenate([left, right])
            self.assertEqual(len(merged), len(self.rects))
            self.assertEqual(len(np.unique(merged)), len(self.rects))
            self.assertGreaterEqual(page_count(len(left), capacity), 1)
            self.assertGreaterEqual(page_count(len(right), capacity), 1)
            self.assertEqual(
                page_count(len(left), capacity) + page_count(len(right), capacity),
                expected_pages,
            )


if __name__ == "__main__":
    unittest.main()
