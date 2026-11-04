from __future__ import annotations

import unittest

import numpy as np

from realtrain.actions import (
    ACTION_COUNT,
    ACTION_TABLE,
    feasible_left_count,
    left_page_count,
    page_count,
    split_positions,
)


class ActionSpaceTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        low = rng.uniform(0.0, 0.95, size=(385, 2))
        high = np.minimum(low + 0.01, 1.0)
        self.rects = np.column_stack([low, high]).astype(np.float32)

    def test_action_table_has_expected_size(self) -> None:
        self.assertEqual(ACTION_COUNT, 80)
        self.assertEqual(len(set(ACTION_TABLE)), 80)

    def test_page_quota_rounds_half_to_even_then_clamps(self) -> None:
        self.assertEqual(left_page_count(10, 0.25), 2)
        self.assertEqual(left_page_count(14, 0.25), 4)
        self.assertEqual(left_page_count(2, 0.25), 1)
        self.assertEqual(left_page_count(2, 0.75), 1)

    def test_entry_quota_rounds_half_to_even_within_feasible_interval(self) -> None:
        self.assertEqual(feasible_left_count(5, 2, 1, 4), 2)
        self.assertEqual(feasible_left_count(7, 2, 1, 4), 4)
        self.assertEqual(feasible_left_count(17, 5, 1, 4), 3)
        self.assertEqual(feasible_left_count(17, 5, 4, 4), 14)

    def test_feasible_quota_preserves_each_child_page_count(self) -> None:
        for capacity in (4, 8, 128):
            for n in range(capacity + 1, 8 * capacity + 1):
                pages = page_count(n, capacity)
                for left_pages in range(1, pages):
                    count = feasible_left_count(n, pages, left_pages, capacity)
                    self.assertEqual(page_count(count, capacity), left_pages)
                    self.assertEqual(page_count(n - count, capacity), pages - left_pages)

    def test_equal_projections_preserve_current_input_order(self) -> None:
        rects = np.tile(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32), (7, 1))
        for action_id in range(ACTION_COUNT):
            left, right = split_positions(rects, action_id, capacity=4)
            np.testing.assert_array_equal(left, np.arange(4))
            np.testing.assert_array_equal(right, np.arange(4, 7))

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
