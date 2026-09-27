from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.custom_dataset_benchmark import (
    load_spatial_data,
    make_workload,
    normalize_rectangles,
)


class CustomDatasetBenchmarkTests(unittest.TestCase):
    def test_csv_point_aliases_and_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.csv"
            path.write_text(
                "Longitude,Latitude,label\n90.0,23.0,a\n90.5,23.2,b\n91.0,24.0,c\n",
                encoding="utf-8",
            )
            values = load_spatial_data(path, synthetic_objects=0, seed=7)
        rectangles, metadata = normalize_rectangles(values)
        self.assertEqual(rectangles.shape, (3, 4))
        self.assertEqual(metadata["input_kind"], "points")
        self.assertTrue(np.all(rectangles >= 0.0))
        self.assertTrue(np.all(rectangles <= 1.0))
        self.assertTrue(np.all(rectangles[:, :2] <= rectangles[:, 2:]))

    def test_workload_is_reproducible(self) -> None:
        rng = np.random.default_rng(11)
        points = rng.uniform(size=(100, 2))
        rectangles, _ = normalize_rectangles(points)
        construction_a, suite_a = make_workload(rectangles, 30, 5, 7, 9, seed=19)
        construction_b, suite_b = make_workload(rectangles, 30, 5, 7, 9, seed=19)
        np.testing.assert_array_equal(construction_a, construction_b)
        np.testing.assert_array_equal(suite_a["points"], suite_b["points"])
        np.testing.assert_array_equal(suite_a["knn_points"], suite_b["knn_points"])
        for key in suite_a["ranges"]:
            np.testing.assert_array_equal(suite_a["ranges"][key], suite_b["ranges"][key])


if __name__ == "__main__":
    unittest.main()
