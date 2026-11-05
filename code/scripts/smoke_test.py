from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realtrain.actions import ACTION_COUNT, page_count  # noqa: E402
from realtrain.features import state_feature  # noqa: E402
from realtrain.train import load_model  # noqa: E402
from realtrain.tree import build_neural_tree  # noqa: E402


def rectangles(rng: np.random.Generator, count: int) -> np.ndarray:
    low = rng.uniform(0.0, 0.98, size=(count, 2))
    size = rng.uniform(0.001, 0.02, size=(count, 2))
    high = np.minimum(low + size, 1.0)
    return np.column_stack([low, high]).astype(np.float32)


def queries(rng: np.random.Generator, count: int = 256) -> np.ndarray:
    low = rng.uniform(0.0, 0.85, size=(count, 2))
    size = rng.uniform(0.02, 0.15, size=(count, 2))
    high = np.minimum(low + size, 1.0)
    return np.column_stack([low, high]).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(20260903)
    query_boxes = queries(rng)
    model = load_model(ROOT / "artifacts" / "model" / "selected.pt", device="cpu")
    assert model.input_dim == 498
    assert model.action_count == ACTION_COUNT == 80

    probe = rectangles(rng, 300)
    feature = state_feature(probe, query_boxes[:96], 128, 12, 0)
    assert feature.shape == (498,)

    for capacity in (128, 256, 512):
        rects = rectangles(rng, 2 * capacity + 17)
        _, diagnostics = build_neural_tree(
            rects,
            query_boxes,
            model,
            "cpu",
            capacity,
            hist_bins=12,
            query_cap=96,
        )
        assert diagnostics["validation"]["passed"]
        assert diagnostics["leaf_pages"] == page_count(len(rects), capacity)
        assert diagnostics["neural_decision_count"] > 0
        print(
            f"B={capacity}: {diagnostics['leaf_pages']} leaves, "
            f"height={diagnostics['validation']['height']}, valid"
        )

    print("WAHARP frozen-model smoke test passed")


if __name__ == "__main__":
    main()
