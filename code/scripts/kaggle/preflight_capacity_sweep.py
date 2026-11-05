from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_SHA256 = "8709a7f9025292636f0526ccf7b702eda8ce0bce953c02cc1c05245bb13d7e1e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    sys.path.insert(0, str(bundle))

    from realtrain.actions import ACTION_COUNT, deterministic_rollout_groups, page_count
    from realtrain.features import state_feature
    from realtrain.train import load_model
    from realtrain.tree import build_neural_tree
    from realtrain.utils import sha256_file

    checkpoint = bundle / "frozen_model" / "member_2.pt"
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("frozen checkpoint hash mismatch")
    model = load_model(checkpoint, "cpu")
    rng = np.random.default_rng(20260901)
    results = []
    for capacity in (256, 512):
        n = capacity * 5 + 17
        points = rng.random((n, 2), dtype=np.float32)
        sizes = rng.random((n, 2), dtype=np.float32) * 0.002
        rects = np.column_stack((points, np.minimum(1.0, points + sizes))).astype(np.float32)
        queries = rng.random((128, 4), dtype=np.float32)
        queries[:, 2:] = np.maximum(queries[:, :2], queries[:, 2:])
        groups = deterministic_rollout_groups(rects, capacity)
        expected = page_count(n, capacity)
        if len(groups) != expected or max(map(len, groups)) > capacity:
            raise RuntimeError(f"capacity packing failed for B={capacity}")
        feature = state_feature(rects, queries, capacity, 12, 0)
        with torch.no_grad():
            logits = model(torch.from_numpy(feature[None, :]))
        if tuple(logits.shape) != (1, ACTION_COUNT):
            raise RuntimeError(f"model output shape failed for B={capacity}: {tuple(logits.shape)}")
        tree, build = build_neural_tree(rects, queries, model, "cpu", capacity, 12, 128)
        if not build["validation"]["passed"]:
            raise RuntimeError(f"full neural-tree validation failed for B={capacity}")
        results.append({
            "capacity": capacity,
            "objects": n,
            "pages": expected,
            "max_group_size": max(map(len, groups)),
            "feature_size": int(feature.size),
            "actions": ACTION_COUNT,
            "neural_decisions": int(build["neural_decision_count"]),
            "tree_height": int(build["validation"]["height"]),
            "platon_nominal_capacity_at_0.8": int(round(capacity / 0.8)),
        })
    print(json.dumps({"status": "PASS", "results": results}, indent=2))


if __name__ == "__main__":
    main()
