from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import centers
from .utils import atomic_json, derive_seed, sha256_array

RANGE_SIDES = (0.001, 0.005, 0.01, 0.05, 0.10)
ASPECTS = (1/16, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0, 8.0, 16.0)
KNN_K = (1, 5, 25, 125, 625)


def _boxes_for_shape(rng: np.random.Generator, count: int, width: float, height: float) -> np.ndarray:
    width = min(max(float(width), 1e-8), 1.0)
    height = min(max(float(height), 1e-8), 1.0)
    x0 = rng.random(count) * max(1.0 - width, 0.0)
    y0 = rng.random(count) * max(1.0 - height, 0.0)
    return np.column_stack([x0, y0, x0 + width, y0 + height]).astype(np.float32)


def make_range_conditions(per_condition: int, seed: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    specs: list[dict[str, Any]] = []
    for side in RANGE_SIDES:
        name = f"square_side_{side:g}"
        arr = _boxes_for_shape(rng, per_condition, side, side)
        out[name] = arr
        specs.append({"name": name, "query_type": "range", "count": per_condition, "side": side})
    area = 0.01
    for ar in ASPECTS:
        width = math.sqrt(area * ar)
        height = math.sqrt(area / ar)
        scale = max(width, height, 1.0)
        if scale > 1.0:
            width /= scale; height /= scale
        name = f"aspect_{ar:g}"
        arr = _boxes_for_shape(rng, per_condition, width, height)
        out[name] = arr
        specs.append({"name": name, "query_type": "range", "count": per_condition, "aspect": ar, "area": float(width*height)})
    return out, specs


def make_points(rects: np.ndarray, count: int, seed: int) -> np.ndarray:
    c = centers(rects)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), size=count)
    return c[idx].astype(np.float32)


def estimate_knn_surrogate_boxes(rects: np.ndarray, query_points: np.ndarray, ks: np.ndarray, seed: int, sample_cap: int = 100_000) -> np.ndarray:
    c = centers(rects).astype(np.float64)
    q = np.asarray(query_points, dtype=np.float64)
    ks = np.asarray(ks, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if len(c) > sample_cap:
        sample_idx = rng.choice(len(c), size=sample_cap, replace=False)
        sample = c[sample_idx]
        scale = len(c) / len(sample)
        k_sample = np.maximum(1, np.ceil(ks / scale).astype(int))
    else:
        sample = c
        k_sample = ks.copy()
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(sample)
        radii = np.empty(len(q), dtype=np.float64)
        for kval in np.unique(k_sample):
            mask = k_sample == kval
            d, _ = tree.query(q[mask], k=int(kval))
            if int(kval) == 1:
                radii[mask] = np.asarray(d, dtype=np.float64)
            else:
                radii[mask] = np.asarray(d, dtype=np.float64)[:, -1]
    except Exception:
        density = max(len(sample), 1)
        radii = np.sqrt(np.maximum(k_sample, 1) / (math.pi * density))
    radii = np.clip(radii, 1e-6, 0.25)
    return np.column_stack([
        np.clip(q[:, 0] - radii, 0.0, 1.0),
        np.clip(q[:, 1] - radii, 0.0, 1.0),
        np.clip(q[:, 0] + radii, 0.0, 1.0),
        np.clip(q[:, 1] + radii, 0.0, 1.0),
    ]).astype(np.float32)


def make_construction_boxes(rects: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    # Match the 20 evaluation conditions: 14/20 range, 1/20 point, 5/20 kNN.
    range_count = int(round(count * 0.70))
    point_count = int(round(count * 0.05))
    knn_count = count - range_count - point_count
    rng = np.random.default_rng(seed)
    # Draw range shapes from the same condition family but not from held-out arrays.
    shapes = []
    for _ in range(range_count):
        if rng.random() < 0.45:
            side = float(rng.choice(np.asarray(RANGE_SIDES)))
            shapes.append(_boxes_for_shape(rng, 1, side, side)[0])
        else:
            ar = float(rng.choice(np.asarray(ASPECTS)))
            area = 0.01
            w = math.sqrt(area * ar); h = math.sqrt(area / ar)
            scale = max(w, h, 1.0)
            shapes.append(_boxes_for_shape(rng, 1, w/scale, h/scale)[0])
    range_boxes = np.asarray(shapes, dtype=np.float32)
    p = make_points(rects, point_count + knn_count, seed + 1)
    point_boxes = np.column_stack([p[:point_count, 0], p[:point_count, 1], p[:point_count, 0], p[:point_count, 1]]).astype(np.float32)
    knn_points = p[point_count:]
    k_values = rng.choice(np.asarray(KNN_K, dtype=np.int64), size=knn_count)
    knn_boxes = estimate_knn_surrogate_boxes(rects, knn_points, k_values, seed + 2)
    boxes = np.concatenate([range_boxes, point_boxes, knn_boxes], axis=0)
    perm = rng.permutation(len(boxes))
    boxes = boxes[perm]
    return boxes, {
        "count": int(len(boxes)), "range_count": range_count, "point_count": point_count,
        "knn_surrogate_count": knn_count, "knn_k_values": list(map(int, KNN_K)),
        "sha256": sha256_array(boxes),
    }


def create_dataset_workloads(rects: np.ndarray, dataset: str, root: Path, namespace: str, seed: int,
                             construction_count: int, val_range_n: int, val_point_n: int, val_knn_n: int,
                             final_range_n: int, final_point_n: int, final_knn_n: int) -> dict[str, Any]:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "WORKLOAD_MANIFEST.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    def ds(label):
        return derive_seed(namespace, seed, f"{dataset}|{label}")
    construction, construction_meta = make_construction_boxes(rects, construction_count, ds("construction"))
    np.save(root / "construction_boxes.npy", construction)
    val_ranges, val_specs = make_range_conditions(val_range_n, ds("validation_ranges"))
    final_ranges, final_specs = make_range_conditions(final_range_n, ds("final_ranges"))
    for name, arr in val_ranges.items(): np.save(root / f"validation_range_{name}.npy", arr)
    for name, arr in final_ranges.items(): np.save(root / f"final_range_{name}.npy", arr)
    val_points = make_points(rects, val_point_n, ds("validation_points")); np.save(root / "validation_points.npy", val_points)
    final_points = make_points(rects, final_point_n, ds("final_points")); np.save(root / "final_points.npy", final_points)
    val_knn = make_points(rects, val_knn_n, ds("validation_knn")); np.save(root / "validation_knn_points.npy", val_knn)
    final_knn = make_points(rects, final_knn_n, ds("final_knn")); np.save(root / "final_knn_points.npy", final_knn)
    manifest = {
        "dataset": dataset, "namespace": namespace, "seed": int(seed),
        "construction": construction_meta,
        "validation_range_specs": val_specs, "final_range_specs": final_specs,
        "validation_points": {"count": len(val_points), "sha256": sha256_array(val_points)},
        "final_points": {"count": len(final_points), "sha256": sha256_array(final_points)},
        "validation_knn": {"count_per_k": len(val_knn), "k_values": list(KNN_K), "sha256": sha256_array(val_knn)},
        "final_knn": {"count_per_k": len(final_knn), "k_values": list(KNN_K), "sha256": sha256_array(final_knn)},
        "leakage_rule": "construction boxes only may influence training/tree construction; validation selects checkpoint; final is evaluation only",
    }
    atomic_json(manifest_path, manifest)
    return manifest


def load_query_suite(root: Path, split: str) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads((root / "WORKLOAD_MANIFEST.json").read_text(encoding="utf-8"))
    specs = manifest[f"{split}_range_specs"]
    ranges = {s["name"]: np.load(root / f"{split}_range_{s['name']}.npy") for s in specs}
    points = np.load(root / f"{split}_points.npy")
    knn = np.load(root / f"{split}_knn_points.npy")
    return {"ranges": ranges, "points": points, "knn_points": knn, "manifest": manifest}
