from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def centers(rects: np.ndarray) -> np.ndarray:
    r = np.asarray(rects, dtype=np.float32)
    return np.column_stack(((r[:, 0] + r[:, 2]) * 0.5, (r[:, 1] + r[:, 3]) * 0.5)).astype(np.float32)


def bbox(rects: np.ndarray) -> np.ndarray:
    r = np.asarray(rects, dtype=np.float32)
    if len(r) == 0:
        return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray([r[:, 0].min(), r[:, 1].min(), r[:, 2].max(), r[:, 3].max()], dtype=np.float32)


def bboxes_for_groups(rects: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    out = np.empty((len(groups), 4), dtype=np.float32)
    for i, g in enumerate(groups):
        out[i] = bbox(rects[np.asarray(g, dtype=np.int64)])
    return out


def intersect_matrix(queries: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    q = np.asarray(queries, dtype=np.float32)
    b = np.asarray(boxes, dtype=np.float32)
    if len(q) == 0 or len(b) == 0:
        return np.zeros((len(q), len(b)), dtype=bool)
    return (
        (q[:, None, 0] <= b[None, :, 2])
        & (q[:, None, 2] >= b[None, :, 0])
        & (q[:, None, 1] <= b[None, :, 3])
        & (q[:, None, 3] >= b[None, :, 1])
    )


def intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return bool(a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1])


def rect_area(rects: np.ndarray) -> np.ndarray:
    r = np.asarray(rects, dtype=np.float32)
    return np.maximum(0.0, r[:, 2] - r[:, 0]) * np.maximum(0.0, r[:, 3] - r[:, 1])


def rect_margin(rects: np.ndarray) -> np.ndarray:
    r = np.asarray(rects, dtype=np.float32)
    return 2.0 * (np.maximum(0.0, r[:, 2] - r[:, 0]) + np.maximum(0.0, r[:, 3] - r[:, 1]))


def normalized_overlap_pair_area(boxes: np.ndarray, root_box: np.ndarray | None = None) -> float:
    b = np.asarray(boxes, dtype=np.float32)
    if len(b) < 2:
        return 0.0
    total = 0.0
    for i in range(len(b)):
        x0 = np.maximum(b[i + 1 :, 0], b[i, 0])
        y0 = np.maximum(b[i + 1 :, 1], b[i, 1])
        x1 = np.minimum(b[i + 1 :, 2], b[i, 2])
        y1 = np.minimum(b[i + 1 :, 3], b[i, 3])
        total += float(np.sum(np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)))
    rb = bbox(b) if root_box is None else np.asarray(root_box, dtype=np.float32)
    denom = max(float((rb[2] - rb[0]) * (rb[3] - rb[1])), 1e-12)
    return total / denom


def min_dist_point_rect(point: np.ndarray, rects: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=np.float64)
    r = np.asarray(rects, dtype=np.float64)
    dx = np.maximum(np.maximum(r[:, 0] - p[0], 0.0), p[0] - r[:, 2])
    dy = np.maximum(np.maximum(r[:, 1] - p[1], 0.0), p[1] - r[:, 3])
    return np.hypot(dx, dy)


def morton_codes(points: np.ndarray, bits: int = 16) -> np.ndarray:
    p = np.clip(np.asarray(points, dtype=np.float64), 0.0, 1.0)
    scale = (1 << bits) - 1
    x = np.floor(p[:, 0] * scale + 0.5).astype(np.uint32)
    y = np.floor(p[:, 1] * scale + 0.5).astype(np.uint32)

    def part1by1(v):
        v = v & np.uint32(0x0000FFFF)
        v = (v | (v << np.uint32(8))) & np.uint32(0x00FF00FF)
        v = (v | (v << np.uint32(4))) & np.uint32(0x0F0F0F0F)
        v = (v | (v << np.uint32(2))) & np.uint32(0x33333333)
        v = (v | (v << np.uint32(1))) & np.uint32(0x55555555)
        return v

    return (part1by1(y).astype(np.uint64) << np.uint64(1)) | part1by1(x).astype(np.uint64)


def normalize_rects_local(rects: np.ndarray, root_box: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(rects, dtype=np.float32)
    rb = bbox(r) if root_box is None else np.asarray(root_box, dtype=np.float32)
    dx = max(float(rb[2] - rb[0]), 1e-9)
    dy = max(float(rb[3] - rb[1]), 1e-9)
    out = r.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - rb[0]) / dx
    out[:, [1, 3]] = (out[:, [1, 3]] - rb[1]) / dy
    return np.clip(out, 0.0, 1.0), rb


def normalize_queries_local(queries: np.ndarray, root_box: np.ndarray) -> np.ndarray:
    q = np.asarray(queries, dtype=np.float32).copy()
    rb = np.asarray(root_box, dtype=np.float32)
    dx = max(float(rb[2] - rb[0]), 1e-9)
    dy = max(float(rb[3] - rb[1]), 1e-9)
    q[:, [0, 2]] = (q[:, [0, 2]] - rb[0]) / dx
    q[:, [1, 3]] = (q[:, [1, 3]] - rb[1]) / dy
    return np.clip(q, 0.0, 1.0)
