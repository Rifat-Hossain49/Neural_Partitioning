from __future__ import annotations

import math
import numpy as np

from .actions import page_count
from .geometry import bbox, centers, intersect_matrix, normalize_queries_local, normalize_rects_local, rect_area


def _safe_stats(x: np.ndarray) -> list[float]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(a) == 0:
        return [0.0] * 7
    return [
        float(np.mean(a)), float(np.std(a)), float(np.min(a)), float(np.quantile(a, 0.25)),
        float(np.median(a)), float(np.quantile(a, 0.75)), float(np.max(a)),
    ]


def local_queries_for_state(entries: np.ndarray, construction_queries: np.ndarray, cap: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rb = bbox(entries)
    q = np.asarray(construction_queries, dtype=np.float32)
    mask = intersect_matrix(q, rb.reshape(1, 4))[:, 0]
    hit = q[mask]
    if len(hit) == 0:
        qc = np.column_stack(((q[:, 0] + q[:, 2]) * 0.5, (q[:, 1] + q[:, 3]) * 0.5))
        rc = np.asarray([(rb[0] + rb[2]) * 0.5, (rb[1] + rb[3]) * 0.5])
        d = np.sum((qc - rc[None, :]) ** 2, axis=1)
        hit = q[np.argsort(d)[: min(cap, len(q))]]
    elif len(hit) > cap:
        if rng is None:
            # Deterministic evenly-spaced subsample for deployment.
            idx = np.linspace(0, len(hit) - 1, cap, dtype=np.int64)
        else:
            idx = rng.choice(len(hit), size=cap, replace=False)
        hit = hit[idx]
    return hit.astype(np.float32, copy=False)


def state_feature(entries: np.ndarray, queries: np.ndarray, capacity: int, hist_bins: int, level: int) -> np.ndarray:
    nr, rb = normalize_rects_local(entries)
    nq = normalize_queries_local(queries, rb) if len(queries) else np.empty((0, 4), dtype=np.float32)
    c = centers(nr)
    h_obj, _, _ = np.histogram2d(c[:, 0], c[:, 1], bins=hist_bins, range=((0, 1), (0, 1)))
    h_obj = h_obj.astype(np.float32) / max(len(c), 1)
    if len(nq):
        qc = np.column_stack(((nq[:, 0] + nq[:, 2]) * 0.5, (nq[:, 1] + nq[:, 3]) * 0.5))
        qarea = np.maximum(0, nq[:, 2] - nq[:, 0]) * np.maximum(0, nq[:, 3] - nq[:, 1])
        h_q, _, _ = np.histogram2d(qc[:, 0], qc[:, 1], bins=hist_bins, range=((0, 1), (0, 1)))
        h_qa, _, _ = np.histogram2d(qc[:, 0], qc[:, 1], bins=hist_bins, range=((0, 1), (0, 1)), weights=qarea + 1e-6)
        h_q = h_q.astype(np.float32) / max(len(qc), 1)
        h_qa = h_qa.astype(np.float32) / max(float(np.sum(qarea + 1e-6)), 1e-9)
    else:
        h_q = np.zeros((hist_bins, hist_bins), dtype=np.float32)
        h_qa = np.zeros_like(h_q)
        qc = np.empty((0, 2), dtype=np.float32)
        qarea = np.empty(0, dtype=np.float32)
    widths = nr[:, 2] - nr[:, 0]
    heights = nr[:, 3] - nr[:, 1]
    areas = rect_area(nr)
    cov = np.cov(c.T) if len(c) > 1 else np.zeros((2, 2), dtype=np.float64)
    eig = np.linalg.eigvalsh(cov) if np.isfinite(cov).all() else np.zeros(2)
    stats = []
    stats += _safe_stats(widths) + _safe_stats(heights) + _safe_stats(areas)
    stats += _safe_stats(c[:, 0]) + _safe_stats(c[:, 1])
    stats += [float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1]), float(eig[0]), float(eig[-1])]
    if len(nq):
        qw = nq[:, 2] - nq[:, 0]; qh = nq[:, 3] - nq[:, 1]
        stats += _safe_stats(qw) + _safe_stats(qh) + _safe_stats(qarea)
    else:
        stats += [0.0] * 21
    pages = page_count(len(entries), capacity)
    stats += [
        math.log1p(len(entries)) / 16.0,
        math.log1p(pages) / 8.0,
        float(level) / 4.0,
        float(len(queries)) / 128.0,
        float((rb[2] - rb[0]) / max(rb[3] - rb[1], 1e-9)),
    ]
    return np.concatenate([h_obj.ravel(), h_q.ravel(), h_qa.ravel(), np.asarray(stats, dtype=np.float32)]).astype(np.float32)
