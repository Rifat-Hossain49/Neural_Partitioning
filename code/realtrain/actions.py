from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ANGLES_DEG = tuple(float(x) for x in np.arange(0.0, 180.0, 11.25))
PAGE_FRACTIONS = (0.25, 0.375, 0.5, 0.625, 0.75)
ACTION_TABLE = tuple((angle, frac) for angle in ANGLES_DEG for frac in PAGE_FRACTIONS)
ACTION_COUNT = len(ACTION_TABLE)
HALF_ACTION_IDS = tuple(i for i, (_, f) in enumerate(ACTION_TABLE) if abs(f - 0.5) < 1e-12)


def action_params(action_id: int) -> tuple[float, float]:
    return ACTION_TABLE[int(action_id)]


def page_count(n: int, capacity: int) -> int:
    return (int(n) + int(capacity) - 1) // int(capacity)


def left_page_count(total_pages: int, fraction: float) -> int:
    if total_pages < 2:
        raise ValueError("split requires at least two pages")
    p = int(round(total_pages * float(fraction)))
    return min(total_pages - 1, max(1, p))


def feasible_left_count(n: int, total_pages: int, left_pages: int, capacity: int) -> int:
    right_pages = total_pages - left_pages
    lower = max((left_pages - 1) * capacity + 1, n - right_pages * capacity)
    upper = min(left_pages * capacity, n - ((right_pages - 1) * capacity + 1))
    if lower > upper:
        raise RuntimeError(f"no feasible quota n={n} pages={total_pages} left_pages={left_pages} B={capacity}")
    target = int(round(n * left_pages / total_pages))
    return min(upper, max(lower, target))


def split_positions(rects: np.ndarray, action_id: int, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(rects, dtype=np.float32)
    n = len(r)
    pages = page_count(n, capacity)
    if pages < 2:
        raise ValueError("split_positions called on one-page state")
    angle, frac = action_params(action_id)
    lp = left_page_count(pages, frac)
    quota = feasible_left_count(n, pages, lp, capacity)
    theta = math.radians(angle)
    c = np.column_stack(((r[:, 0] + r[:, 2]) * 0.5, (r[:, 1] + r[:, 3]) * 0.5))
    proj = c[:, 0] * math.cos(theta) + c[:, 1] * math.sin(theta)
    order = np.argsort(proj, kind="mergesort")
    return order[:quota].astype(np.int64), order[quota:].astype(np.int64)


def deterministic_rollout_groups(rects: np.ndarray, capacity: int) -> list[np.ndarray]:
    r = np.asarray(rects, dtype=np.float32)
    out: list[np.ndarray] = []

    def rec(pos: np.ndarray):
        if len(pos) <= capacity:
            out.append(pos.copy())
            return
        sub = r[pos]
        pages = page_count(len(pos), capacity)
        lp = pages // 2
        quota = feasible_left_count(len(pos), pages, lp, capacity)
        c = np.column_stack(((sub[:, 0] + sub[:, 2]) * 0.5, (sub[:, 1] + sub[:, 3]) * 0.5))
        span = np.ptp(c, axis=0)
        axis = int(np.argmax(span))
        order_local = np.argsort(c[:, axis], kind="mergesort")
        rec(pos[order_local[:quota]])
        rec(pos[order_local[quota:]])

    rec(np.arange(len(r), dtype=np.int64))
    expected = page_count(len(r), capacity)
    if len(out) != expected:
        raise RuntimeError(f"rollout page count mismatch {len(out)} != {expected}")
    if max(map(len, out), default=0) > capacity:
        raise RuntimeError("rollout overflow")
    return out


def candidate_action_ids(rng: np.random.Generator, count: int) -> np.ndarray:
    mandatory = np.asarray(HALF_ACTION_IDS, dtype=np.int64)
    if count <= len(mandatory):
        return mandatory[:count]
    others = np.asarray([i for i in range(ACTION_COUNT) if i not in set(HALF_ACTION_IDS)], dtype=np.int64)
    extra_n = min(count - len(mandatory), len(others))
    extra = rng.choice(others, size=extra_n, replace=False)
    return np.concatenate([mandatory, np.asarray(extra, dtype=np.int64)])
