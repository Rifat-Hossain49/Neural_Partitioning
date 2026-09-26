from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .actions import ACTION_COUNT, candidate_action_ids, deterministic_rollout_groups, page_count, split_positions
from .features import local_queries_for_state, state_feature
from .geometry import bbox, bboxes_for_groups, intersect_matrix, normalized_overlap_pair_area, rect_margin, morton_codes, centers
from .utils import atomic_json, sha256_array


def action_cost_components(entries: np.ndarray, queries: np.ndarray, action_id: int,
                           capacity: int) -> tuple[float, float, float]:
    """Return the independently reusable hit, overlap, and margin terms."""
    left, right = split_positions(entries, action_id, capacity)
    groups: list[np.ndarray] = []
    for side in (left, right):
        subgroups = deterministic_rollout_groups(entries[side], capacity)
        groups.extend([side[g] for g in subgroups])
    boxes = bboxes_for_groups(entries, groups)
    if len(queries):
        hits = intersect_matrix(queries, boxes)
        page_hits = float(hits.sum()) / max(len(queries), 1)
    else:
        page_hits = float(len(boxes))
    overlap = normalized_overlap_pair_area(boxes, bbox(entries))
    root = bbox(entries)
    root_margin = max(float(2.0 * ((root[2] - root[0]) + (root[3] - root[1]))), 1e-9)
    margin = float(np.sum(rect_margin(boxes))) / root_margin
    return page_hits, overlap, margin


def action_cost(entries: np.ndarray, queries: np.ndarray, action_id: int, capacity: int,
                overlap_weight: float, margin_weight: float) -> float:
    page_hits, overlap, margin = action_cost_components(entries, queries, action_id, capacity)
    return page_hits + overlap_weight * overlap + margin_weight * margin


def label_state(entries: np.ndarray, queries: np.ndarray, domain: int, level: int, capacity: int,
                hist_bins: int, candidate_count: int, rng: np.random.Generator,
                overlap_weight: float, margin_weight: float) -> dict[str, Any]:
    if page_count(len(entries), capacity) < 2:
        raise ValueError("teacher state must span at least two pages")
    action_ids = candidate_action_ids(rng, candidate_count)
    costs = np.asarray([
        action_cost(entries, queries, int(a), capacity, overlap_weight, margin_weight) for a in action_ids
    ], dtype=np.float32)
    best = float(np.min(costs))
    regrets = (costs - best) / max(best, 1.0)
    x = state_feature(entries, queries, capacity, hist_bins, level)
    return {
        "x": x, "action_ids": action_ids.astype(np.int16), "regrets": regrets.astype(np.float32),
        "raw_costs": costs.astype(np.float32), "domain": int(domain), "level": int(level),
        "entry_count": int(len(entries)), "query_count": int(len(queries)),
    }


def _str_groups_simple(rects: np.ndarray, capacity: int) -> list[np.ndarray]:
    # Fast spatial groups used only to manufacture realistic internal-level training states.
    c = centers(rects)
    order = np.argsort(morton_codes(c), kind="mergesort")
    return [order[i:i+capacity].astype(np.int64) for i in range(0, len(order), capacity)]


def preliminary_levels(rects: np.ndarray, capacity: int) -> list[np.ndarray]:
    levels = [np.asarray(rects, dtype=np.float32)]
    current = levels[0]
    while len(current) > capacity:
        groups = _str_groups_simple(current, capacity)
        current = bboxes_for_groups(current, groups)
        levels.append(current)
    return levels[:-1] if len(levels) > 1 else levels


def sample_states_for_domain(rects: np.ndarray, construction_queries: np.ndarray, domain_id: int,
                             count: int, capacity: int, min_pages: int, max_pages: int,
                             local_query_cap: int, hist_bins: int, candidate_count: int,
                             seed: int, overlap_weight: float, margin_weight: float,
                             progress_prefix: str = "") -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    levels = preliminary_levels(rects, capacity)
    level_orders = [np.argsort(morton_codes(centers(level)), kind="mergesort") for level in levels]
    records: list[dict[str, Any]] = []
    attempts = 0
    while len(records) < count:
        attempts += 1
        level = int(rng.integers(0, len(levels)))
        arr = levels[level]; order = level_orders[level]
        if len(arr) <= capacity:
            continue
        maxp = min(max_pages, page_count(len(arr), capacity))
        if maxp < min_pages:
            level = 0; arr = levels[0]; order = level_orders[0]; maxp = min(max_pages, page_count(len(arr), capacity))
        pages = int(rng.integers(min_pages, maxp + 1))
        n_target = min(len(arr), int(rng.integers(max((pages - 1) * capacity + 1, 2), pages * capacity + 1)))
        if n_target <= capacity:
            continue
        start = int(rng.integers(0, max(1, len(arr) - n_target + 1)))
        pos = order[start:start+n_target]
        entries = np.asarray(arr[pos], dtype=np.float32)
        queries = local_queries_for_state(entries, construction_queries, local_query_cap, rng)
        try:
            rec = label_state(entries, queries, domain_id, level, capacity, hist_bins, candidate_count, rng, overlap_weight, margin_weight)
        except Exception:
            if attempts > count * 20:
                raise
            continue
        records.append(rec)
        if len(records) % 100 == 0:
            print(f"{progress_prefix}teacher states {len(records)}/{count}", flush=True)
    return records


def save_records(path: Path, records: list[dict[str, Any]], action_count: int = ACTION_COUNT) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("no records")
    x = np.stack([r["x"] for r in records]).astype(np.float32)
    ids = np.full((len(records), action_count), -1, dtype=np.int16)
    reg = np.full((len(records), action_count), np.nan, dtype=np.float32)
    raw = np.full((len(records), action_count), np.nan, dtype=np.float32)
    for i, r in enumerate(records):
        a = np.asarray(r["action_ids"], dtype=np.int64)
        ids[i, a] = a.astype(np.int16)
        reg[i, a] = np.asarray(r["regrets"], dtype=np.float32)
        raw[i, a] = np.asarray(r["raw_costs"], dtype=np.float32)
    np.savez_compressed(
        path, x=x, action_ids=ids, regrets=reg, raw_costs=raw,
        domain=np.asarray([r["domain"] for r in records], dtype=np.int8),
        level=np.asarray([r["level"] for r in records], dtype=np.int8),
        entry_count=np.asarray([r["entry_count"] for r in records], dtype=np.int32),
        query_count=np.asarray([r["query_count"] for r in records], dtype=np.int16),
    )
    atomic_json(path.with_suffix(".json"), {
        "records": len(records), "feature_dim": int(x.shape[1]), "action_count": action_count,
        "x_sha256": sha256_array(x), "domains": {str(i): int(np.sum(np.asarray([r['domain'] for r in records]) == i)) for i in range(3)},
    })


def load_records(paths: list[Path]) -> dict[str, np.ndarray]:
    parts = [np.load(p, allow_pickle=False) for p in paths]
    keys = ("x", "action_ids", "regrets", "raw_costs", "domain", "level", "entry_count", "query_count")
    return {k: np.concatenate([z[k] for z in parts], axis=0) for k in keys}
