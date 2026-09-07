"""
Top-down Greedy Split (TGS) bulk-loading of R-trees.

Reference: Garcia, Y. J., Lopez, M. A., Leutenegger, S. T. (1998).
"A Greedy Algorithm for Bulk Loading R-trees".
In Proceedings of the 6th ACM GIS, pp. 163-164.
DOI: 10.1145/288692.288723

This implementation follows the paper's basic split step:

* build the tree top-down;
* at each tree level set S to the maximum number of rectangles that can fit
  in one child subtree;
* consider packed binary cuts whose left side has cardinality i*S;
* evaluate all requested axis/order combinations; and
* choose the cut with the best user-supplied cost.

The paper reports results using area as the cost function and center-value
ordering. Those are therefore the defaults here. Other cost/order choices are
kept as explicit options, but no candidate sampling or flexible split window is
used by default.

Minimum occupancy: this is a static packed bulk loader, and like other packed
loaders (for example STR's final run per level) it intentionally does not
enforce the dynamic R-tree minimum-fill rule.  Because every cut is placed at
a multiple of the subtree capacity S, all subtrees are completely full except
for at most one remainder chain, so at most one node per level can fall below
``ceil(max_entries / 2)`` entries (the root is exempt by definition).  The
paper's published two-page version fixes cut positions to ``i * S`` and defers
remainder redistribution to the unavailable technical report (#97-02);
enforcing minimum fill would require cuts at non-multiples of S and therefore
deviate from the published algorithm.  Use ``minimum_occupancy_report`` to
quantify the underfilled remainder nodes of a built tree.
"""

import math
from typing import Any, List, Sequence, Tuple

import numpy as np

from rtreelib.models import Rect
from rtreelib.rtree import RTreeEntry, RTreeNode
from rtreelib.strategies.guttman import RTreeGuttman


def _tgs_height_for_n(n: int, max_entries: int) -> int:
    """Minimum tree height h such that max_entries ** (h + 1) >= n."""
    if n <= 0:
        return 0
    h = 0
    capacity = int(max_entries)
    while capacity < n:
        h += 1
        capacity *= int(max_entries)
    return h


def _tgs_subtree_cap(height: int, max_entries: int) -> int:
    """Maximum objects in a subtree of the given height."""
    capacity = int(max_entries)
    for _ in range(int(height)):
        capacity *= int(max_entries)
    return capacity


def _expanded_orderings(orderings: Sequence[str]) -> Tuple[str, ...]:
    expanded = []
    for ordering in orderings or ("center",):
        name = str(ordering).lower()
        if name == "both":
            # The paper lists "both" as a distinct ordering but defines it only
            # in the unavailable technical report (#97-02).  Refusing is safer
            # than silently substituting a different search.
            raise ValueError(
                "The paper's 'both' ordering is not defined in the published "
                "two-page version; pass ('min', 'max') explicitly to evaluate "
                "both endpoint orderings instead"
            )
        expanded.append(name)
    return tuple(expanded) or ("center",)


def _ordered_indices_for_axis(sub: np.ndarray, axis: int, ordering: str) -> np.ndarray:
    if ordering == "center":
        key = (sub[:, axis] + sub[:, axis + 2]) * 0.5
        return np.argsort(key, kind="stable")
    if ordering == "min":
        return np.argsort(sub[:, axis], kind="stable")
    if ordering == "max":
        return np.argsort(sub[:, axis + 2], kind="stable")
    raise ValueError(f"Unknown TGS ordering: {ordering!r}")


def _packed_split_positions(n: int, group_cap: int) -> np.ndarray:
    group_count = int(math.ceil(n / group_cap))
    positions = [
        i * group_cap
        for i in range(1, group_count)
        if 0 < i * group_cap < n
    ]
    if not positions:
        positions = [min(max(1, group_cap), n - 1)]
    return np.asarray(positions, dtype=np.intp)


def _tgs_greedy_binary_split(
    rects_arr: np.ndarray,
    indices: List[int],
    group_cap: int,
    *,
    cost: str = "area",
    orderings: Sequence[str] = ("center",),
) -> Tuple[List[int], List[int]]:
    """Find the best paper-style packed binary cut for the current subset."""
    n = len(indices)
    if n == 0:
        return [], []
    if n == 1:
        return list(indices), []
    if group_cap <= 0:
        raise ValueError("group_cap must be positive")

    cost = str(cost).lower()
    if cost == "weighted_perimeter":
        # Kamel-Faloutsos weighted perimeter is area + q * (w + h) + q * q for
        # an expected query extent q; without that extent it cannot be
        # computed, and plain margin is not an acceptable stand-in.
        raise NotImplementedError(
            "weighted_perimeter requires an expected query extent; "
            "use cost='margin' (unweighted perimeter ranking) instead"
        )
    if cost == "perimeter":
        # Dropping the constant factor of two does not change the ranking.
        cost = "margin"
    if cost not in ("area", "overlap", "margin"):
        raise ValueError("TGS cost must be one of: area, overlap, margin, perimeter")

    idx_arr = np.asarray(indices, dtype=np.intp)
    sub = rects_arr[idx_arr]
    splits = _packed_split_positions(n, group_cap)
    orderings = _expanded_orderings(orderings)

    best_primary = np.inf
    best_secondary = np.inf
    best_tertiary = np.inf
    best_left: List[int] = []
    best_right: List[int] = []

    for axis in range(2):
        for ordering_name in orderings:
            order = _ordered_indices_for_axis(sub, axis, ordering_name)
            sorted_rects = sub[order]

            pfx_xmin = np.minimum.accumulate(sorted_rects[:, 0])
            pfx_ymin = np.minimum.accumulate(sorted_rects[:, 1])
            pfx_xmax = np.maximum.accumulate(sorted_rects[:, 2])
            pfx_ymax = np.maximum.accumulate(sorted_rects[:, 3])

            sfx_xmin = np.minimum.accumulate(sorted_rects[::-1, 0])[::-1]
            sfx_ymin = np.minimum.accumulate(sorted_rects[::-1, 1])[::-1]
            sfx_xmax = np.maximum.accumulate(sorted_rects[::-1, 2])[::-1]
            sfx_ymax = np.maximum.accumulate(sorted_rects[::-1, 3])[::-1]

            l_xmin = pfx_xmin[splits - 1]
            l_ymin = pfx_ymin[splits - 1]
            l_xmax = pfx_xmax[splits - 1]
            l_ymax = pfx_ymax[splits - 1]
            r_xmin = sfx_xmin[splits]
            r_ymin = sfx_ymin[splits]
            r_xmax = sfx_xmax[splits]
            r_ymax = sfx_ymax[splits]

            overlap_x = np.maximum(
                0.0,
                np.minimum(l_xmax, r_xmax) - np.maximum(l_xmin, r_xmin),
            )
            overlap_y = np.maximum(
                0.0,
                np.minimum(l_ymax, r_ymax) - np.maximum(l_ymin, r_ymin),
            )
            overlaps = overlap_x * overlap_y
            areas = (
                (l_xmax - l_xmin) * (l_ymax - l_ymin)
                + (r_xmax - r_xmin) * (r_ymax - r_ymin)
            )
            margins = (
                (l_xmax - l_xmin)
                + (l_ymax - l_ymin)
                + (r_xmax - r_xmin)
                + (r_ymax - r_ymin)
            )

            if cost == "area":
                primary, secondary, tertiary = areas, overlaps, margins
            elif cost == "overlap":
                primary, secondary, tertiary = overlaps, areas, margins
            else:
                primary, secondary, tertiary = margins, overlaps, areas

            best_idx = int(np.lexsort((tertiary, secondary, primary))[0])
            primary_value = float(primary[best_idx])
            secondary_value = float(secondary[best_idx])
            tertiary_value = float(tertiary[best_idx])

            if (
                primary_value < best_primary
                or (
                    primary_value == best_primary
                    and secondary_value < best_secondary
                )
                or (
                    primary_value == best_primary
                    and secondary_value == best_secondary
                    and tertiary_value < best_tertiary
                )
            ):
                best_primary = primary_value
                best_secondary = secondary_value
                best_tertiary = tertiary_value
                split = int(splits[best_idx])
                best_left = list(idx_arr[order[:split]])
                best_right = list(idx_arr[order[split:]])

    if not best_left and not best_right:
        order = np.argsort(sub[:, 0], kind="stable")
        split = min(max(1, group_cap), n - 1)
        best_left = list(idx_arr[order[:split]])
        best_right = list(idx_arr[order[split:]])

    return best_left, best_right


def _tgs_recursive_split(
    rects_arr: np.ndarray,
    indices: List[int],
    group_cap: int,
    *,
    cost: str = "area",
    orderings: Sequence[str] = ("center",),
) -> List[List[int]]:
    """Split indices into packed groups with at most group_cap objects each."""
    if not indices:
        return []
    if len(indices) <= group_cap:
        return [list(indices)]

    left_idx, right_idx = _tgs_greedy_binary_split(
        rects_arr,
        indices,
        group_cap,
        cost=cost,
        orderings=orderings,
    )

    groups: List[List[int]] = []
    if left_idx:
        groups.extend(
            _tgs_recursive_split(
                rects_arr,
                left_idx,
                group_cap,
                cost=cost,
                orderings=orderings,
            )
        )
    if right_idx:
        groups.extend(
            _tgs_recursive_split(
                rects_arr,
                right_idx,
                group_cap,
                cost=cost,
                orderings=orderings,
            )
        )
    return [group for group in groups if group]


def build_tree_tgs(
    rects: Any,
    data: Sequence[Any],
    *,
    max_entries: int = 128,
    cost: str = "area",
    orderings: Sequence[str] = ("center",),
) -> RTreeGuttman:
    """Bulk-load an R-tree using Top-down Greedy Split.

    Defaults follow Garcia, Lopez, and Leutenegger's reported TGS variant:
    area cost and center-value ordering.

    The result is a balanced packed tree: every leaf is at the same depth and
    no node exceeds ``max_entries``.  Minimum occupancy is deliberately not
    enforced; at most one remainder node per level may be underfilled (see the
    module docstring and ``minimum_occupancy_report``).
    """
    rects_arr = np.asarray(rects, dtype=np.float64)
    n = len(rects_arr)
    max_entries = int(max_entries)
    if max_entries < 2:
        raise ValueError("max_entries must be at least 2")
    if len(data) != n:
        raise ValueError("data length must match number of rectangles")
    if n > 0:
        if rects_arr.ndim != 2 or rects_arr.shape[1] != 4:
            raise ValueError("rects must have shape (n, 4)")
        if not np.all(np.isfinite(rects_arr)):
            raise ValueError("Rectangle coordinates must be finite")
        if np.any(rects_arr[:, 0] > rects_arr[:, 2]):
            raise ValueError("min_x cannot exceed max_x")
        if np.any(rects_arr[:, 1] > rects_arr[:, 3]):
            raise ValueError("min_y cannot exceed max_y")

    tree = RTreeGuttman(max_entries=max_entries)
    if n == 0:
        tree.root = RTreeNode(tree, True)
        return tree

    tree_height = _tgs_height_for_n(n, max_entries)

    def _make_rect(idx: int) -> Rect:
        rect = rects_arr[idx]
        return Rect(float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))

    def _build(indices: List[int], height: int) -> RTreeNode:
        # Leaves are created by remaining height, never by cardinality alone:
        # an R-tree requires every leaf at the same level, so an underfilled
        # remainder still receives a chain of internal nodes down to height 0.
        if height == 0:
            if len(indices) > max_entries:
                raise RuntimeError(
                    "Leaf capacity exceeded; tree-height calculation is inconsistent"
                )
            leaf = RTreeNode(tree, True)
            for idx in indices:
                leaf.entries.append(RTreeEntry(_make_rect(idx), data=data[idx]))
            return leaf

        group_cap = _tgs_subtree_cap(height - 1, max_entries)
        groups = _tgs_recursive_split(
            rects_arr,
            indices,
            group_cap,
            cost=cost,
            orderings=orderings,
        )

        node = RTreeNode(tree, False)
        for group in groups:
            child = _build(group, height - 1)
            child.parent = node
            node.entries.append(RTreeEntry(child.get_bounding_rect(), child=child))
        return node

    tree.root = _build(list(range(n)), tree_height)
    tree.root.parent = None
    return tree


def compute_normalized_total_overlap(tree: RTreeGuttman) -> float:
    """Total pairwise child-MBR overlap across all internal nodes / root area.

    This accumulates overlap over every node pair at every internal level, so
    the same spatial region can be counted repeatedly and the result may
    exceed 1.  It is a normalized total, not a bounded ratio or percentage.
    """
    root_mbr = tree.root.get_bounding_rect()
    if root_mbr is None:
        return 0.0
    root_area = (root_mbr.max_x - root_mbr.min_x) * (root_mbr.max_y - root_mbr.min_y)
    if root_area <= 0.0:
        return 0.0

    total = 0.0
    for node in tree.get_nodes():
        if node.is_leaf or len(node.entries) < 2:
            continue
        rects = np.array(
            [
                [entry.rect.min_x, entry.rect.min_y, entry.rect.max_x, entry.rect.max_y]
                for entry in node.entries
            ],
            dtype=np.float64,
        )
        xmin, ymin, xmax, ymax = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]
        overlap_x = np.maximum(
            0.0,
            np.minimum(xmax[:, None], xmax[None, :])
            - np.maximum(xmin[:, None], xmin[None, :]),
        )
        overlap_y = np.maximum(
            0.0,
            np.minimum(ymax[:, None], ymax[None, :])
            - np.maximum(ymin[:, None], ymin[None, :]),
        )
        total += float(np.sum(np.triu(overlap_x * overlap_y, k=1)))
    return total / root_area


# Backwards-compatible alias; the old name suggested a bounded ratio, which
# this metric is not.
compute_tree_overlap_ratio = compute_normalized_total_overlap


def minimum_occupancy_report(tree: RTreeGuttman) -> List[str]:
    """Describe every node outside the dynamic R-tree occupancy bounds.

    Maximum-capacity violations are always defects.  Underfilled non-root
    nodes are the expected packed-loader remainder artifact (at most one per
    level for TGS); callers comparing against dynamic R-trees can use this
    report to state that deviation explicitly.
    """
    violations: List[str] = []
    for node in tree.get_nodes():
        count = len(node.entries)
        if count > tree.max_entries:
            violations.append(
                f"Node exceeds maximum occupancy: {count} > {tree.max_entries}"
            )
        if node is not tree.root and count < tree.min_entries:
            violations.append(
                f"Non-root node is underfilled: {count} < {tree.min_entries}"
            )
    return violations
