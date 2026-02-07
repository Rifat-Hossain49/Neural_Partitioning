"""
Implementation of the Guttman R-Tree strategies described in this paper:
http://www-db.deis.unibo.it/courses/SI-LS/papers/Gut84.pdf

This implementation is used as the default for this library.
"""

import math
import itertools
import numpy as np
from typing import List, TypeVar
from ..rtree import RTreeBase, RTreeEntry, RTreeNode, DEFAULT_MAX_ENTRIES, EPSILON
from rtreelib.models import Rect
from .base import insert, least_area_enlargement, adjust_tree_strategy

T = TypeVar('T')


def guttman_choose_leaf(tree: RTreeBase[T], entry: RTreeEntry[T]) -> RTreeNode[T]:
    """
    Select a leaf node in which to place a new index entry. This strategy always inserts into the subtree that requires
    least enlargement of its bounding box.
    """
    node = tree.root
    while not node.is_leaf:
        e: RTreeEntry = least_area_enlargement(node.entries, entry.rect)
        node = e.child
    return node


def quadratic_split(tree: RTreeBase[T], node: RTreeNode[T]) -> RTreeNode[T]:
    """
    Split an overflowing node. This algorithm attempts to find a small-area split, but is not guaranteed to
    find one with the smallest area possible. It's a good tradeoff between runtime efficiency and optimal area.
    Pages in this tree tend to overlap a lot, but the bounding rectangles are generally small, which makes for
    fast lookup.

    From the original paper:

    "The division should be done in a way that makes it as unlikely as possible that both new nodes will need to
    be examined on subsequent searches. Since the decision whether to visit a node depends on whether its covering
    rectangle overlaps the search area, the total area of the two covering rectangles after a split should be
    minimized."

    :param tree: RTreeBase[T]: R-tree instance.
    :param node: RTreeNode[T]: Overflowing node that needs to be split.
    :return: Newly-created split node whose entries are a subset of the original node's entries.
    """
    entries = node.entries[:]
    seed1, seed2 = _pick_seeds(entries)
    entries.remove(seed1)
    entries.remove(seed2)
    group1, group2 = ([seed1], [seed2])
    rect1, rect2 = (seed1.rect, seed2.rect)
    num_entries = len(entries)
    while num_entries > 0:
        # If one group has so few entries that all the rest must be assigned to it in order for it to meet the
        # min_entries requirement, assign them and stop. (If both groups are underfull, then proceed with the
        # algorithm to determine the best group to extend.)
        len1, len2 = (len(group1), len(group2))
        group1_underfull = len1 < tree.min_entries <= len1 + num_entries
        group2_underfull = len2 < tree.min_entries <= len2 + num_entries
        if group1_underfull and not group2_underfull:
            group1.extend(entries)
            break
        if group2_underfull and not group1_underfull:
            group2.extend(entries)
            break
        # Pick the next entry to assign
        area1, area2 = rect1.area(), rect2.area()
        entry = _pick_next(entries, rect1, area1, rect2, area2)
        
        # Add it to the group whose covering rectangle will have to be enlarged the least
        urect1, urect2 = rect1.union(entry.rect), rect2.union(entry.rect)
        ua1, ua2 = urect1.area(), urect2.area()
        
        if not np.isfinite(ua1) or not np.isfinite(ua2) or not np.isfinite(area1) or not np.isfinite(area2):
            enlargement1 = urect1.perimeter() / 2.0 - rect1.perimeter() / 2.0
            enlargement2 = urect2.perimeter() / 2.0 - rect2.perimeter() / 2.0
        else:
            enlargement1 = ua1 - area1
            enlargement2 = ua2 - area2
            if not np.isfinite(enlargement1) or not np.isfinite(enlargement2):
                enlargement1 = urect1.perimeter() / 2.0 - rect1.perimeter() / 2.0
                enlargement2 = urect2.perimeter() / 2.0 - rect2.perimeter() / 2.0

        if math.isclose(enlargement1, enlargement2, rel_tol=EPSILON):
            if math.isclose(area1, area2, rel_tol=EPSILON):
                group = group1 if len1 <= len2 else group2
            else:
                group = group1 if area1 < area2 else group2
        else:
            group = group1 if enlargement1 < enlargement2 else group2
        group.append(entry)
        # Update the winning group's covering rectangle
        if group is group1:
            rect1 = urect1
        else:
            rect2 = urect2
        # Update entries list
        entries.remove(entry)
        num_entries = len(entries)
    return tree.perform_node_split(node, group1, group2)


def _pick_seeds(entries: List[RTreeEntry[T]]) -> (RTreeEntry[T], RTreeEntry[T]):
    seeds = None
    max_wasted_area = None
    for e1, e2 in itertools.combinations(entries, 2):
        a1 = e1.rect.area()
        a2 = e2.rect.area()
        u_rect = e1.rect.union(e2.rect)
        ua = u_rect.area()
        
        if not np.isfinite(ua):
            # Fallback to margin
            wasted_area = u_rect.perimeter() / 2.0 - e1.rect.perimeter() / 2.0 - e2.rect.perimeter() / 2.0
        else:
            wasted_area = ua - a1 - a2
            if not np.isfinite(wasted_area):
                wasted_area = u_rect.perimeter() / 2.0 - e1.rect.perimeter() / 2.0 - e2.rect.perimeter() / 2.0
                
        if max_wasted_area is None or wasted_area > max_wasted_area:
            max_wasted_area = wasted_area
            seeds = (e1, e2)
    return seeds


def _pick_next(remaining_entries: List[RTreeEntry[T]],
               group1_rect: Rect,
               group1_area: float,
               group2_rect: Rect,
               group2_area: float) -> RTreeEntry[T]:
    max_diff = None
    result = None
    for e in remaining_entries:
        u1 = group1_rect.union(e.rect)
        u2 = group2_rect.union(e.rect)
        ua1 = u1.area()
        ua2 = u2.area()
        
        if not np.isfinite(ua1) or not np.isfinite(ua2) or not np.isfinite(group1_area) or not np.isfinite(group2_area):
            d1 = u1.perimeter() / 2.0 - group1_rect.perimeter() / 2.0
            d2 = u2.perimeter() / 2.0 - group2_rect.perimeter() / 2.0
        else:
            d1 = ua1 - group1_area
            d2 = ua2 - group2_area
            if not np.isfinite(d1) or not np.isfinite(d2):
                d1 = u1.perimeter() / 2.0 - group1_rect.perimeter() / 2.0
                d2 = u2.perimeter() / 2.0 - group2_rect.perimeter() / 2.0
                
        diff = math.fabs(d1 - d2)
        if max_diff is None or diff > max_diff:
            max_diff = diff
            result = e
    return result


class RTreeGuttman(RTreeBase[T]):
    """R-Tree implementation that uses Guttman's strategies for insertion, splitting, and deletion."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES, min_entries: int = None):
        """
        Initializes the R-Tree using Guttman's strategies for insertion, splitting, and deletion.
        :param max_entries: Maximum number of entries per node.
        :param min_entries: Minimum number of entries per node. Defaults to ceil(max_entries/2).
        """
        super().__init__(
            max_entries=max_entries,
            min_entries=min_entries,
            insert=insert,
            choose_leaf=guttman_choose_leaf,
            adjust_tree=adjust_tree_strategy,
            overflow_strategy=quadratic_split
        )
