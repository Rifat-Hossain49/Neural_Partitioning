"""
This module defines strategies and helper functions that are shared by more than one R-tree variant.
"""

import math
import numpy as np
from typing import TypeVar, List
from ..rtree import RTreeBase, RTreeEntry, RTreeNode, EPSILON
from rtreelib.models import Rect, union_all


T = TypeVar('T')


def insert(tree: RTreeBase[T], data: T, rect: Rect) -> RTreeEntry[T]:
    """
    Strategy for inserting a new entry into the tree. This makes use of the choose_leaf strategy to find an
    appropriate leaf node where the new entry should be inserted. If the node is overflowing after inserting the entry,
    then overflow_strategy is invoked (either to split the node in case of Guttman, or do a combination of forced
    reinsert and/or split in the case of R*).
    :param tree: R-tree instance
    :param data: Entry data
    :param rect: Bounding rectangle
    :return: RTreeEntry instance for the newly-inserted entry.
    """
    entry = RTreeEntry(rect, data=data)
    node = tree.choose_leaf(tree, entry)
    node.entries.append(entry)
    split_node = None
    if len(node.entries) > tree.max_entries:
        split_node = tree.overflow_strategy(tree, node)
    tree.adjust_tree(tree, node, split_node)
    return entry


def least_area_enlargement(entries: List[RTreeEntry[T]], rect: Rect) -> RTreeEntry[T]:
    """
    Selects a child entry that requires least area enlargement.
    Robust for high dimensions: if area overflows, falls back to sum of edge lengths (margin).
    """
    if not entries:
        raise ValueError("least_area_enlargement called with empty entries")
        
    areas = []
    enlargements = []
    
    for child in entries:
        a = child.rect.area()
        # Use log or margin if area is inf
        if not np.isfinite(a):
            # Fallback to margin (sum of edge lengths)
            a = child.rect.perimeter() / 2.0
            enlargement = child.rect.union(rect).perimeter() / 2.0 - a
        else:
            u_area = child.rect.union(rect).area()
            if not np.isfinite(u_area):
                # Fallback to margin enlargement
                enlargement = child.rect.union(rect).perimeter() / 2.0 - child.rect.perimeter() / 2.0
            else:
                enlargement = u_area - a
        
        areas.append(a)
        enlargements.append(enlargement)
        
    min_enlargement = min(enlargements)
    
    # Handle nan in min_enlargement (should not happen with isfinite check but just in case)
    if not np.isfinite(min_enlargement):
        # Last resort: just pick first
        return entries[0]
        
    indices = [i for i, v in enumerate(enlargements) if math.isclose(v, min_enlargement, rel_tol=EPSILON)]
    
    # If a single entry is a clear winner, choose that entry. Otherwise, if there are multiple entries having the
    # same enlargement, choose the entry having the smallest area as a tie-breaker.
    if len(indices) == 1:
        return entries[indices[0]]
    elif not indices:
        # This could happen if math.isclose fails for some reason
        return entries[0]
    else:
        # Tie-breaker: smallest area (or margin)
        min_area = min([areas[i] for i in indices])
        # Find index in the original list
        for i in indices:
            if math.isclose(areas[i], min_area, rel_tol=EPSILON):
                return entries[i]
        return entries[indices[0]]


def adjust_tree_strategy(tree: RTreeBase[T], node: RTreeNode[T], split_node: RTreeNode[T] = None) -> None:
    """
    Ascend from a leaf node to the root, adjusting covering rectangles and propagating node splits as necessary.
    """
    while not node.is_root:
        parent = node.parent
        node.parent_entry.rect = union_all([entry.rect for entry in node.entries])
        if split_node is not None:
            rect = union_all([e.rect for e in split_node.entries])
            entry = RTreeEntry(rect, child=split_node)
            parent.entries.append(entry)
            if len(parent.entries) > tree.max_entries:
                split_node = tree.overflow_strategy(tree, parent)
            else:
                split_node = None
        node = parent
    if split_node is not None:
        tree.grow_tree([node, split_node])
