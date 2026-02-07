from rtreelib.models import Rect, Point, Location
from .rtree import RTreeBase, RTreeNode, RTreeEntry, DEFAULT_MAX_ENTRIES, EPSILON
from .strategies import (
    RTreeGuttman, RTreeGuttman as RTree, RStarTree, insert, adjust_tree_strategy, least_area_enlargement)

# Flow Matching implementation (optional - requires PyTorch)
try:
    from .strategies.flow_matching import FlowMatchingTree
    __all__ = [
        'Rect', 'Point', 'Location',
        'RTreeBase', 'RTreeNode', 'RTreeEntry', 'DEFAULT_MAX_ENTRIES', 'EPSILON',
        'RTreeGuttman', 'RTree', 'RStarTree', 'FlowMatchingTree',
        'insert', 'adjust_tree_strategy', 'least_area_enlargement'
    ]
except ImportError:
    __all__ = [
        'Rect', 'Point', 'Location',
        'RTreeBase', 'RTreeNode', 'RTreeEntry', 'DEFAULT_MAX_ENTRIES', 'EPSILON',
        'RTreeGuttman', 'RTree', 'RStarTree',
        'insert', 'adjust_tree_strategy', 'least_area_enlargement'
    ]
