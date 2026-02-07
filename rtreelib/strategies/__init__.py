from .guttman import RTreeGuttman
from .rstar import RStarTree
from .base import insert, adjust_tree_strategy, least_area_enlargement

# Flow Matching implementation
try:
    from .flow_matching import FlowMatchingTree
    __all__ = [
        'RTreeGuttman', 'RStarTree', 'FlowMatchingTree',
        'insert', 'adjust_tree_strategy', 'least_area_enlargement'
    ]
except ImportError:
    # Flow Matching requires PyTorch, so it might not be available
    __all__ = [
        'RTreeGuttman', 'RStarTree',
        'insert', 'adjust_tree_strategy', 'least_area_enlargement'
    ]
