"""
Neural Partitioning Network (NPN) Implementation

This package implements the research paper algorithm:
1. Exact DP supervisor on small sets (k_0 ≤ 12)
2. Query-aware supervision with REC
3. Neural Partitioning Network for top-down construction
"""

from .dp_supervisor import compute_optimal_tree, get_optimal_partition_labels
from .query_set import generate_query_set, compute_query_probability
from .model import PartitioningNetwork
from .inference import PartitioningRTree
from .utils import load_rectangles_from_csv

__all__ = [
    'compute_optimal_tree',
    'get_optimal_partition_labels',
    'generate_query_set',
    'compute_query_probability',
    'PartitioningNetwork',
    'PartitioningRTree',
    'load_rectangles_from_csv'
]
