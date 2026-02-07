"""
Neural Partitioning-based R-tree Construction (§6)

Implements PartitioningRTree which builds an R-tree top-down using Algorithm 3.
"""

import torch
import numpy as np
import sys
import os
import argparse
import time
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtreelib.models import Rect
from rtreelib.strategies import RTreeGuttman
from rtreelib.rtree import RTreeNode as LibRTreeNode, RTreeEntry as LibRTreeEntry
from neural_partitioning.model import PartitioningNetwork
from neural_partitioning.utils import load_rectangles_from_csv, test_query_performance, generate_random_queries, build_baseline_rtrees

class PartitioningRTree(RTreeGuttman):
    """
    R-tree wrapper that uses a trained Neural Partitioning Network for bulk loading.
    """
    def __init__(self, model: PartitioningNetwork, max_entries: int = 8, device: str = 'cpu'):
        super().__init__(max_entries=max_entries)
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        # Determine num_classes from model
        self.num_classes = model.partition_mlp[-1].out_features
        if self.num_classes > self.max_entries:
            print(f"Warning: Model num_classes ({self.num_classes}) > max_entries ({self.max_entries}). "
                  f"Capping effective split to {self.max_entries}.")
            self.num_classes = self.max_entries

    def insert_all(self, rectangles: List[Rect]):
        """Bulk load using partitioning."""
        n = len(rectangles)
        print(f"    Building Partitioning R-Tree with {n} items...")
        start = time.time()
        root = construct_tree_top_down(
            self,
            rectangles, 
            list(range(n)), 
            self.model, 
            self.max_entries, 
            self.device, 
            self.num_classes
        )
        self.root = root
        print(f"    Building done in {time.time() - start:.2f}s")

def construct_tree_top_down(
    tree: PartitioningRTree,
    rectangles: List[Rect],
    indices: List[int],
    model: PartitioningNetwork,
    max_entries: int,
    device: torch.device,
    num_classes: int = 2
) -> LibRTreeNode:
    """
    Iteratively build tree top-down using NPN (Algorithm 3).
    """
    n_total = len(indices)
    if n_total == 0:
        return LibRTreeNode(tree, is_leaf=True)

    # Determine dimensionality for round-robin fallback
    ndim = rectangles[0].dims

    # Create root
    root = LibRTreeNode(tree, is_leaf=True) # Will change to False if we split
    
    # Work list: stack of (node, indices, depth)
    stack = [(root, indices, 0)]
    
    while stack:
        node, current_indices, depth = stack.pop()
        n = len(current_indices)
        
        if n > 1000:
            print(f"      [Depth {depth}] Building node with {n} rects...")
            
        # Check leaf capacity
        if n <= max_entries:
            node._is_leaf = True
            entries = []
            for idx in current_indices:
                entries.append(LibRTreeEntry(rectangles[idx], child=None, data=idx))
            node.entries = entries
            continue
            
        # Internal node
        node._is_leaf = False
        
        # Policy Inference
        subset_rects = [rectangles[i] for i in current_indices]
        
        with torch.no_grad():
            logits = model(subset_rects)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
        # Partition
        groups = [[] for _ in range(num_classes)]
        for i, label in enumerate(preds):
            groups[label].append(current_indices[i])
            
        non_empty_groups = [g for g in groups if len(g) > 0]
        
        # Fallback check
        if len(non_empty_groups) < 2:
            # Round-robin axis sort fallback
            axis = depth % ndim
            # Sort by center coordinate along 'axis'
            sorted_indices = sorted(
                current_indices, 
                key=lambda i: (rectangles[i].min[axis] + rectangles[i].max[axis]) / 2
            )
            
            chunk_size = (n + num_classes - 1) // num_classes
            non_empty_groups = []
            for i in range(0, n, chunk_size):
                non_empty_groups.append(sorted_indices[i:i+chunk_size])
        
        # Create children and add to work list
        entries = []
        for group in non_empty_groups:
            child = LibRTreeNode(tree, is_leaf=True)
            child.parent = node
            
            # Compute MBR of the group
            group_mins = np.array([rectangles[i].min for i in group])
            group_maxs = np.array([rectangles[i].max for i in group])
            mbr_min = np.min(group_mins, axis=0)
            mbr_max = np.max(group_maxs, axis=0)
            mbr = Rect(mbr_min, mbr_max)
            
            entries.append(LibRTreeEntry(mbr, child=child, data=None))
            
            # Add to stack
            stack.append((child, group, depth + 1))
            
        node.entries = entries

    return root



