#!/usr/bin/env python
"""
Node Access Benchmark - Varying Relative Window Size

Compares node access efficiency across different relative window sizes
for Partitioning Model vs Guttman and R* baselines.

Based on paper's Node Access Efficiency Analysis section.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rtreelib.models import Rect
from rtreelib.strategies import RTreeGuttman, RStarTree
from neural_partitioning.model import PartitioningNetwork
from neural_partitioning.inference import PartitioningRTree
from neural_partitioning.utils import load_rectangles_from_csv

# ---------------- NODE ACCESS TRACKER ----------------
class NodeAccessTracker:
    """Track node accesses during tree traversal."""
    def __init__(self):
        self.access_count = 0
    
    def reset(self):
        self.access_count = 0
    
    def count(self):
        self.access_count += 1

tracker = NodeAccessTracker()

def wrap_tree_for_counting(tree):
    """Monkey-patch tree to count node accesses."""
    from rtreelib.rtree import RTreeNode
    original_get_mbr = RTreeNode.get_bounding_rect
    
    def tracked_get_mbr(self):
        tracker.count()
        return original_get_mbr(self)
    
    RTreeNode.get_bounding_rect = tracked_get_mbr

# ---------------- QUERY GENERATION ----------------
def compute_global_mbr(rectangles):
    """Compute the global MBR of all rectangles."""
    mins = np.array([r.min for r in rectangles])
    maxs = np.array([r.max for r in rectangles])
    global_min = np.min(mins, axis=0)
    global_max = np.max(maxs, axis=0)
    return Rect(global_min, global_max)

def generate_queries_for_window_size(global_mbr, relative_size, num_queries=1000):
    """
    Generate random range queries for a given relative window size.
    
    Args:
        global_mbr: Global bounding rectangle of dataset
        relative_size: Target relative window size (e.g., 0.001 for 0.1%)
        num_queries: Number of queries to generate
        
    Returns:
        List of query rectangles
    """
    global_area = global_mbr.area()
    target_area = global_area * relative_size
    
    ndim = len(global_mbr.min)
    extent = global_mbr.max - global_mbr.min
    
    queries = []
    for _ in range(num_queries):
        # For 2D: if we want area A, and we draw L uniformly from [0, sqrt(A)],
        # then the rectangle [0,L] x [0,L] has expected area around A/4.
        # We'll use a simple approach: draw width and height s.t. width*height ≈ target_area
        
        # Random aspect ratio
        aspect_ratio = np.random.uniform(0.5, 2.0)
        
        # Compute width and height
        # width * height = target_area
        # height = aspect_ratio * width
        # width^2 * aspect_ratio = target_area
        width = np.sqrt(target_area / aspect_ratio)
        height = aspect_ratio * width
        
        # Random center position
        center_normalized = np.random.rand(ndim)
        
        # For 2D case
        if ndim == 2:
            # Adjust to prevent query from going outside bounds
            half_w = width / extent[0] / 2
            half_h = height / extent[1] / 2
            
            # Clamp center to valid range
            cx = max(half_w, min(1 - half_w, center_normalized[0]))
            cy = max(half_h, min(1 - half_h, center_normalized[1]))
            
            # Convert to actual coordinates
            center_x = global_mbr.min[0] + cx * extent[0]
            center_y = global_mbr.min[1] + cy * extent[1]
            
            q_min = np.array([center_x - width/2, center_y - height/2])
            q_max = np.array([center_x + width/2, center_y + height/2])
            
            # Clamp to global bounds
            q_min = np.maximum(q_min, global_mbr.min)
            q_max = np.minimum(q_max, global_mbr.max)
            
            queries.append(Rect(q_min, q_max))
    
    return queries

# ---------------- BENCHMARKING ----------------
def load_model(model_path, device='cpu'):
    """Load a trained partitioning model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model = PartitioningNetwork(
        input_dim=config.get('input_dim', 9),
        hidden_dim=config.get('hidden_dim', 128),
        num_classes=config.get('num_classes', 3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def benchmark_method(method_name, tree, queries, window_size_label):
    """
    Run benchmark for a single method and window size.
    
    Returns:
        Average node accesses
    """
    wrap_tree_for_counting(tree)
    
    total_accesses = 0
    for query in queries:
        tracker.reset()
        list(tree.query(query))  # Execute query
        total_accesses += tracker.access_count
    
    avg_accesses = total_accesses / len(queries)
    print(f"    {method_name} @ {window_size_label}: {avg_accesses:.2f} avg nodes")
    
    return avg_accesses

def run_benchmark(dataset_path, model_path, max_entries=256, num_queries=1000, device='cpu'):
    """
    Run the full benchmark across all window sizes and methods.
    """
    print(f"\\n{'='*80}")
    print("NODE ACCESS BENCHMARK - Varying Relative Window Size")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print(f"Max Entries: {max_entries}")
    print(f"Queries per window size: {num_queries}")
    print(f"{'='*80}\\n")
    
    # Load dataset
    rectangles = load_rectangles_from_csv(dataset_path)
    global_mbr = compute_global_mbr(rectangles)
    print(f"Loaded {len(rectangles)} rectangles")
    print(f"Global MBR: {global_mbr}\\n")
    
    # Define relative window sizes (as percentages, converted to ratios)
    window_sizes = {
        '0.1%': 0.001,
        '1%': 0.01,
        '10%': 0.10,
        '50%': 0.50
    }
    
    # Results structure
    results = {}
    
    # Build trees (once per method)
    print("Building trees...\\n")
    
    # 1. Partitioning Model
    print(f"Building Partitioning R-Tree...")
    model = load_model(model_path, device)
    model_tree = PartitioningRTree(model, max_entries=max_entries, device=device)
    model_tree.insert_all(rectangles)
    print(f"  ✓ Partitioning R-Tree built\\n")
    
    # 2. R* Tree
    print(f"Building R* Tree...")
    rstar_tree = RStarTree(max_entries=max_entries)
    for i, r in enumerate(rectangles):
        rstar_tree.insert(i, r)
    print(f"  ✓ R* Tree built\\n")
    
    # 3. Guttman R-Tree
    print(f"Building Guttman R-Tree...")
    guttman_tree = RTreeGuttman(max_entries=max_entries)
    for i, r in enumerate(rectangles):
        guttman_tree.insert(i, r)
    print(f"  ✓ Guttman R-Tree built\\n")
    
    # Run benchmarks for each window size
    for window_label, window_ratio in sorted(window_sizes.items(), key=lambda x: x[1]):
        print(f"\\nTesting window size: {window_label} (ratio={window_ratio})")
        print("-" * 60)
        
        # Generate queries
        queries = generate_queries_for_window_size(global_mbr, window_ratio, num_queries)
        
        # Test each method
        results[window_label] = {}
        
        results[window_label]['Partitioning'] = benchmark_method(
            'Partitioning', model_tree, queries, window_label
        )
        
        results[window_label]['R*'] = benchmark_method(
            'R*', rstar_tree, queries, window_label
        )
        
        results[window_label]['Guttman'] = benchmark_method(
            'Guttman', guttman_tree, queries, window_label
        )
    
    return results

# ---------------- PLOTTING ----------------
def plot_results(results, dataset_name, output_path='node_access_plot.png'):
    """
    Plot average node accesses vs relative window size.
    """
    # Extract data
    window_labels = sorted(results.keys(), key=lambda x: float(x.rstrip('%')))
    window_percentages = [float(x.rstrip('%')) for x in window_labels]
    
    methods = ['Partitioning', 'R*', 'Guttman']
    
    plt.figure(figsize=(10, 6))
    
    for method in methods:
        avg_accesses = [results[wl][method] for wl in window_labels]
        
        marker = 's' if method == 'Partitioning' else ('o' if method == 'R*' else '^')
        plt.plot(window_percentages, avg_accesses, marker=marker, label=method, 
                linewidth=2, markersize=8)
    
    plt.xlabel('Relative Window Size (%)', fontsize=12)
    plt.ylabel('Average Node Accesses', fontsize=12)
    plt.title(f'Node Access Efficiency - {dataset_name}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xscale('log')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\\n✓ Plot saved to {output_path}")
    plt.close()

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark node access efficiency with varying window sizes"
    )
    parser.add_argument('--model', required=True, help='Path to trained .pth model')
    parser.add_argument('--dataset', required=True, help='Path to CSV dataset')
    parser.add_argument('--max_entries', type=int, default=256, help='Max entries per node (default: 256)')
    parser.add_argument('--num_queries', type=int, default=1000, help='Queries per window size (default: 1000)')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--output', default='node_access_results.json', help='Output JSON file')
    parser.add_argument('--plot', default='plots_node_access/node_access_plot.png', help='Output plot file')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        args.dataset,
        args.model,
        max_entries=args.max_entries,
        num_queries=args.num_queries,
        device=args.device
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\\n✓ Results saved to {args.output}")
    
    # Print summary table
    print(f"\\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Window Size':<15} {'Partitioning':<15} {'R*':<15} {'Guttman':<15}")
    print(f"{'-'*60}")
    for window_label in sorted(results.keys(), key=lambda x: float(x.rstrip('%'))):
        print(f"{window_label:<15} "
              f"{results[window_label]['Partitioning']:<15.2f} "
              f"{results[window_label]['R*']:<15.2f} "
              f"{results[window_label]['Guttman']:<15.2f}")
    
    # Generate plot
    dataset_name = os.path.basename(args.dataset).replace('.csv', '')
    plot_results(results, dataset_name, args.plot)
    
    print(f"\\n{'='*80}")
    print("✓ Benchmark complete!")
    print(f"{'='*80}\\n")

if __name__ == '__main__':
    main()
