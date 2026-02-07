#!/usr/bin/env python
"""
Arizona State-of-the-Art Benchmark - Fixed Window Width

Compares Partitioning Model node access efficiency against state-of-the-art
baselines (H4R, HR, PR, R*, STR, TGS, ACR) on the Arizona dataset.

Based on paper's Section "Comparison with State-of-the-Art Baselines".
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

def generate_queries_fixed_width(global_mbr, window_width_meters, num_queries=1000):
    """
    Generate random range queries with fixed window width in meters.
    
    Args:
        global_mbr: Global bounding rectangle of dataset
        window_width_meters: Fixed window width in meters
        num_queries: Number of queries to generate
        
    Returns:
        List of query rectangles
    """
    extent = global_mbr.max - global_mbr.min
    
    queries = []
    for _ in range(num_queries):
        # Random center position
        center = np.random.rand(2)
        
        # For 2D case, assume coordinates are in meters or similar
        # Create square query window
        half_width = window_width_meters / 2
        
        # Convert to normalized space
        half_w_norm = half_width / extent[0]
        half_h_norm = half_width / extent[1]
        
        # Clamp center to valid range
        cx = max(half_w_norm, min(1 - half_w_norm, center[0]))
        cy = max(half_h_norm, min(1 - half_h_norm, center[1]))
        
        # Convert to actual coordinates
        center_x = global_mbr.min[0] + cx * extent[0]
        center_y = global_mbr.min[1] + cy * extent[1]
        
        q_min = np.array([center_x - half_width, center_y - half_width])
        q_max = np.array([center_x + half_width, center_y + half_width])
        
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

def load_baselines(baseline_file='arizona_baselines.json'):
    """Load baseline comparison data."""
    with open(baseline_file, 'r') as f:
        data = json.load(f)
    return data

def benchmark_partitioning_model(dataset_path, model_path, max_entries=102, 
                                 window_widths=[100, 200, 400, 800, 1600],
                                 num_queries=1000, device='cpu'):
    """
    Benchmark the partitioning model on Arizona dataset.
    
    Args:
        dataset_path: Path to Arizona CSV dataset
        model_path: Path to trained model
        max_entries: Node capacity (default: 102 for Arizona comparison)
        window_widths: List of window widths in meters
        num_queries: Queries per window width
        device: Device to use
        
    Returns:
        Dictionary of results
    """
    print(f"\\n{'='*80}")
    print("ARIZONA STATE-OF-THE-ART BENCHMARK")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print(f"Node Capacity (M): {max_entries}")
    print(f"Window Widths: {window_widths} meters")
    print(f"Queries per width: {num_queries}")
    print(f"{'='*80}\\n")
    
    # Load dataset
    print("Loading dataset...")
    rectangles = load_rectangles_from_csv(dataset_path)
    global_mbr = compute_global_mbr(rectangles)
    print(f"Loaded {len(rectangles):,} rectangles")
    print(f"Global MBR: {global_mbr}\\n")
    
    # Build tree
    print("Building Partitioning R-Tree...")
    model = load_model(model_path, device)
    tree = PartitioningRTree(model, max_entries=max_entries, device=device)
    tree.insert_all(rectangles)
    print(f"✓ Tree built\\n")
    
    # Run benchmarks for each window width
    results = {}
    
    for width in window_widths:
        print(f"Testing window width: {width}m")
        print("-" * 60)
        
        # Generate queries
        queries = generate_queries_fixed_width(global_mbr, width, num_queries)
        
        # Count node accesses
        wrap_tree_for_counting(tree)
        total_accesses = 0
        
        for query in queries:
            tracker.reset()
            list(tree.query(query))
            total_accesses += tracker.access_count
        
        avg_accesses = total_accesses / len(queries)
        results[f"{width}m"] = total_accesses
        
        print(f"  Total node accesses: {total_accesses:,.0f}")
        print(f"  Average per query: {avg_accesses:.2f}\\n")
    
    return results

def generate_comparison_table(partitioning_results, baselines):
    """Generate a comparison table with all methods."""
    print(f"\\n{'='*80}")
    print("COMPARISON TABLE - Node Access (vs Window Width)")
    print(f"{'='*80}")
    
    # Header
    widths = ["100m", "200m", "400m", "800m", "1600m"]
    print(f"{'Method':<20} " + " ".join([f"{w:>12}" for w in widths]))
    print("-" * 80)
    
    # Baseline methods
    methods = ["H4R", "HR", "PR", "R*", "STR", "TGS", "ACR", "Ours (Paper)"]
    for method in methods:
        if method in baselines['baselines']:
            values = [baselines['baselines'][method].get(w, 0) for w in widths]
            print(f"{method:<20} " + " ".join([f"{v:>12.1e}" for v in values]))
    
    print("-" * 80)
    
    # Our result
    our_values = [partitioning_results.get(w, 0) for w in widths]
    print(f"{'Ours (Current)':<20} " + " ".join([f"{v:>12.1e}" for v in our_values]))
    
    print(f"{'='*80}\\n")

def plot_comparison(partitioning_results, baselines, output_path='arizona_comparison.png'):
    """Create a comparison plot."""
    widths = [100, 200, 400, 800, 1600]
    width_labels = ["100m", "200m", "400m", "800m", "1600m"]
    
    plt.figure(figsize=(12, 7))
    
    # Plot baselines
    methods = ["H4R", "HR", "PR", "R*", "STR", "TGS", "ACR", "Ours (Paper)"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods) + 1))
    
    for i, method in enumerate(methods):
        if method in baselines['baselines']:
            values = [baselines['baselines'][method].get(f"{w}m", 0) for w in widths]
            linestyle = '--' if method == "Ours (Paper)" else '-'
            alpha = 1.0 if method == "Ours (Paper)" else 0.6
            plt.plot(widths, values, marker='o', label=method, 
                    color=colors[i], linestyle=linestyle, alpha=alpha, linewidth=2)
    
    # Plot our result
    our_values = [partitioning_results.get(f"{w}m", 0) for w in widths]
    plt.plot(widths, our_values, marker='s', label='Ours (Current)', 
            color=colors[-1], linestyle='-', linewidth=2.5, markersize=10)
    
    plt.xlabel('Window Width (meters)', fontsize=13)
    plt.ylabel('Total Nodes Accessed', fontsize=13)
    plt.title('Arizona Dataset - State-of-the-Art Comparison\\n(N=1,464,257, M=102)', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(widths, width_labels)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    plt.close()

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Arizona State-of-the-Art Benchmark"
    )
    parser.add_argument('--model', required=True, help='Path to trained .pth model')
    parser.add_argument('--dataset', required=True, help='Path to Arizona CSV dataset')
    parser.add_argument('--max_entries', type=int, default=102, help='Max entries per node (default: 102)')
    parser.add_argument('--num_queries', type=int, default=1000, help='Queries per window width (default: 1000)')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--baselines', default='arizona_baselines.json', help='Baseline data JSON file')
    parser.add_argument('--output', default='arizona_results.json', help='Output JSON file')
    parser.add_argument('--plot', default='plots_arizona/arizona_comparison.png', help='Output plot file')
    
    args = parser.parse_args()
    
    # Load baselines
    baselines = load_baselines(args.baselines)
    
    # Run benchmark
    window_widths = baselines['window_widths_meters']
    results = benchmark_partitioning_model(
        args.dataset,
        args.model,
        max_entries=args.max_entries,
        window_widths=window_widths,
        num_queries=args.num_queries,
        device=args.device
    )
    
    # Save results
    output_data = {
        'method': 'Ours (Current)',
        'dataset': args.dataset,
        'num_objects': len(load_rectangles_from_csv(args.dataset)),
        'max_entries': args.max_entries,
        'num_queries': args.num_queries,
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✓ Results saved to {args.output}\\n")
    
    # Generate comparison table
    generate_comparison_table(results, baselines)
    
    # Generate plot
    plot_comparison(results, baselines, args.plot)
    
    print(f"\\n{'='*80}")
    print("✓ Benchmark complete!")
    print(f"{'='*80}\\n")

if __name__ == '__main__':
    main()
