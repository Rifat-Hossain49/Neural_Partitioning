#!/usr/bin/env python
"""
Impact of Dimensionality Ablation Study

Benchmarks query latency across varying dimensions d in {2, 8, 20, 50, 100}.
Reproduces the "Impact of Dimensionality" ablation from the paper.

Usage:
    python benchmark_impact_of_dims.py
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import subprocess

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_partitioning.inference import PartitioningRTree
from neural_partitioning.model import PartitioningNetwork
from neural_partitioning.utils import load_rectangles_from_csv, generate_random_queries
from rtreelib.models import Rect

def benchmark_dimension(dim, size=20000, query_count=10000, max_entries=512, epochs=20, device='cpu', 
                        output_dir='ablation_dims', skip_training=False):
    """
    Run full benchmark pipeline for a single dimension.
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARKING DIMENSION: {dim} (N={size})")
    print(f"{'='*60}")
    
    # 1. Paths
    dataset_path = os.path.join(output_dir, f"data_{dim}d.csv")
    model_path = os.path.join(output_dir, f"model_{dim}d.pth")
    
    # 2. Generate Data
    if not os.path.exists(dataset_path):
        print(f"Generating {dim}D dataset...")
        subprocess.check_call([
            sys.executable, 'generate_synthetic_ndim.py',
            '--dim', str(dim),
            '--size', str(size),
            '--output', dataset_path
        ])
    else:
        print(f"Dataset {dataset_path} exists.")
        
    # 3. Train Model
    if not skip_training and (not os.path.exists(model_path)):
        print(f"Training {dim}D model...")
        subprocess.check_call([
            sys.executable, 'train_ndim_model.py',
            '--dataset', dataset_path,
            '--output', model_path,
            '--dim', str(dim),
            '--epochs', str(epochs),
            '--k', '2', # Default K=2
            '--device', device,
            '--samples', '1000'
        ])
    else:
        print(f"Model {model_path} exists or training skipped.")
        
    # 4. Benchmark
    print("Running benchmark...")
    
    # Load data
    rectangles = load_rectangles_from_csv(dataset_path)
    print(f"  Loaded {len(rectangles)} items.")
    
    # Load model
    print("  Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    input_dim = checkpoint['config'].get('input_dim', 4*dim+1)
    hidden_dim = checkpoint['config'].get('hidden_dim', 256) # Default to 256 if not found (train_custom default)
    model = PartitioningNetwork(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Build Tree
    print(f"  Building index (M={max_entries})...")
    start_build = time.time()
    tree = PartitioningRTree(model, max_entries=max_entries, device=device)
    tree.insert_all(rectangles)
    build_time = time.time() - start_build
    print(f"  Built in {build_time:.2f}s")
    
    # Generate Queries
    print(f"  Generating {query_count} random queries...")
    queries = generate_random_queries(rectangles, query_count)
    
    # Run Queries
    print("  Executing queries...")
    start_q = time.time()
    for q in queries:
        list(tree.query(q))
    total_q_time = time.time() - start_q
    avg_ms = (total_q_time / query_count) * 1000
    
    print(f"  ✓ Avg Latency: {avg_ms:.4f} ms")
    return avg_ms

def plot_results(results, output_file):
    """Plot latency vs dimension."""
    dims = sorted(results.keys())
    latencies = [results[d] for d in dims]
    
    plt.figure(figsize=(8, 6))
    plt.plot(dims, latencies, marker='o', linewidth=2, color='#1f77b4', markersize=8)
    plt.xlabel('Dimension d', fontsize=12)
    plt.ylabel('Avg Query Latency (ms)', fontsize=12)
    plt.title('Impact of Dimensionality (M=512, N=20k)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate
    for d, lat in zip(dims, latencies):
        plt.annotate(f"{lat:.2f}", (d, lat), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.tight_layout()
    # Create directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Impact of Dimensionality Benchmark")
    parser.add_argument('--dims', default="2,8,20,50,100", help="Comma-separated dimensions")
    parser.add_argument('--output_dir', default="ablation_dims", help="Directory for data/models")
    parser.add_argument('--plot_output', default="plots_hyper_param_k/impact_of_dims.png", help="Output plot path")
    parser.add_argument('--device', default="cpu", help="Device")
    parser.add_argument('--epochs', type=int, default=20, help="Training epochs")
    parser.add_argument('--max_entries', type=int, default=512, help="Max entries M")
    parser.add_argument('--dataset_size', type=int, default=20000, help="Dataset size")
    parser.add_argument('--query_count', type=int, default=10000, help="Query count")
    parser.add_argument('--skip_training', action='store_true', help="Skip training")
    
    args = parser.parse_args()
    
    dims = [int(d) for d in args.dims.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    try:
        for d in dims:
            latency = benchmark_dimension(
                d, 
                size=args.dataset_size,
                query_count=args.query_count,
                max_entries=args.max_entries,
                epochs=args.epochs,
                device=args.device,
                output_dir=args.output_dir,
                skip_training=args.skip_training
            )
            results[d] = latency
            
        print("\nSUMMARY RESULTS (Avg Latency ms):")
        print("Dimension | Latency (ms)")
        print("----------|-------------")
        for d in dims:
            print(f"{d:<9} | {results[d]:.4f}")
            
        # Save JSON
        json_path = os.path.join(args.output_dir, "results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Plot
        plot_results(results, args.plot_output)
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")

if __name__ == '__main__':
    main()
