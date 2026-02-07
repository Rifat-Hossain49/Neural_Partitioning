#!/usr/bin/env python
"""
Hyperparameter K Variation Study

Trains models with different partition factors K and benchmarks their query latency.
Reproduces the "Impact of Partition Factor" experiment from the paper.

Usage:
    python benchmark_hyper_param_k.py \
        --train_dataset twitter_100k.csv \
        --test_dataset twitter_100k.csv \
        --k_values "2,3,4" \
        --max_entries 512 \
        --epochs 100 \
        --query_count 10000
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_custom import train_custom
from rtreelib.models import Rect
from neural_partitioning.model import PartitioningNetwork
from neural_partitioning.inference import PartitioningRTree
from neural_partitioning.utils import load_rectangles_from_csv

# ---------------- TRAINING ----------------
def train_model_for_k(train_dataset, k_value, epochs, output_dir, device='cpu'):
    """
    Train a partitioning model with specific partition factor K.
    
    Args:
        train_dataset: Path to training CSV
        k_value: Partition factor (number of classes)
        epochs: Training epochs
        output_dir: Directory to save model
        device: Device to use
        
    Returns:
        Path to trained model
    """
    model_name = f"model_k{k_value}.pth"
    model_path = os.path.join(output_dir, model_name)
    
    print(f"\n{'='*80}")
    print(f"Training Model with K={k_value}")
    print(f"{'='*80}")
    
    # Train using train_custom
    train_custom(
        csv_file=train_dataset,
        output_model=model_path,
        epochs=epochs,
        num_classes=k_value,
        min_k=6,
        max_k=8,
        samples_per_file=1000,
        batch_size=32,
        learning_rate=1e-3,
        hidden_dim=128,
        device=device,
        seed=42,
        objective='range',
        k_knn=10
    )
    
    print(f"✓ Model saved to {model_path}\n")
    return model_path

# ---------------- BENCHMARKING ----------------
def benchmark_model(model_path, test_dataset, max_entries, query_count, device='cpu'):
    """
    Benchmark a trained model's query latency.
    
    Args:
        model_path: Path to trained model
        test_dataset: Path to test CSV
        max_entries: Node capacity
        query_count: Number of queries
        device: Device to use
        
    Returns:
        Average query latency in milliseconds
    """
    print(f"Benchmarking {os.path.basename(model_path)}...")
    
    # Load dataset
    rectangles = load_rectangles_from_csv(test_dataset)
    print(f"  Loaded {len(rectangles):,} rectangles")
    
    # Load model
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
    
    # Build tree
    print(f"  Building R-tree with M={max_entries}...")
    start_build = time.time()
    tree = PartitioningRTree(model, max_entries=max_entries, device=device)
    tree.insert_all(rectangles)
    build_time = time.time() - start_build
    print(f"  ✓ Tree built in {build_time:.2f}s")
    
    # Generate queries
    # Compute global MBR
    mins = np.array([r.min for r in rectangles])
    maxs = np.array([r.max for r in rectangles])
    global_min = np.min(mins, axis=0)
    global_max = np.max(maxs, axis=0)
    extent = global_max - global_min
    
    # Generate random range queries
    queries = []
    for _ in range(query_count):
        # Random window size (0.1% to 1% of global extent)
        window_size = np.random.uniform(0.001, 0.01) * extent
        
        # Random center
        center = global_min + np.random.rand(len(global_min)) * extent
        
        # Create query rectangle
        q_min = np.maximum(center - window_size/2, global_min)
        q_max = np.minimum(center + window_size/2, global_max)
        
        queries.append(Rect(q_min, q_max))
    
    # Run queries
    print(f"  Running {query_count:,} queries...")
    start_query = time.time()
    for q in queries:
        list(tree.query(q))
    query_time = time.time() - start_query
    
    avg_latency_ms = (query_time / query_count) * 1000
    
    print(f"  ✓ Average query latency: {avg_latency_ms:.2f}ms\n")
    
    return avg_latency_ms

# ---------------- PLOTTING ----------------
def plot_results(results, dataset_name, output_path='ablation_partition_factor.png'):
    """
    Plot query latency vs partition factor K.
    
    Args:
        results: Dict mapping K -> latency_ms
        dataset_name: Name of dataset for title
        output_path: Output plot path
    """
    k_values = sorted(results.keys())
    latencies = [results[k] for k in k_values]
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, latencies, marker='o', color='#d62728', 
             linewidth=2.5, markersize=10)
    
    plt.xlabel('Partition Factor K', fontsize=13)
    plt.ylabel('Latency (ms)', fontsize=13)
    plt.title(f'{dataset_name}', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(k_values)
    
    # Add value labels on points
    for k, lat in zip(k_values, latencies):
        plt.annotate(f'{lat:.2f}', (k, lat), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    plt.close()

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter K Variation Study"
    )
    
    # Data
    parser.add_argument('--train_dataset', required=True, help='Training dataset CSV')
    parser.add_argument('--test_dataset', required=True, help='Test dataset CSV')
    
    # K values
    parser.add_argument('--k_values', default='2,3,4', help='Comma-separated K values (default: 2,3,4)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
    parser.add_argument('--output_dir', default='ablation_models', help='Output directory for models')
    
    # Benchmarking parameters
    parser.add_argument('--max_entries', type=int, default=512, help='Node capacity M (default: 512)')
    parser.add_argument('--query_count', type=int, default=10000, help='Number of queries (default: 10000)')
    
    # Other
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, use existing models')
    parser.add_argument('--output_json', default='ablation_results.json', help='Output JSON file')
    parser.add_argument('--output_plot', default='plots_hyper_param_k/k_variation_plot.png', help='Output plot file')
    
    args = parser.parse_args()
    
    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(',')]
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER K VARIATION STUDY")
    print(f"{'='*80}")
    print(f"Training Dataset: {args.train_dataset}")
    print(f"Test Dataset: {args.test_dataset}")
    print(f"K Values: {k_values}")
    print(f"Max Entries (M): {args.max_entries}")
    print(f"Epochs: {args.epochs}")
    print(f"Query Count: {args.query_count}")
    print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    # Train and benchmark each K
    for k in k_values:
        model_path = os.path.join(args.output_dir, f"model_k{k}.pth")
        
        # Train if needed
        if not args.skip_training or not os.path.exists(model_path):
            model_path = train_model_for_k(
                args.train_dataset,
                k,
                args.epochs,
                args.output_dir,
                args.device
            )
        else:
            print(f"\n✓ Using existing model: {model_path}\n")
        
        # Benchmark
        latency = benchmark_model(
            model_path,
            args.test_dataset,
            args.max_entries,
            args.query_count,
            args.device
        )
        
        results[k] = latency
    
    # Save results
    output_data = {
        'train_dataset': args.train_dataset,
        'test_dataset': args.test_dataset,
        'max_entries': args.max_entries,
        'query_count': args.query_count,
        'results': {str(k): v for k, v in results.items()}
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✓ Results saved to {args.output_json}\n")
    
    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'K':<10} {'Latency (ms)':<15}")
    print("-" * 30)
    for k in sorted(results.keys()):
        print(f"{k:<10} {results[k]:<15.2f}")
    print(f"{'='*80}\n")
    
    # Plot
    dataset_name = os.path.basename(args.test_dataset).replace('.csv', '')
    plot_results(results, dataset_name, args.output_plot)
    
    print(f"\n{'='*80}")
    print("✓ Ablation study complete!")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
