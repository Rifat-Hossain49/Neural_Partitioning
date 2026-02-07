#!/usr/bin/env python
"""
Generate Synthetic N-Dimensional Dataset

Generates synthetic spatial data (points or small rectangles) in N dimensions
uniformly distributed in [0, 1]^d.

Usage:
    python generate_synthetic_ndim.py --dim 50 --size 20000 --output data_50d.csv
"""

import argparse
import numpy as np
import pandas as pd
import os

def generate_synthetic_data(dim, size, output_file, seed=42):
    """
    Generate synthetic uniform data in [0,1]^dim.
    """
    np.random.seed(seed)
    
    print(f"Generating {size} items of {dim} dimensions...")
    
    # Generate centers uniformly in [0, 1]
    centers = np.random.uniform(0, 1, size=(size, dim))
    
    # Make them small rectangles (or points if size=0)
    # Paper implies "synthetic datasets" often used points or specific distributions.
    # Uniform distribution usually implies points for index benchmarking unless specified.
    # But R-trees store rectangles. We can make small rectangles (width 0 or very small).
    # Let's make them points (min=max) to be safe and standard for high-dim synthetic benchmarks.
    mins = centers
    maxs = centers
    
    # Prepare DataFrame columns
    data = {}
    cols = []
    for d in range(dim):
        data[f'min_{d}'] = mins[:, d]
        data[f'max_{d}'] = maxs[:, d]
        cols.append(f'min_{d}')
        cols.append(f'max_{d}')
        
    df = pd.DataFrame(data)
    
    # Save to CSV
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) if os.path.dirname(output_file) else '.', exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✓ Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic N-Dimensional Dataset")
    parser.add_argument('--dim', type=int, required=True, help='Number of dimensions')
    parser.add_argument('--size', type=int, required=True, help='Number of data points')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    generate_synthetic_data(args.dim, args.size, args.output, args.seed)

if __name__ == '__main__':
    main()
