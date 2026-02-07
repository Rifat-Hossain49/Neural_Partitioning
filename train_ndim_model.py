#!/usr/bin/env python
"""
Train N-Dimensional Partitioning Model

Wrapper script to train NPN on N-dimensional data.

Usage:
    python train_ndim_model.py --dataset data_50d.csv --output models/model_50d.pth --dim 50
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_custom import train_custom

def main():
    parser = argparse.ArgumentParser(description="Train N-Dimensional Partitioning Model")
    parser.add_argument('--dataset', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output model path')
    parser.add_argument('--dim', type=int, help='Dimension (optional, for verification)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--k', type=int, default=2, help='Partition factor K')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--samples', type=int, default=1000, help='Samples per epoch')
    
    args = parser.parse_args()
    
    print(f"Training N-Dim Model on {args.dataset}")
    print(f"Output: {args.output}")
    print(f"K={args.k}, Epochs={args.epochs}")
    
    # Train
    train_custom(
        csv_file=args.dataset,
        output_model=args.output,
        epochs=args.epochs,
        num_classes=args.k,
        min_k=6,   # Lower bound for supervision generator subset size
        max_k=10,  # Upper bound
        samples_per_file=args.samples,
        device=args.device,
        objective='range' # Range query optimization
    )

if __name__ == '__main__':
    main()
