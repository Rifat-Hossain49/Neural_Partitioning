
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from typing import List, Tuple
from tqdm import tqdm
import random
import sys

# Add parent directory to path to ensure correct imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_partitioning.model import PartitioningNetwork
from neural_partitioning.dp_supervisor import get_optimal_partition_labels
from neural_partitioning.utils import load_rectangles_from_csv
from neural_partitioning.train_npn import canonicalize_labels, augment_rectangles, set_global_seed

def generate_custom_samples(
    rectangles,
    min_k=6,
    max_k=8,
    max_groups=3,
    samples_per_file=1000,
    augment=True,
    objective='range',
    k_knn=1
):
    """
    Generate training samples with specific objective.
    """
    n = len(rectangles)
    samples = []
    
    for _ in tqdm(range(samples_per_file), desc=f"Generating {objective} samples"):
        k = np.random.randint(min_k, max_k + 1)
        
        if n <= k:
            indices = list(range(n))
        else:
            start = random.randint(0, n - k)
            indices = list(range(start, start + k))
            
        subset_rects = [rectangles[i] for i in indices]
        
        if augment:
            subset_rects = augment_rectangles(subset_rects)
        
        try:
            labels = get_optimal_partition_labels(
                subset_rects, 
                max_groups=max_groups,
                objective=objective,
                k_knn=k_knn  # Important for KNN objective
            )
            if labels is not None:
                labels = canonicalize_labels(subset_rects, labels)
                samples.append((subset_rects, labels))
        except Exception:
            continue
            
    return samples

def train_custom(
    csv_file: str,
    objective: str = 'range',
    k_knn: int = 10,
    epochs: int = 10,  # Reduced default for speed, user can override
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    hidden_dim: int = 256,  # Increased default
    num_classes: int = 3,
    min_k: int = 6,
    max_k: int = 8,
    device: str = 'cpu',
    output_model: str = 'model.pth',
    seed: int = 42,
    samples_per_file: int = 500
):
    print(f"TRAINING Custom NPN: Objective={objective}, File={csv_file}")
    print(f"  Params: k={num_classes}, subset=[{min_k}, {max_k}], epochs={epochs}")
    
    device_obj = torch.device(device)
    set_global_seed(seed)
    
    # Load data
    rectangles = load_rectangles_from_csv(csv_file)
    print(f"Loaded {len(rectangles)} rectangles")
    
    # Generate samples
    all_samples = generate_custom_samples(
        rectangles, 
        min_k=min_k, max_k=max_k, 
        max_groups=num_classes,
        samples_per_file=samples_per_file,
        augment=True,
        objective=objective,
        k_knn=k_knn
    )
    
    if not all_samples:
        print("Error: No samples generated.")
        return

    # Infer dims
    first_rect = all_samples[0][0][0]
    dim = len(first_rect.min)
    input_dim = 4 * dim + 1
    
    model = PartitioningNetwork(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model.to(device_obj)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    for epoch in range(epochs):
        np.random.shuffle(all_samples)
        epoch_loss = 0.0
        batches = 0
        
        curr_batch_loss = 0.0
        count = 0
        
        for subset_rects, labels in all_samples:
            logits = model(subset_rects)
            target = torch.tensor(labels, dtype=torch.long, device=device_obj)
            
            loss = criterion(logits, target)
            loss.backward()
            
            curr_batch_loss += loss.item()
            count += 1
            
            if count >= batch_size:
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += curr_batch_loss / count
                batches += 1
                curr_batch_loss = 0.0
                count = 0
                
        if count > 0:
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += curr_batch_loss / count
            batches += 1
            
        avg_loss = epoch_loss / batches if batches else 0
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
            
    # Save
    os.makedirs(os.path.dirname(output_model) if os.path.dirname(output_model) else '.', exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'hidden_dim': hidden_dim,
            'num_classes': num_classes,
            'input_dim': input_dim
        }
    }
    torch.save(checkpoint, output_model)
    print(f"Saved model to {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--objective', default='range', choices=['range', 'knn'])
    parser.add_argument('--k_knn', type=int, default=10)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--device', default='cpu')
    
    parser.add_argument('--num_classes', type=int, default=3, help="Partition factor k")
    parser.add_argument('--min_k', type=int, default=6)
    parser.add_argument('--max_k', type=int, default=8)
    
    args = parser.parse_args()
    
    train_custom(
        args.csv, 
        objective=args.objective, 
        k_knn=args.k_knn, 
        output_model=args.output,
        epochs=args.epochs,
        samples_per_file=args.samples,
        device=args.device,
        num_classes=args.num_classes,
        min_k=args.min_k,
        max_k=args.max_k
    )
