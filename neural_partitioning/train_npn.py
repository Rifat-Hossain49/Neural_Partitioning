"""
Training Script for Neural Partitioning Network (§5)

Trains the NPN to imitate the Exact DP optimal partitions.
Adapted for N-dimensionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse
from typing import List, Tuple
import time
import random
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path to ensure correct imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
# Use absolute imports relative to project root
from rtreelib.models import Rect
from neural_partitioning.model import PartitioningNetwork
from neural_partitioning.dp_supervisor import get_optimal_partition_labels
from neural_partitioning.utils import load_rectangles_from_csv


def set_global_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment_rectangles(rectangles: List[Rect], flip_prob: float = 0.5, 
                       rotate_prob: float = 0.5, scale_range: Tuple[float, float] = (0.8, 1.2)) -> List[Rect]:
    """
    Augment rectangles with random transformations for better generalization.
    N-dimensional implementation.
    """
    if not rectangles:
        return rectangles
    
    # Get dimensionality safely
    first_rect = rectangles[0]
    # Check for 'dims' property or fall back to len(min)
    dim = getattr(first_rect, 'dims', len(first_rect.min))
    
    augmented = []
    
    # Random scaling
    scale = np.random.uniform(*scale_range)
    
    # Random flips per dimension
    flips = [np.random.random() < flip_prob for _ in range(dim)]
    
    for rect in rectangles:
        # Safety check
        if not hasattr(rect, 'min') or not hasattr(rect, 'max'):
            augmented.append(rect)
            continue
            
        mins = rect.min.copy()
        maxs = rect.max.copy()
        
        # Apply scaling
        mins = mins * scale
        maxs = maxs * scale
        
        # Apply flips
        for d in range(dim):
            if flips[d]:
                mins[d], maxs[d] = -maxs[d], -mins[d]
        
        # Create new Rect with contiguous arrays
        new_rect = Rect(np.ascontiguousarray(mins), np.ascontiguousarray(maxs))
        augmented.append(new_rect)
    
    return augmented


def canonicalize_labels(rectangles: List[Rect], labels: List[int]) -> List[int]:
    """
    Renumber labels so that label 0 is the group with smallest min coords vector.
    This makes the learning problem permutation-invariant.
    N-dimensional implementation.
    """
    if not labels or not rectangles:
        return labels
        
    unique_labels = sorted(list(set(labels)))
    group_stats = []
    
    # Get dimensionality
    first_rect = rectangles[0]
    dim = getattr(first_rect, 'dims', len(first_rect.min))
    
    for lbl in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == lbl]
        if not indices:
            continue
        
        # Use group min coords vector as sort key
        # key = (min(dim0_min), min(dim1_min), ..., min(dimN_min))
        min_coords = []
        for d in range(dim):
            # Safe extraction of coordinates
            vals = [rectangles[i].min[d] for i in indices if hasattr(rectangles[i], 'min')]
            if vals:
                min_coords.append(min(vals))
            else:
                min_coords.append(0.0)
            
        group_stats.append((lbl, tuple(min_coords)))
        
    # Sort groups lexicographically by min_coords vector
    group_stats.sort(key=lambda x: x[1])
    
    # Create mapping old -> new
    mapping = {old: new for new, (old, _) in enumerate(group_stats)}
    
    return [mapping[l] for l in labels]


def generate_policy_training_samples(
    rectangles: List[Rect],
    min_k: int = 5,
    max_k: int = 8,
    max_groups: int = 4,
    samples_per_file: int = 1000,
    augment: bool = True
) -> List[Tuple[List[Rect], List[int]]]:
    """
    Generate training samples (subset, labels).
    Offline generation logic from reference.
    """
    n = len(rectangles)
    samples = []
    
    print(f"  Generating {samples_per_file} samples from {n} rectangles...")
    
    for _ in tqdm(range(samples_per_file), desc="  Generating samples"):
        # Random subset size
        k = np.random.randint(min_k, max_k + 1)
        
        # Sliding Window Sampling (Preserves Locality)
        if n <= k:
            indices = list(range(n))
        else:
            # Pick a random start index for sliding window
            # Preserves spatial locality from the input file
            start = random.randint(0, n - k)
            indices = list(range(start, start + k))
            
        subset_rects = [rectangles[i] for i in indices]
        
        # Augment
        if augment:
            subset_rects = augment_rectangles(subset_rects)
        
        # Compute optimal labels via DP
        try:
            labels = get_optimal_partition_labels(
                subset_rects, 
                max_groups=max_groups,
                objective='range'  # Default to 'range' (REC)
            )
            if labels is not None:
                # Canonicalize labels (CRITICAL)
                labels = canonicalize_labels(subset_rects, labels)
                samples.append((subset_rects, labels))
        except Exception:
            # Skip invalid samples
            continue
            
    return samples


def train_policy(
    csv_files: List[str],
    epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    hidden_dim: int = 64,
    num_classes: int = 4,
    device: str = 'cpu',
    output_model: str = 'models_paper/split_policy_model.pth',
    seed: int = 42,
    min_k: int = 10,
    max_k: int = 14,
    samples_per_file: int = 200,
    augment: bool = True
):
    """
    Train the policy model.
    """
    print("TRAINING NEURAL PARTITIONING POLICY MODEL")
    print("="*60)
    print(f"Files: {csv_files}")
    print(f"Device: {device}")
    print(f"k range: [{min_k}, {max_k}]")
    print(f"Max groups (classes): {num_classes}")
    print(f"Samples per file: {samples_per_file}")
    
    device = torch.device(device)
    set_global_seed(seed)
    
    # Load data
    print("\nLoading data and generating samples...")
    print("(Note: Exact DP solver is running. Speed is limited by O(k^M) complexity.)")
    all_samples = []
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}...")
        rectangles = load_rectangles_from_csv(csv_file)
        print(f"  Loaded {len(rectangles)} rectangles")
        
        # Generate samples
        samples = generate_policy_training_samples(
            rectangles, 
            min_k=min_k, 
            max_k=max_k, 
            max_groups=num_classes,
            samples_per_file=samples_per_file,
            augment=augment
        )
        all_samples.extend(samples)
        print(f"  Generated {len(samples)} valid samples")
        
    print(f"\nTotal training samples: {len(all_samples)}")
    
    if not all_samples:
        print("ERROR: No training samples generated. Check parameters.")
        return
        
    # Determine input dimension from data (N-dim support)
    input_dim = None
    if all_samples:
        # Check first sample's first rectangle
        rect = all_samples[0][0][0]
        # Use property access
        dim = getattr(rect, 'dims', len(rect.min))
        input_dim = 4 * dim + 1
    else:
        # Fallback (2D default)
        input_dim = 9
    
    print(f"Detected Data Dimensionality: {dim if 'dim' in locals() else 'Unknown'}D")
    print(f"Model Input Dimension: {input_dim}")
    
    # Model
    model = PartitioningNetwork(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nStarting training...\n")
    losses = []
    
    for epoch in range(epochs):
        np.random.shuffle(all_samples)
        epoch_loss = 0.0
        batches = 0
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        count = 0
        
        pbar = tqdm(all_samples, desc=f"Epoch {epoch+1}/{epochs}")
        
        for subset_rects, labels in pbar:
            # Forward
            logits = model(subset_rects) # (N, num_classes)
            
            # Target
            target = torch.tensor(labels, dtype=torch.long, device=device) # (N,)
            
            # Loss (CrossEntropy)
            loss = criterion(logits, target)
            loss.backward()
            
            accumulated_loss += loss.item()
            count += 1
            
            # Gradient accumulation
            if count >= batch_size:
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += accumulated_loss / count
                batches += 1
                accumulated_loss = 0.0
                count = 0
                
                pbar.set_postfix({'loss': epoch_loss / batches if batches > 0 else 0})
                
        # Final batch
        if count > 0:
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += accumulated_loss / count
            batches += 1
            
        avg_loss = epoch_loss / batches if batches > 0 else 0
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
        
    # Save with config
    os.makedirs(os.path.dirname(output_model) if os.path.dirname(output_model) else '.', exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'hidden_dim': hidden_dim,
            'num_classes': num_classes,
            'input_dim': input_dim,
            'dataset_dim': dim if 'dim' in locals() else 2
        }
    }
    torch.save(checkpoint, output_model)
    print(f"\n✓ Model saved to {output_model}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.title("NPN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    plot_path = output_model.replace('.pth', '_loss.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss plot saved to {plot_path}")


def main():
    import glob
    parser = argparse.ArgumentParser(description="Train Neural Partitioning Policy")
    parser.add_argument('--csv_files', nargs='+', required=True, help="Input CSV files")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='models_paper/npn_model.pth')
    parser.add_argument('--min_k', type=int, default=6)
    parser.add_argument('--max_k', type=int, default=12)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--num_classes', type=int, default=3, help='Max split groups')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--no_augment', action='store_true')
    # dim argument removed in reference, inferred from data. But we can add for clarity? 
    # Reference didn't use it. But load_rectangles might need specific CSV format?
    # Our load_rectangles doesn't require dim arg (it infers from columns).
    # So we don't strictly need --dim.
    
    args = parser.parse_args()
    
    expanded_files = []
    for p in args.csv_files:
        matches = glob.glob(p)
        if matches:
            expanded_files.extend(matches)
        else:
            expanded_files.append(p)
            
    train_policy(
        expanded_files, 
        epochs=args.epochs, 
        device=args.device, 
        output_model=args.output,
        min_k=args.min_k,
        max_k=args.max_k,
        samples_per_file=args.samples,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        augment=not args.no_augment
    )

if __name__ == '__main__':
    main()
