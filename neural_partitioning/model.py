"""
Neural Partitioning Network (NPN) for Learned R-tree Splitting (§4)

Implements a PointNet-based model that predicts the optimal K-way partition
for a set of rectangles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtreelib.models import Rect

class PartitioningNetwork(nn.Module):
    """
    Neural Partitioning Network (NPN) - PointNet-based architecture.
    See Section 4.2 for details.
    
    Architecture:
    1. Local MLP (f_loc): (N, 4d+1) -> (N, H)
    2. Global Context (g): componentwise max-pooling
    3. Concatenation: h_j + g -> (N, 2H)
    4. Partitioning MLP (f_partition): (N, 2H) -> (N, K) logits
    """
    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()

        self.local_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        

        self.partition_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes) # Logits for multi-way partitioning
        )
        
    def forward(self, rectangles: List[Rect]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rectangles: List of N rectangles
            
        Returns:
            logits: (N, 1) tensor of split logits
        """
        device = next(self.parameters()).device
        n = len(rectangles)
        
        if n == 0:
            return torch.empty(0, 1, device=device)
            
        # Extract features (N, 9)
        features = self._extract_features(rectangles).to(device)
        

        local_feats = self.local_mlp(features)
        
        global_feat, _ = torch.max(local_feats, dim=0, keepdim=True) # (1, H)
        
        global_feat_expanded = global_feat.expand(n, -1)
        
        combined_feats = torch.cat([local_feats, global_feat_expanded], dim=1)
        
        logits = self.partition_mlp(combined_feats)
        return logits

    def _extract_features(self, rectangles: List[Rect]) -> torch.Tensor:
        """Extract normalized features for N-dimensions (4*d + 1)."""
        n = len(rectangles)
        if n == 0:
            return torch.empty(0, 0)

        # Determine dimension from first rectangle
        ndim = rectangles[0].dims
        
        # Vectorized implementation
        mins = np.array([r.min for r in rectangles])
        maxs = np.array([r.max for r in rectangles])
        
        # Calculate scene bounds
        scene_min = np.min(mins, axis=0)
        scene_max = np.max(maxs, axis=0)
        scene_extent = np.maximum(1e-9, scene_max - scene_min)
        
        # Normalization
        n_mins = (mins - scene_min) / scene_extent
        n_maxs = (maxs - scene_min) / scene_extent
        
        sizes = n_maxs - n_mins
        centers = (n_mins + n_maxs) / 2
        
        # Volumes (N, 1)
        volumes = np.prod(sizes, axis=1, keepdims=True)
        
        # Concatenate: min, max, size, volume, center
        features = np.concatenate([n_mins, n_maxs, sizes, volumes, centers], axis=1)
            
        return torch.tensor(features, dtype=torch.float32)



class MultiHeadPartitioningNetwork(nn.Module):
    """
    Multi-head Neural Partitioning Network for adaptive K-way splitting.
    
    Architecture:
    1. Shared backbone: Local MLP + Global Context + Concatenation
    2. Multiple heads: One classification head per arity K (K=2,3,4,...)
    """
    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, k_min: int = 2, k_max: int = 4):
        super().__init__()
        
        self.k_min = k_min
        self.k_max = k_max
        self.input_dim = input_dim # Store input_dim
        
        # Shared backbone
        self.local_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Shared intermediate processing
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Multiple heads: one per K value
        self.heads = nn.ModuleDict()
        for k in range(k_min, k_max + 1):
            self.heads[f'head_{k}'] = nn.Linear(hidden_dim, k)
    
    def forward(self, rectangles: List[Rect], k: int = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rectangles: List of N rectangles
            k: Number of split groups (if None, returns dict of all heads)
            
        Returns:
            logits: (N, k) tensor of split logits, or dict of {k: logits}
        """
        device = next(self.parameters()).device
        n = len(rectangles)
        
        if n == 0:
            if k is not None:
                return torch.empty(0, k, device=device)
            return {f'head_{kk}': torch.empty(0, kk, device=device) for kk in range(self.k_min, self.k_max + 1)}
            
        # Extract features (N, 9)
        features = self._extract_features(rectangles).to(device)
        
        local_feats = self.local_mlp(features)
        
        global_feat, _ = torch.max(local_feats, dim=0, keepdim=True) # (1, H)
        
        global_feat_expanded = global_feat.expand(n, -1)
        
        combined_feats = torch.cat([local_feats, global_feat_expanded], dim=1)
        
        shared_feats = self.shared_mlp(combined_feats)
        
        if k is not None:
            # Single head inference
            if k < self.k_min or k > self.k_max:
                raise ValueError(f"k={k} out of range [{self.k_min}, {self.k_max}]")
            return self.heads[f'head_{k}'](shared_feats)
        else:
            # Multi-head output for training
            outputs = {}
            for kk in range(self.k_min, self.k_max + 1):
                outputs[f'head_{kk}'] = self.heads[f'head_{kk}'](shared_feats)
            return outputs
    
    def _extract_features(self, rectangles: List[Rect]) -> torch.Tensor:
        """Extract normalized features for N-dimensions (4*d + 1)."""
        n = len(rectangles)
        if n == 0:
            return torch.empty(0, 0) # Should be handled by caller knowing input_dim or error

        # Vectorized implementation
        # Extract mins and maxs directly from Rect objects
        # This list comprehension is the only linear python part, but faster than full loop
        mins = np.array([r.min for r in rectangles])
        maxs = np.array([r.max for r in rectangles])
        
        # Calculate scene bounds
        # (d,)
        scene_min = np.min(mins, axis=0)
        scene_max = np.max(maxs, axis=0)
        scene_extent = np.maximum(1e-9, scene_max - scene_min)
        
        # Broadcasting normalization (N, d)
        n_mins = (mins - scene_min) / scene_extent
        n_maxs = (maxs - scene_min) / scene_extent
        
        sizes = n_maxs - n_mins
        centers = (n_mins + n_maxs) / 2
        
        # Volumes (N, 1)
        volumes = np.prod(sizes, axis=1, keepdims=True)
        
        # Concatenate features (N, 4d+1)
        # Order: min, max, size, volume, center
        features = np.concatenate([n_mins, n_maxs, sizes, volumes, centers], axis=1)
            
        return torch.tensor(features, dtype=torch.float32)




