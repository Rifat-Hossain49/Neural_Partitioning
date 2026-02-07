"""

Dynamic programming to compute the optimal R-tree for a small set S (|S| ≤ k_0).
Minimizes expected query visits over Q_ex(S).
"""

import numpy as np
from typing import List, Dict, Tuple, Set, FrozenSet
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtreelib.models import Rect
from neural_partitioning.query_set import compute_query_probability, preprocess_query_set


def compute_optimal_tree(
    rectangles: List[Rect],
    max_entries: int,
    return_structure: bool = False,
    objective: str = 'range',  # 'range', 'knn', 'mix', 'volume'
    mix_alpha: float = 0.5,     # Weight for range query in mix (1.0 = range, 0.0 = knn)
    k_knn: int = 1,
    mc_samples: int = 100
) -> Tuple[float, Dict]:
    """
    Compute the optimal R-tree structure using Dynamic Programming.
    
    Minimizes Expected Query Visits (EQV).
    
    Args:
        rectangles: List of rectangles (|S| ≤ k_0, typically k_0=8-12)
        max_entries: Maximum fanout M
        return_structure: If True, return tree structure for reconstruction
        objective: Cost function ('range', 'knn', 'mix')
        mix_alpha: Weight for range query cost in 'mix' objective
        k_knn: k parameter for k-NN queries
        mc_samples: Number of Monte Carlo samples for k-NN probability estimation
        
    Returns:
        optimal_cost: Minimum expected visits
        structure: Tree structure (if return_structure=True)
    """
    n = len(rectangles)
    
    if n == 0:
        return 0.0, {}
    
    if n <= max_entries:
        return 0.0, {'type': 'leaf', 'indices': list(range(n)), 'mbr': _compute_mbr(rectangles)}
    
    # Preprocess query set for O(1) probability computation (Range Query)
    preprocessed_data = None
    if objective in ['range', 'mix'] and n > 0:
        from neural_partitioning.query_set import preprocess_query_set_ndim
        preprocessed_data = preprocess_query_set_ndim(rectangles)

    # Pre-computation for k-NN MC sampling
    # We sample points once for the whole problem to ensure consistency (common random numbers)
    # Sampling domain: Global MBR of all rectangles
    global_mbr = _compute_mbr(rectangles)
    
    query_points = []
    if objective in ['knn', 'mix']:
        np.random.seed(42) # Fixed seed for deterministic DP
        for _ in range(mc_samples):
            # Sample point in N-dim global MBR
            q_coords = []
            for d in range(global_mbr.dims):
                q_coords.append(np.random.uniform(global_mbr.min[d], global_mbr.max[d]))
            q_coords = np.array(q_coords, dtype=np.float32)
            
            # Distance to all rects in S
            dists = []
            for r in rectangles:
                dists.append(r.min_dist_sq(q_coords))
            dists.sort()
            k_dist_sq = dists[min(k_knn - 1, len(dists) - 1)]
            
            query_points.append({'coords': q_coords, 'k_dist_sq': k_dist_sq})
            
    # Memoization: F[frozenset of indices] = (cost, best_partition)
    memo = {}
    
    # Pre-extract raw coordinate arrays for fast MBR computation
    raw_mins = np.array([r.min for r in rectangles])
    raw_maxs = np.array([r.max for r in rectangles])
    _mbr_cache = {}

    def mbr_of_indices(indices: List[int]) -> Rect:
        """Compute MBR of rectangles at given indices with caching."""
        if not indices:
            return None
        
        # Use frozen set as cache key
        key = frozenset(indices)
        if key in _mbr_cache:
            return _mbr_cache[key]
        
        # Fast vectorized MBR
        idx_arr = list(indices)
        m = np.min(raw_mins[idx_arr], axis=0)
        M = np.max(raw_maxs[idx_arr], axis=0)
        res = Rect(m, M)
        _mbr_cache[key] = res
        return res
    
    def compute_knn_prob(bbox: Rect) -> float:
        """Compute P(visit | k-NN query) via MC."""
        count = 0
        for q in query_points:
            # Check intersection with query ball
            # i.e., dist(q, bbox) <= kth_dist(q)
            # We work with squared distances
            d_sq = bbox.min_dist_sq(q['coords'])
            if d_sq <= q['k_dist_sq'] + 1e-9:
                count += 1
        return count / len(query_points)

    def F(indices_set: FrozenSet[int]) -> Tuple[float, List[List[int]]]:
        if indices_set in memo:
            return memo[indices_set]
        
        indices = list(indices_set)
        
        # Base case: |A| <= M (Eq. 8)
        if len(indices) <= max_entries:
            result = (0.0, [indices])
            memo[indices_set] = result
            return result
        
        best_cost = float('inf')
        best_partition = None
        
        for partition in _generate_partitions(indices, max_entries):
            cost = 0.0
            for group in partition:
                # Cost of node = P(visit) + Cost(subtree)
                group_mbr = mbr_of_indices(group)
                
                # Calculate Weight W(B)
                w_node = 0.0
                
                if objective == 'range' or (objective == 'mix' and mix_alpha > 0):
                    from neural_partitioning.query_set import compute_query_probability_ndim
                    p_range = compute_query_probability_ndim(
                        group_mbr, rectangles, preprocessed_data
                    )

                    if objective == 'range':
                        w_node = p_range
                    else:
                        w_node += mix_alpha * p_range
                        
                if objective == 'knn' or (objective == 'mix' and mix_alpha < 1):
                    p_knn = compute_knn_prob(group_mbr)
                    if objective == 'knn':
                        w_node = p_knn
                    else:
                        w_node += (1 - mix_alpha) * p_knn

                if objective == 'volume':
                    # Simple volume minimization (standard R-tree proxy)
                    w_node = group_mbr.area()


                # F(group)
                group_set = frozenset(group)
                group_cost, _ = F(group_set)
                
                cost += w_node + group_cost
            
            if cost < best_cost:
                best_cost = cost
                best_partition = partition
        
        result = (best_cost, best_partition)
        memo[indices_set] = result
        return result
    
    # Compute optimal tree for all indices
    all_indices = frozenset(range(n))
    optimal_cost, optimal_partition = F(all_indices)
    
    # Build structure if requested
    structure = {}
    if return_structure:
        structure = _build_tree_structure(
            optimal_partition, rectangles, memo, max_entries
        )
    
    return optimal_cost, structure


_PARTITION_CACHE = {}

def _generate_partitions(indices: List[int], max_groups: int) -> List[List[List[int]]]:
    """
    Generate all valid partitions of indices into 2 to max_groups groups.
    Caches results based on (n, max_groups) for performance.
    """
    n = len(indices)
    cache_key = (n, max_groups)
    if cache_key in _PARTITION_CACHE:
        # Map template partitions back to the specific indices provided
        template_partitions = _PARTITION_CACHE[cache_key]
        results = []
        for partition in template_partitions:
            results.append([[indices[i] for i in group] for group in partition])
        return results

    template_indices = list(range(n))
    partitions = []
    
    def backtrack(remaining: List[int], current_partition: List[List[int]]):
        """Backtracking helper."""
        if not remaining:
            # Valid partition if we have 2+ groups
            if 2 <= len(current_partition) <= max_groups:
                partitions.append([g[:] for g in current_partition])
            return
        
        if len(current_partition) >= max_groups:
            # Can't add more groups, must add to existing groups
            for i, group in enumerate(current_partition):
                current_partition[i].append(remaining[0])
                backtrack(remaining[1:], current_partition)
                current_partition[i].pop()
        else:
            # Can add to existing groups or create new group
            
            # Option 1: Add to existing groups
            for i, group in enumerate(current_partition):
                current_partition[i].append(remaining[0])
                backtrack(remaining[1:], current_partition)
                current_partition[i].pop()
            
            # Option 2: Create new group
            current_partition.append([remaining[0]])
            backtrack(remaining[1:], current_partition)
            current_partition.pop()
    
    # Start backtracking
    backtrack(template_indices, [])
    
    # Cache and return mapped
    _PARTITION_CACHE[cache_key] = partitions
    
    results = []
    for partition in partitions:
        results.append([[indices[i] for i in group] for group in partition])
    return results


def _build_tree_structure(
    partition: List[List[int]],
    rectangles: List[Rect],
    memo: Dict,
    max_entries: int
) -> Dict:
    """
    Build tree structure from optimal partition.
    
    Returns a nested dictionary representing the tree.
    """
    if partition is None:
        # Fallback for failed partition (should not happen for n >= 2 in theory)
        # but prevents crash. Create a single leaf-like internal node or similar.
        if len(rectangles) > 1:
             # Just split roughly
             m = len(rectangles) // 2
             return {
                 'type': 'internal',
                 'children': [
                     {'type': 'leaf', 'indices': list(range(m)), 'mbr': _compute_mbr([rectangles[i] for i in range(m)])},
                     {'type': 'leaf', 'indices': list(range(m, len(rectangles))), 'mbr': _compute_mbr([rectangles[i] for i in range(m, len(rectangles))])}
                 ],
                 'mbr': _compute_mbr(rectangles)
             }
        return {'type': 'leaf', 'indices': [0], 'mbr': rectangles[0]}

    if len(partition) == 1:
        # Leaf node (contains 1 to max_entries items)
        indices = partition[0]
        return {
            'type': 'leaf',
            'indices': indices,
            'mbr': _compute_mbr([rectangles[i] for i in indices])
        }
    
    # Internal node
    children = []
    for group in partition:
        group_set = frozenset(group)
        if group_set in memo:
            _, sub_partition = memo[group_set]
            child_structure = _build_tree_structure(
                sub_partition, rectangles, memo, max_entries
            )
        else:
            # Leaf group
            child_structure = {
                'type': 'leaf',
                'indices': group,
                'mbr': _compute_mbr([rectangles[i] for i in group])
            }
        children.append(child_structure)
    
    return {
        'type': 'internal',
        'children': children,
        'mbr': _compute_mbr([child['mbr'] for child in children])
    }


def _compute_mbr(rectangles: List[Rect]) -> Rect:
    """Compute MBR of a list of rectangles."""
    if not rectangles:
        return None
    
    if len(rectangles) == 1:
        return rectangles[0]
    
    # Filter rectangles with proper attributes
    valid_rects = [r for r in rectangles if hasattr(r, 'min') and hasattr(r, 'max')]
    if not valid_rects:
        return None
        
    mins = np.array([r.min for r in valid_rects])
    maxs = np.array([r.max for r in valid_rects])
    
    return Rect(np.min(mins, axis=0), np.max(maxs, axis=0))


def build_ultrametric_matrix(
    rectangles: List[Rect],
    max_entries: int
) -> np.ndarray:
    """
    Build teacher ultrametric H^teach from optimal tree.
    
    H^teach[i, j] = LCA depth of leaves i and j in optimal tree.
    
    Args:
        rectangles: List of rectangles
        max_entries: Maximum fanout M
        
    Returns:
        H: (n, n) ultrametric matrix
    """
    n = len(rectangles)
    
    if n <= 1:
        return np.zeros((n, n))
    
    # Compute optimal tree structure
    _, structure = compute_optimal_tree(rectangles, max_entries, return_structure=True)
    
    # Build ultrametric from tree structure
    H = np.zeros((n, n))
    
    def assign_depths(node: Dict, depth: int, leaf_indices: List[int]):
        """
        Traverse tree and assign LCA depths.
        
        For all pairs of leaves in different subtrees of this node,
        their LCA is this node, so H[i,j] = depth.
        """
        if node['type'] == 'leaf':
            # Collect leaf indices
            leaf_indices.extend(node['indices'])
            return
        
        # Internal node - process children
        children_leaves = []
        for child in node['children']:
            child_leaves = []
            assign_depths(child, depth + 1, child_leaves)
            children_leaves.append(child_leaves)
        
        # Set LCA depth for pairs across different children
        for i in range(len(children_leaves)):
            for j in range(i + 1, len(children_leaves)):
                leaves_i = children_leaves[i]
                leaves_j = children_leaves[j]
                
                for leaf_i in leaves_i:
                    for leaf_j in leaves_j:
                        H[leaf_i, leaf_j] = depth
                        H[leaf_j, leaf_i] = depth
        
        # Add all leaves to parent's list
        for child_leaves in children_leaves:
            leaf_indices.extend(child_leaves)
    
    # Build ultrametric starting from root
    root_leaves = []
    assign_depths(structure, 0, root_leaves)
    
    return H


def get_optimal_binary_partition_mask(rectangles: List[Rect]) -> List[int]:
    """
    Compute optimal binary partition for a set of rectangles.
    
    Returns:
        mask: List of 0/1 integers where 0 indicates group 1 and 1 indicates group 2.
              Returns None if n < 2.
    """
    n = len(rectangles)
    if n < 2:
        return None
        
    # Force binary split by setting max_entries=2
    # This forces the DP to find the best partition into exactly 2 groups
    _, structure = compute_optimal_tree(rectangles, max_entries=2, return_structure=True)
    
    # The structure root should be an internal node with 2 children
    if structure['type'] != 'internal':
        # Should not happen for n >= 2 with max_entries=2
        return [0] * n
        
    children = structure['children']
    if len(children) != 2:
        # Fallback if DP found a single group (unlikely with forced split logic)
        # or if something else weird happened.
        # For n >= 2, optimal tree with M=2 should have 2 children at root.
        # If it has > 2, that violates M=2.
        # If it has 1, it means no split was better? (But we enforce 2..M groups in partition)
        return [0] * n
        
    # Extract indices for group 1 (label 0) and group 2 (label 1)
    # We need to map back to original indices.
    # The structure is recursive, but we only care about the top-level split.
    
    mask = [0] * n
    
    # Helper to collect all indices in a subtree
    def collect_indices(node):
        if node['type'] == 'leaf':
            return node['indices']
        indices = []
        for child in node['children']:
            indices.extend(collect_indices(child))
        return indices
        
    group1_indices = collect_indices(children[0])
    group2_indices = collect_indices(children[1])
    
    for idx in group2_indices:
        mask[idx] = 1
        
    return mask


def get_optimal_partition_labels(
    rectangles: List[Rect], 
    max_groups: int = 4,
    objective: str = 'range',
    mix_alpha: float = 0.5,
    k_knn: int = 1,
    mc_samples: int = 100
) -> List[int]:
    """
    Compute optimal partition labels for a set of rectangles into k groups (2 <= k <= max_groups).
    
    Args:
        rectangles: List of rectangles
        max_groups: Maximum number of groups (fanout)
        objective: 'range', 'knn', or 'mix'
        mix_alpha: weight for range in mix
        k_knn: k for k-NN
        mc_samples: MC samples for k-NN
        
    Returns:
        labels: List of integer labels [0, k-1] for each rectangle.
                Returns None if n < 2.
    """
    n = len(rectangles)
    if n < 2:
        return None
        
    # Run DP with specified max_groups
    _, structure = compute_optimal_tree(
        rectangles, 
        max_entries=max_groups, 
        return_structure=True,
        objective=objective,
        mix_alpha=mix_alpha,
        k_knn=k_knn,
        mc_samples=mc_samples
    )
    
    if structure['type'] != 'internal':
        # Should not happen for n >= 2
        return [0] * n
        
    children = structure['children']
    # Number of groups chosen by DP
    k = len(children)
    
    labels = [0] * n
    
    # Helper to collect all indices in a subtree
    def collect_indices(node):
        if node['type'] == 'leaf':
            return node['indices']
        indices = []
        for child in node['children']:
            indices.extend(collect_indices(child))
        return indices
        
    # Assign labels based on which child group the rectangle belongs to
    for label, child in enumerate(children):
        group_indices = collect_indices(child)
        for idx in group_indices:
            labels[idx] = label
            
    return labels


