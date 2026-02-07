"""
Query Set Generation (§2.1)

Generate Q_ex(S) - the set of combinatorially distinct axis-aligned queries
induced by the sorted distinct edge coordinates of rectangles in S.
"""

import numpy as np
from typing import List, Tuple, Set
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtreelib.models import Rect


def generate_query_set(rectangles: List[Rect]) -> List[Rect]:
    """
    Generate Q_ex(S) - all combinatorially distinct queries.
    Only practical for small |S| and small d.
    """
    if not rectangles:
        return []
    
    ndim = rectangles[0].dims
    coords_per_dim = [set() for _ in range(ndim)]
    
    for rect in rectangles:
        for d in range(ndim):
            coords_per_dim[d].add(rect.min[d])
            coords_per_dim[d].add(rect.max[d])
    
    sorted_coords = [sorted(c) for c in coords_per_dim]
    
    # This grows exponentially with ndim: (O(|S|^2))^ndim
    # Only use for d=2 or very small sets.
    if ndim > 2:
        print(f"Warning: generate_query_set called for {ndim}D. Result size will be large.")

    import itertools
    dim_intervals = []
    for d in range(ndim):
        intervals = []
        sc = sorted_coords[d]
        p = len(sc)
        for a in range(p):
            for b in range(a + 1, p):
                intervals.append((sc[a], sc[b]))
        dim_intervals.append(intervals)
    
    queries = []
    for combined in itertools.product(*dim_intervals):
        # combined is ((min0, max0), (min1, max1), ...)
        q_min = [c[0] for c in combined]
        q_max = [c[1] for c in combined]
        queries.append(Rect(q_min, q_max))
    
    return queries


def compute_query_probability(
    mbr: Rect, 
    rectangles: List[Rect],
    x_sorted: List[float] = None,
    y_sorted: List[float] = None,
    x_prefix: np.ndarray = None,
    y_prefix: np.ndarray = None
) -> float:
    """Legacy 2D support."""
    if not rectangles:
        return 0.0
    if rectangles[0].dims != 2:
        return compute_query_probability_ndim(mbr, rectangles)
        
    # Extract and sort coordinates if not provided
    if x_sorted is None or y_sorted is None:
        x_coords = set()
        y_coords = set()
        for rect in rectangles:
            x_coords.add(rect.min[0])
            x_coords.add(rect.max[0])
            y_coords.add(rect.min[1])
            y_coords.add(rect.max[1])
        x_sorted = sorted(x_coords)
        y_sorted = sorted(y_coords)
    
    p = len(x_sorted)
    q = len(y_sorted)
    
    if p < 2 or q < 2:
        return 0.0
    
    # Build prefix sums if not provided (identity prefix in this implementation)
    if x_prefix is None:
        x_prefix = np.arange(1, p + 1, dtype=np.int32)
    if y_prefix is None:
        y_prefix = np.arange(1, q + 1, dtype=np.int32)
    
    K_x = _count_intersecting_pairs_dim(mbr.min[0], mbr.max[0], x_sorted, x_prefix)
    K_y = _count_intersecting_pairs_dim(mbr.min[1], mbr.max[1], y_sorted, y_prefix)
    
    total_queries = (p * (p - 1) // 2) * (q * (q - 1) // 2)
    
    if total_queries == 0:
        return 0.0
    
    return (K_x * K_y) / total_queries


def compute_query_probability_ndim(
    mbr: Rect,
    rectangles: List[Rect],
    preprocessed_data: List[dict] = None
) -> float:
    """
    Compute P_S(B) for N-dimensions using factorized probability.
    P_S(B) = prod_{d=1}^D (K_d / T_d)
    """
    if not rectangles:
        return 0.0
    
    ndim = rectangles[0].dims
    if preprocessed_data is None:
        preprocessed_data = preprocess_query_set_ndim(rectangles)
        
    prob = 1.0
    for d in range(ndim):
        data = preprocessed_data[d]
        sorted_coords = data['sorted']
        prefix = data['prefix']
        p = len(sorted_coords)
        
        if p < 2:
            return 0.0
            
        K_d = _count_intersecting_pairs_dim(mbr.min[d], mbr.max[d], sorted_coords, prefix)
        T_d = p * (p - 1) // 2
        prob *= (K_d / T_d)
        
    return prob


def _count_intersecting_pairs_dim(min_val: float, max_val: float, sorted_coords: List[float], prefix: np.ndarray) -> int:
    """
    Count intervals [a, b] from sorted_coords that intersect [min_val, max_val].
    Interval [a, b] intersects [L, R] iff a <= R and b >= L.
    """
    n = len(sorted_coords)
    L_rank = _find_rank(sorted_coords, min_val) # first index >= L
    R_rank = _find_rank(sorted_coords, max_val) # first index >= R (or n if none)
    
    # We want count of (a, b) such that 0 <= a < b <= n-1
    # AND a <= R_val (where R_val is index of largest coord <= max_val)
    # AND b >= L_val (where L_val is index of first coord >= min_val)
    
    # Indices are i in [0..n-1]
    # L_idx = L_rank
    # R_idx = if sorted_coords[R_rank] == max_val: R_rank else: R_rank - 1
    if R_rank < n and sorted_coords[R_rank] <= max_val:
        R_idx = R_rank
    else:
        R_idx = R_rank - 1
    
    if R_idx < 0 or L_rank >= n:
        return 0
        
    # Count pairs (a, b) with 0 <= a < b <= n-1 satisfying a <= R_idx and b >= L_idx
    # total pairs 0 <= a < b <= n-1 is n*(n-1)//2
    # pairs entirely to the left: b < L_idx => both a, b in [0, L_idx-1] => L_idx*(L_idx-1)//2
    # pairs entirely to the right: a > R_idx => both a, b in [R_idx+1, n-1] => (n-1-R_idx)*(n-R_idx-2)//2?
    # Correct right count: number of elements m = n - 1 - R_idx. Pairs m*(m-1)//2
    
    L_idx = L_rank
    n_left = L_idx
    pairs_left = n_left * (n_left - 1) // 2 if n_left > 1 else 0
    
    n_right = n - 1 - R_idx
    pairs_right = n_right * (n_right - 1) // 2 if n_right > 1 else 0
    
    total_dim_pairs = n * (n - 1) // 2
    intersecting = total_dim_pairs - (pairs_left + pairs_right)
    return max(0, intersecting)


def _find_rank(sorted_coords: List[float], value: float) -> int:
    """Find the first index i such that sorted_coords[i] >= value."""
    import bisect
    return bisect.bisect_left(sorted_coords, value)


def preprocess_query_set_ndim(rectangles: List[Rect]) -> List[dict]:
    """Preprocess for N-dimensional probability computation."""
    if not rectangles:
        return []
    
    # Find first valid rectangle with min attribute
    first_valid = None
    for r in rectangles:
        if hasattr(r, 'min') and hasattr(r, 'max'):
            first_valid = r
            break
    
    if first_valid is None:
        # No valid rectangles found, return empty
        return []
    
    # Use len(rect.min) instead of rect.dims for compatibility
    ndim = len(first_valid.min)
    coords_per_dim = [set() for _ in range(ndim)]
    
    for r in rectangles:
        # Skip rectangles without proper attributes
        if not hasattr(r, 'min') or not hasattr(r, 'max'):
            continue
        for d in range(ndim):
            if len(r.min) > d and len(r.max) > d:
                coords_per_dim[d].add(r.min[d])
                coords_per_dim[d].add(r.max[d])
                
    results = []
    for d in range(ndim):
        sorted_c = sorted(coords_per_dim[d])
        p = len(sorted_c)
        results.append({
            'sorted': sorted_c,
            'prefix': np.arange(1, p + 1, dtype=np.int32) # kept for compat
        })
    return results

def preprocess_query_set(rectangles: List[Rect]):
    """Compatibility wrapper for 2D."""
    if not rectangles: return None
    if rectangles[0].dims != 2: return preprocess_query_set_ndim(rectangles)
    
    data = preprocess_query_set_ndim(rectangles)
    # Return legacy 4-tuple for 2D
    return data[0]['sorted'], data[1]['sorted'], data[0]['prefix'], data[1]['prefix']
