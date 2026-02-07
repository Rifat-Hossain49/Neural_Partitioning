import pandas as pd
import numpy as np
from typing import List
import time
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtreelib.models import Rect
from rtreelib.strategies import RTreeGuttman, RStarTree
from rtreelib.rtree import RTreeNode as LibRTreeNode, RTreeEntry as LibRTreeEntry

def load_rectangles_from_csv(csv_file: str) -> List[Rect]:
    """Load rectangles from CSV file. Supports 'x,y', 'min_x,min_y...', or 'min_0,min_1...,max_0...'."""
    df = pd.read_csv(csv_file)
    cols = df.columns
    rectangles = []
    
    # Check for N-dim format: min_0, min_1...
    min_cols = [c for c in cols if c.startswith('min_')]
    max_cols = [c for c in cols if c.startswith('max_')]
    
    if len(min_cols) > 0 and len(min_cols) == len(max_cols):
        try:
            min_cols.sort(key=lambda x: int(x.split('_')[1]))
            max_cols.sort(key=lambda x: int(x.split('_')[1]))
            
            mins = df[min_cols].values
            maxs = df[max_cols].values
            
            for i in range(len(df)):
                rectangles.append(Rect(mins[i], maxs[i]))
            return rectangles
        except Exception:
            pass

    if 'x' in cols and 'y' in cols:
        for _, row in df.iterrows():
            x, y = float(row['x']), float(row['y'])
            rectangles.append(Rect(x, y, x, y))
    elif 'min_x' in cols:
        for _, row in df.iterrows():
            rectangles.append(Rect(
                min_x=float(row['min_x']),
                min_y=float(row['min_y']),
                max_x=float(row['max_x']),
                max_y=float(row['max_y'])
            ))
    else:
        raise ValueError(f"Unknown CSV format in {csv_file}")
    
    return rectangles

def generate_random_queries(rectangles: List[Rect], num_queries: int = 100) -> List[Rect]:
    if not rectangles:
        return []
    
    ndim = rectangles[0].dims
    all_mins = np.array([r.min for r in rectangles])
    all_maxs = np.array([r.max for r in rectangles])
    
    scene_min = np.min(all_mins, axis=0)
    scene_max = np.max(all_maxs, axis=0)
    scene_extent = scene_max - scene_min
    
    queries = []
    for _ in range(num_queries):
        size_factor = np.random.uniform(0.01, 0.20)
        center_n = np.random.rand(ndim)
        center = scene_min + center_n * scene_extent
        q_size = scene_extent * size_factor
        q_min = center - q_size / 2
        q_max = center + q_size / 2
        queries.append(Rect(q_min, q_max))
    
    return queries

def test_query_performance(tree, queries: List[Rect]) -> dict:
    start = time.time()
    total_results = 0
    for query_rect in queries:
        results = list(tree.query(query_rect))
        total_results += len(results)
    elapsed = time.time() - start
    return {
        'total_time': elapsed,
        'avg_time': elapsed / len(queries) if queries else 0,
        'total_results': total_results
    }

def build_baseline_rtrees(rectangles: List[Rect], max_entries: int):
    results = {}
    
    # Guttman
    start = time.time()
    guttman_tree = RTreeGuttman(max_entries=max_entries)
    for i, rect in enumerate(rectangles):
        guttman_tree.insert(i, rect)
    results['guttman'] = {'tree': guttman_tree, 'time': time.time() - start}
    
    # R*
    start = time.time()
    rstar_tree = RStarTree(max_entries=max_entries)
    for i, rect in enumerate(rectangles):
        rstar_tree.insert(i, rect)
    results['rstar'] = {'tree': rstar_tree, 'time': time.time() - start}
    
    return results
