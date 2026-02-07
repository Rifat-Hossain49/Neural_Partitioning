
import argparse
import time
import json
import os
import sys
import numpy as np
import torch
import random
import pandas as pd
import subprocess
import matplotlib.pyplot as plt

# Add parent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rtreelib.models import Rect, Point
from rtreelib.strategies import RTreeGuttman, RStarTree
from neural_partitioning.model import PartitioningNetwork
from neural_partitioning.inference import PartitioningRTree
from neural_partitioning.utils import load_rectangles_from_csv

# ---------------- HELPERS ----------------
class NodeAccessTracker:
    def __init__(self):
        self.access_count = 0
    def reset(self):
        self.access_count = 0
    def count(self):
        self.access_count += 1

tracker = NodeAccessTracker()

def wrap_tree_for_counting(tree):
    # Monkey-patch to count accesses
    from rtreelib.rtree import RTreeNode
    original_get_mbr = RTreeNode.get_bounding_rect
    
    def tracked_get_mbr(self):
        tracker.count()
        return original_get_mbr(self)
    
    RTreeNode.get_bounding_rect = tracked_get_mbr

def generate_random_queries(rectangles, num_queries=100):
    # Generates random range queries (1% - 20% size factor)
    mins = np.array([r.min for r in rectangles])
    maxs = np.array([r.max for r in rectangles])
    scene_min = np.min(mins, axis=0)
    scene_max = np.max(maxs, axis=0)
    extent = scene_max - scene_min
    ndim = len(scene_min)
    
    queries = []
    for _ in range(num_queries):
        size_factor = np.random.uniform(0.01, 0.20)
        center_n = np.random.rand(ndim)
        center = scene_min + center_n * extent
        q_size = extent * size_factor
        q_min = center - q_size / 2
        q_max = center + q_size / 2
        queries.append(Rect(q_min, q_max))
    return queries

def gen_point_queries(rectangles, num_queries=100):
    mins = np.array([r.min for r in rectangles])
    maxs = np.array([r.max for r in rectangles])
    scene_min = np.min(mins, axis=0)
    scene_max = np.max(maxs, axis=0)
    ndim = len(scene_min)
    queries = []
    for _ in range(num_queries):
        coords = [random.uniform(scene_min[d], scene_max[d]) for d in range(ndim)]
        queries.append(Rect(coords, coords)) # Point is zero-area rect
    return queries

PYTHON = sys.executable
DATASETS_REPRO = {
    'Twitter': 'twitter_100k.csv',
    'Crimes': 'crimes_100k.csv'
}
METHODS_REPRO = [
    ('Partitioning(Range)', 'range_model'), 
    ('Partitioning(kNN)', 'knn_model'), 
    ('R* Tree', 'rstar'), 
    ('Guttman', 'guttman')
]
MAX_ENTRIES_DEFAULT = "256,512,784,1024"
RESULTS_DIR = "results_repro"
PLOTS_DIR = "plots_avg_query_time"
PLOTS_BUILD_TIME_DIR = "plots_build_time"
MODELS_DIR = "models_repro"
OVERLAP_RATIOS_FILE = "overlap_ratios.json"

# ---------------- JSON SERIALIZATION HELPER ----------------
def convert_to_serializable(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ---------------- PREPROCESSING ----------------
def sample_and_format(input_path, output_path, n=100000, seed=42):
    print(f"[Preprocess] Processing {input_path} -> {output_path}")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    try:
        if 'twitter' in input_path.lower():
            df = pd.read_csv(input_path, on_bad_lines='skip')
            df.columns = [c.strip().lower() for c in df.columns]
            if 'latitude' in df.columns and 'longitude' in df.columns:
                lat_col, lon_col = 'latitude', 'longitude'
            else:
                cols = df.select_dtypes(include=[np.number]).columns
                if len(cols) >= 2:
                    lon_col, lat_col = cols[0], cols[1]
                else:
                    raise ValueError("Could not identify coordinate columns")
            
        else: # Crimes
            df = pd.read_csv(input_path, on_bad_lines='skip')
            df_cols_lower = {c.lower(): c for c in df.columns}
            if 'latitude' in df_cols_lower and 'longitude' in df_cols_lower:
                lat_col = df_cols_lower['latitude']
                lon_col = df_cols_lower['longitude']
            else:
                 raise ValueError("Could not identify 'Latitude'/'Longitude' columns")

        df = df.dropna(subset=[lat_col, lon_col])
        if len(df) > n:
            df = df.sample(n=n, random_state=seed)

        lats = df[lat_col].values
        lons = df[lon_col].values
        out_df = pd.DataFrame({'min_0': lons, 'min_1': lats, 'max_0': lons, 'max_1': lats})
        out_df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(out_df)} records.")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

# ---------------- BENCHMARKING ----------------
def compute_coverage(tree, node=None):
    """
    Compute the total coverage (sum of all child MBR volumes) across all internal nodes.
    This is the denominator in the overlap ratio formula.
    """
    if node is None:
        node = tree.root
    
    if node.is_leaf:
        return 0.0
    
    coverage = 0.0
    children = node.entries
    
    # Sum volumes of all children at this node
    for child_entry in children:
        coverage += child_entry.rect.area()
    
    # Recursively compute coverage for all child nodes
    for child_entry in children:
        if not child_entry.is_leaf:
            coverage += compute_coverage(tree, child_entry.child)
    
    return coverage

def calculate_overlap_ratio(tree):
    """
    Calculate overlap ratio according to the paper's formula:
    OverlapRatio = (Σ_u Σ_{i<j} Vol(R_{u,i} ∩ R_{u,j})) / (Σ_u Σ_i Vol(R_{u,i}))
    
    Where:
    - Numerator: Sum of all pairwise overlaps across all internal nodes
    - Denominator: Sum of all child MBR volumes across all internal nodes
    """
    total_overlap = tree.compute_overlap()  # Numerator
    total_coverage = compute_coverage(tree)  # Denominator
    
    if total_coverage == 0:
        return 0.0
    
    overlap_ratio = total_overlap / total_coverage
    
    return float(overlap_ratio)  # Ensure it's a Python float, not numpy

def load_model(model_path, device='cpu'):
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
    return model

def run_single_benchmark(csv_file, model_key, query_type, max_entries_list, k_knn, query_count, device='cpu'):
    rectangles = load_rectangles_from_csv(csv_file)
    
    # Generate Queries
    random.seed(42)
    np.random.seed(42)
    queries = []
    
    if query_type == 'range':
        # Random range queries
        queries = generate_random_queries(rectangles, query_count)
    elif query_type == 'point':
        queries = gen_point_queries(rectangles, query_count)
    elif query_type == 'knn':
        mins = np.array([r.min for r in rectangles])
        maxs = np.array([r.max for r in rectangles])
        scene_min = np.min(mins, axis=0)
        scene_max = np.max(maxs, axis=0)
        for _ in range(query_count):
            coords = [random.uniform(scene_min[d], scene_max[d]) for d in range(len(scene_min))]
            queries.append(Point(*coords))

    results = {}
    for max_entries in max_entries_list:
        print(f"    Running M={max_entries}...")
        start_build = time.time()
        
        # Build Tree
        if model_key == 'guttman':
            tree = RTreeGuttman(max_entries=max_entries)
            for i, r in enumerate(rectangles): tree.insert(i, r)
        elif model_key == 'rstar':
            tree = RStarTree(max_entries=max_entries)
            for i, r in enumerate(rectangles): tree.insert(i, r)
        else:
            # Model path
            model = load_model(model_key, device)
            tree = PartitioningRTree(model, max_entries=max_entries, device=device)
            tree.insert_all(rectangles)
            
        build_time = time.time() - start_build
        
        # Calculate overlap ratio
        overlap_ratio = calculate_overlap_ratio(tree)
        
        # Query
        wrap_tree_for_counting(tree)
        tracker.reset()
        start_query = time.time()
        
        if query_type == 'knn':
            for q in queries: tree.nearest_neighbor(q, k=k_knn)
        else:
            for q in queries: list(tree.query(q))
                
        avg_time = ((time.time() - start_query) / len(queries)) * 1000
        avg_access = tracker.access_count / len(queries)
        results[max_entries] = {
            'avg_time_ms': avg_time, 
            'avg_access': avg_access, 
            'build_time_s': build_time,
            'overlap_ratio': overlap_ratio
        }
        print(f"    -> M={max_entries}: {avg_time:.2f}ms (Build: {build_time:.2f}s, Overlap: {overlap_ratio:.4f})")
        
    return results

# ---------------- PLOTTING ----------------
def generate_plots_repro():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for dataset_name in DATASETS_REPRO.keys():
        for query_type in ['range', 'point', 'knn']:
            plt.figure(figsize=(8, 6))
            for method_name, method_key in METHODS_REPRO:
                res_file = os.path.join(RESULTS_DIR, f"{dataset_name}_{query_type}_{method_key}.json")
                if not os.path.exists(res_file): continue
                
                with open(res_file) as f:
                    data = json.load(f)
                x_vals = sorted([int(k) for k in data.keys()])
                y_vals = [data[str(x)]['avg_time_ms'] for x in x_vals]
                
                marker = 'o'
                if 'Range' in method_name: marker = 's'
                elif 'KNN' in method_name: marker = '^'
                plt.plot(x_vals, y_vals, marker=marker, label=method_name, linewidth=2)
            
            plt.title(f"{dataset_name} - {query_type.capitalize()} Query Latency")
            plt.xlabel("Max Entries (M)")
            plt.ylabel("Avg Time (ms)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(PLOTS_DIR, f"{dataset_name}_{query_type}_latency.png"))
            plt.close()
            plt.savefig(os.path.join(PLOTS_DIR, f"{dataset_name}_{query_type}_latency.png"))
            plt.close()
    print(f"Plots saved to {PLOTS_DIR}")
    
    # Generate Build Time Plots for Repro
    for dataset_name in DATASETS_REPRO.keys():
        # Gather results for all methods
        # Use 'range' results as proxy for build time (built once per M)
        combined_results = {}
        for method_name, method_key in METHODS_REPRO:
            res_file = os.path.join(RESULTS_DIR, f"{dataset_name}_range_{method_key}.json")
            if os.path.exists(res_file):
                with open(res_file) as f:
                    combined_results[method_name] = json.load(f)
        
        if combined_results:
            plot_build_time(combined_results, dataset_name)

def plot_comparison(all_results, dataset_name, query_type):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    # Define markers/colors for consistency if possible
    markers = ['o', 's', '^', 'D', 'x']
    
    for i, (model_label, res) in enumerate(all_results.items()):
        x_vals = sorted([k for k in res.keys()])
        y_vals = [res[k]['avg_time_ms'] for k in x_vals]
        plt.plot(x_vals, y_vals, marker=markers[i % len(markers)], label=model_label, linewidth=2)
    
    plt.title(f"{dataset_name} - {query_type} Query Performance")
    plt.xlabel("Max Entries")
    plt.ylabel("Avg Time (ms)")
    plt.legend()
    plt.grid(True)
    
    # Sanitize filename
    safe_name = dataset_name.replace('.csv', '').replace(' ', '_')
    path = os.path.join(PLOTS_DIR, f"custom_compare_{safe_name}_{query_type}.png")
    plt.savefig(path)
    print(f"Saved comparison plot to {path}")

def plot_build_time(all_results, dataset_name):
    # Plot Build Time vs Max Entries (Log Scale)
    os.makedirs(PLOTS_BUILD_TIME_DIR, exist_ok=True)
    
    # We only need to do this once per Dataset (build time is same regardless of query type, 
    # but we might call this from plotting function which is per-query.
    # We should grab entries from the first model result
    
    first_key = list(all_results.keys())[0]
    entries = sorted([k for k in all_results[first_key].keys()])
    
    n_groups = len(entries)
    n_models = len(all_results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar width
    total_width = 0.8
    bar_width = total_width / n_models
    
    # Colors suitable for paper
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
    
    for i, (model_label, res) in enumerate(all_results.items()):
        # Extract build times matching sorted entries
        # Handle missing entries if any
        build_times = []
        for e in entries:
            if e in res:
                build_times.append(res[e].get('build_time_s', 0))
            else:
                build_times.append(0)
        
        # Position bars
        # x locations: index - total_width/2 + i*bar + bar/2
        indices = np.arange(n_groups)
        x_pos = indices - (total_width / 2) + (i * bar_width) + (bar_width / 2)
        
        ax.bar(x_pos, build_times, width=bar_width, label=model_label, color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)

    ax.set_yscale('log')
    ax.set_xlabel('Max Entries (M)', fontsize=12)
    ax.set_ylabel('Build Time (s) - Log Scale', fontsize=12)
    ax.set_title(f'Build Time vs Max Entries - {dataset_name}', fontsize=14)
    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels([str(e) for e in entries])
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    safe_name = dataset_name.replace('.csv', '').replace(' ', '_')
    path = os.path.join(PLOTS_BUILD_TIME_DIR, f"{safe_name}_build_time.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved build time plot to {path}")

def save_overlap_ratios():
    """
    Extract overlap ratios from all result files and save to a JSON file.
    """
    overlap_data = {}
    
    # Iterate through all JSON result files
    if not os.path.exists(RESULTS_DIR):
        print(f"Warning: {RESULTS_DIR} does not exist. No overlap ratios to extract.")
        return
    
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith('.json') or filename == OVERLAP_RATIOS_FILE:
            continue
        
        filepath = os.path.join(RESULTS_DIR, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Format: custom_{dataset}_{query_type}.json or {dataset}_{query_type}_{method}.json
            name_no_ext = filename.replace('.json', '')
            
            if name_no_ext.startswith('custom_'):
                # Custom format: custom_{dataset}_{query_type}
                # Use rsplit to split off the query_type from the right
                remaining, query_type = name_no_ext.rsplit('_', 1)
                dataset = remaining.replace('custom_', '', 1)
                
                # For custom runs, data structure is {model_label: {max_entries: {metrics}}}
                for model_label, model_results in data.items():
                    if dataset not in overlap_data:
                        overlap_data[dataset] = {}
                    
                    if query_type not in overlap_data[dataset]:
                        overlap_data[dataset][query_type] = {}
                    
                    if model_label not in overlap_data[dataset][query_type]:
                        overlap_data[dataset][query_type][model_label] = {}
                    
                    for max_entries, metrics in model_results.items():
                        if 'overlap_ratio' in metrics:
                            overlap_data[dataset][query_type][model_label][max_entries] = metrics['overlap_ratio']
            else:
                # Repro format: {dataset}_{query_type}_{method}
                # Known methods: 'range_model', 'knn_model', 'rstar', 'guttman'
                known_methods = ['range_model', 'knn_model', 'rstar', 'guttman']
                matched_method = None
                for m in known_methods:
                    if name_no_ext.endswith(f"_{m}"):
                        matched_method = m
                        break
                
                if matched_method:
                    remaining = name_no_ext[:-(len(matched_method)+1)]
                    if '_' in remaining:
                        dataset, query_type = remaining.rsplit('_', 1)
                    else:
                        dataset, query_type = remaining, "unknown"
                    
                    if dataset not in overlap_data:
                        overlap_data[dataset] = {}
                    
                    if query_type not in overlap_data[dataset]:
                        overlap_data[dataset][query_type] = {}
                        
                    if matched_method not in overlap_data[dataset][query_type]:
                        overlap_data[dataset][query_type][matched_method] = {}
                    
                    for max_entries, metrics in data.items():
                        if isinstance(metrics, dict) and 'overlap_ratio' in metrics:
                            overlap_data[dataset][query_type][matched_method][max_entries] = metrics['overlap_ratio']
        
        except Exception as e:
            print(f"Warning: Failed to process {filename}: {e}")
            continue
    
    # Save overlap ratios to file
    output_path = os.path.join(RESULTS_DIR, OVERLAP_RATIOS_FILE)
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(overlap_data), f, indent=2)
    
    print(f"\nOverlap ratios saved to {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("OVERLAP RATIO SUMMARY")
    print("="*80)
    for dataset, queries in sorted(overlap_data.items()):
        print(f"\nDataset: {dataset}")
        for q_type, methods in sorted(queries.items()):
            print(f"  Query Type: {q_type}")
            for method, entries in sorted(methods.items()):
                print(f"    {method}:")
                for max_e, ratio in sorted(entries.items(), key=lambda x: int(x[0])):
                    print(f"      M={max_e}: {ratio:.6f}")

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser(description="Run NPN Experiments")
    # Custom run arguments
    parser.add_argument('--model', help="Path to .pth model file, or 'guttman'/'rstar'")
    parser.add_argument('--dataset', help="Path to CSV dataset")
    parser.add_argument('--query_type', choices=['range', 'point', 'knn'], help="Type of query to benchmark")
    
    # Common arguments
    parser.add_argument('--max_entries', default=MAX_ENTRIES_DEFAULT, help="Comma separated max entries list")
    parser.add_argument('--query_count', type=int, default=100000, help="Number of queries to run")
    parser.add_argument('--device', default='cpu', help="Device (cpu/cuda)")
    parser.add_argument('--k_knn', type=int, default=10, help="k for KNN queries")
    parser.add_argument('--compare_baselines', action='store_true', help="Compare with R* and Guttman trees")
    
    # Reproduction paths (optional override)
    parser.add_argument('--twitter_raw', default='twitter.csv')
    parser.add_argument('--crimes_raw', default='Crimes_-_2001_to_Present.csv')
    
    args = parser.parse_args()
    entries_list = [int(x) for x in args.max_entries.split(',')]

    # Mode Selection
    if args.model and args.dataset and args.query_type:
        # --- CUSTOM RUN MODE ---
        print(f"Running Custom Benchmark:")
        print(f"  Model: {args.model}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Query: {args.query_type}")
        print(f"  Compare Baselines: {args.compare_baselines}")
        
        all_results = {}
        
        # 1. Run User Model
        label = os.path.basename(args.model)
        print(f"\n>> Benchmarking {label}...")
        res = run_single_benchmark(
            args.dataset, args.model, args.query_type, entries_list, 
            args.k_knn, args.query_count, args.device
        )
        all_results[label] = res
        
        # 2. Run Baselines if requested
        if args.compare_baselines:
            for base in ['rstar', 'guttman']:
                print(f"\n>> Benchmarking Baseline: {base}...")
                base_res = run_single_benchmark(
                    args.dataset, base, args.query_type, entries_list,
                    args.k_knn, args.query_count, args.device
                )
                all_results[base] = base_res

        # Save result
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_name = f"custom_{os.path.basename(args.dataset)}_{args.query_type}.json"
        with open(os.path.join(RESULTS_DIR, out_name), 'w') as f:
            json.dump(convert_to_serializable(all_results), f, indent=2)
            
        # Plot
        plot_comparison(all_results, os.path.basename(args.dataset), args.query_type)
        if args.compare_baselines:
             plot_build_time(all_results, os.path.basename(args.dataset))
        
        # Save overlap ratios
        save_overlap_ratios()
        
    else:
        # --- REPRODUCTION MODE ---
        print("No specific model/dataset provided. Running FULL REPRODUCTION pipeline.")
        
        # Step 1: Preprocess
        print("\n--- Step 1: Preprocessing ---")
        sample_and_format(args.twitter_raw, DATASETS_REPRO['Twitter'])
        sample_and_format(args.crimes_raw, DATASETS_REPRO['Crimes'])
        
        # Step 2: Training (Skipped - User must train separately)
        # Models are expected to be in MODELS_DIR
        print("\n--- Step 2: Checking Models ---")
        if not os.path.exists(MODELS_DIR):
            print(f"Warning: {MODELS_DIR} does not exist. Please train models using train_custom.py.")
        
        # Step 3: Benchmarking
        print("\n--- Step 3: Benchmarking ---")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        for dataset_name, csv in DATASETS_REPRO.items():
            if not os.path.exists(csv): continue
                
            for query_type in ['range', 'point', 'knn']:
                for method_name, method_key in METHODS_REPRO:
                    res_file = os.path.join(RESULTS_DIR, f"{dataset_name}_{query_type}_{method_key}.json")
                    if os.path.exists(res_file): continue
                    
                    # Resolve model
                    if method_key == 'range_model':
                        model_path = os.path.join(MODELS_DIR, f"{dataset_name.lower()}_range.pth")
                    elif method_key == 'knn_model':
                        model_path = os.path.join(MODELS_DIR, f"{dataset_name.lower()}_knn.pth")
                    else:
                        model_path = method_key
                        
                    if model_path not in ['rstar', 'guttman'] and not os.path.exists(model_path):
                        continue
                        
                    print(f"Benchmarking: {dataset_name} | {query_type} | {method_name}")
                    res = run_single_benchmark(
                        csv, model_path, query_type, entries_list, 10, args.query_count, args.device
                    )
                    with open(res_file, 'w') as f: 
                        json.dump(convert_to_serializable(res), f, indent=2)
                    
        # Step 4: Plotting
        print("\n--- Step 4: Plotting ---")
        generate_plots_repro()
        
        # Step 5: Save overlap ratios
        print("\n--- Step 5: Saving Overlap Ratios ---")
        save_overlap_ratios()

if __name__ == "__main__":
    main()
