# Neural Partitioning Network (NPN) R-Tree

## Overview
This repository implements the **Neural Partitioning Network (NPN)** for R-tree optimization as described in the research paper. The system replaces traditional geometric split heuristics (like Guttman or R*) with a learned top-down partitioning strategy.

Key components:
- **Partitioning Network**: A PointNet-based architecture that predicts optimal K-way node splits.
- **DP Supervisor**: An exact dynamic programming supervisor that generates optimal partitions on small subsets for training.
- **Classic R-Trees**: Guttman's R-Tree and R*-Tree baselines for performance comparison.

## Installation

1.  **Environment Setup**:
    Ensure you have Python 3.6+ installed.
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Codebase Structure
- `neural_partitioning/`: Core implementation.
    - `model.py`: Neural Partitioning Network (NPN) architecture.
    - `dp_supervisor.py`: Dynamic programming supervisor for training labels.
    - `query_set.py`: REC (Reduced Expected Cost) calculation logic.
    - `inference.py`: Top-down R-tree construction using the NPN.
    - `utils.py`: CSV loading and performance utilities.
- `rtreelib/`: Base R-tree library (supports Guttman and R*).
- `models_paper/`: Directory for trained model weights.
- `train_npn.py`: Main script for training the partitioning network.
- `benchmark.py`: Benchmarking script for Arizona and other datasets.
- `verify_npn.py`: Codebase integrity check.

## Usage

### 1. Training the Partitioning Network
Train the partitioning network for range or KNN queries:

```bash
# General training (defaults to k=2,3 splits)
python neural_partitioning/train_npn.py --csv_files <path.csv> --epochs 50 --output models_paper/my_model.pth

# Training specifically for KNN (e.g. k=10)
python neural_partitioning/train_npn.py --csv_files <path.csv> --epochs 50 --min_k 10 --max_k 10 --output models_paper/knn_k10.pth
```

**Arguments:**
- `--csv_files`: Space-separated list of CSV datasets.
- `--epochs`: Number of training epochs (default: 50).
- `--min_k` / `--max_k`: Range of nearest neighbors for DP supervisor labels.
- `--samples`: Number of samples per file for training (default: 1000).
- `--num_classes`: Max split groups (default: 3).

### 2. Running Benchmarks

#### Range Query Benchmark
Evaluate average node accesses and query times for varying window sizes:

```bash
# Relative window size benchmark (0.1%, 1%, 10%, 50%)
python benchmark_custom.py --csv <data.csv> --model <model.pth> --access_vs_window

# Absolute window width benchmark (fixed meters)
python benchmark_custom.py --csv <data.csv> --model <model.pth> --access_vs_width --window_widths "100,200,400"

# Time vs Max Entries (M)
python benchmark_custom.py --csv <data.csv> --model <model.pth> --time_vs_entries --max_entries "128,256,512"
```

#### KNN Benchmark
Evaluate average node accesses and query times for KNN queries:

```bash
python benchmark_knn.py --csv <data.csv> --model <model.pth> --k 10 --query_count 100
```

## Dataset Extraction
### Arizona Dataset
Extract and sample the Arizona dataset from Shapefiles:
```bash
python extract_arizona.py --sample 1464257 --output arizona_large.csv
```

## Dataset Formats
The system supports N-dimensional datasets via CSV files with `min_0, ..., max_{d-1}` columns.

## Verification
Run a quick check on a subset:
```bash
python benchmark_knn.py --csv arizona_large.csv --model models_paper/knn_k10_clustered.pth --k 10 --query_count 10
```

## Reproduction & Benchmarking (Avg Query Time vs Max Entries)

The `benchmark_avg_query_time.py` script serves two purposes: full paper reproduction and custom model benchmarking. It evaluates Average Query Time vs Max Entries (M).

### 1. Reproduce Full Paper Experiments
Run without arguments to compare all models (Range-NPN, KNN-NPN, R*, Guttman) on both Twitter and Crimes datasets:
```bash
python benchmark_avg_query_time.py
```
This process:
1.  **Preprocesses** data (samples 100k records) if needed.
2.  **Checks Models**: Verifies trained NPN models exist in `models_repro/`. *(Note: Train models first using `train_custom.py`)*.
3.  **Benchmarks**: Runs Range, Point, and KNN queries for varying Leaf Capacities (M).
4.  **Plots**: 
    - Query Latency: `plots_avg_query_time/`
    - Build Time (Log Scale): `plots_build_time/`

### 2. Benchmark Single Custom Model
Score a specific model (e.g., your newly trained `model_k3.pth`) against a dataset:

```bash
python benchmark_avg_query_time.py --model trained_models/model_k3.pth --dataset twitter_100k.csv --query_type range
```

**Common Arguments:**
- `--model`: Path to model file OR `guttman`, `rstar`.
- `--dataset`: Path to CSV dataset (must have `min_0, min_1, max_0, max_1` columns).
- `--query_type`: `range`, `point`, or `knn`.
- `--max_entries`: Comma-separated list of leaf capacities to test (e.g. `"256,512,1024"`). Default is `"256,512,784,1024"`.
- `--query_count`: Number of queries to average over (Default: 100,000 for robustness, use less for quick tests).

- `--compare_baselines`: Add this flag to also run and plot R* and Guttman baselines on the same graph.

**Example:**
```bash
python benchmark_avg_query_time.py --model models/my_model.pth --dataset twitter_100k.csv --query_type range --max_entries "128,256" --query_count 1000 --compare_baselines
```

### 3. Overlap Ratio Analysis

The overlap ratio measures tree structural quality by computing the ratio of pairwise overlaps to total coverage. This metric helps evaluate how well different methods minimize spatial overlap.

**Automatic Tracking:**
Overlap ratios are automatically calculated and saved when running `benchmark_avg_query_time.py`. Results are stored in:
- Individual result files: `results_repro/{dataset}_{query_type}_{method}.json`
- Summary file: `results_repro/overlap_ratios.json`

**Formula:**
```
Overlap Ratio = (Σ all pairwise overlaps) / (Σ all child MBR volumes)
```

**Example Output:**
```json
{
  "twitter_100k.csv": {
    "range": {
      "model_k3.pth": { "256": 0.52, "512": 0.47 },
      "rstar": { "256": 0.01, "512": 0.01 },
      "guttman": { "256": 1.25, "512": 0.95 }
    }
  }
}
```

**Console Output:**
```
OVERLAP RATIO SUMMARY
================================================================================
Dataset: twitter_100k.csv
  Query Type: range
    model_k3.pth:
      M=256: 0.520000
      M=512: 0.470000
    rstar:
      M=256: 0.010000
      M=512: 0.010000
    guttman:
      M=256: 1.250000
      M=512: 0.950000
```

### 4. Node Access Efficiency - Varying Window Size

Benchmark node access efficiency across different relative window sizes to reproduce the paper's Node Access Efficiency Analysis.

**Script:** `benchmark_node_access.py`

**Usage:**
```bash
python benchmark_node_access.py \
  --model trained_models/model_k3.pth \
  --dataset twitter_100k.csv \
  --max_entries 256 \
  --num_queries 1000 \
  --output results_repro/node_access_twitter.json \
  --plot plots_avg_query_time/node_access_twitter.png
```

**Arguments:**
- `--model`: Path to trained `.pth` model file (required)
- `--dataset`: Path to CSV dataset (required)
- `--max_entries`: Max entries per node (default: 256)
- `--num_queries`: Number of queries per window size (default: 1000)
- `--device`: Device to use - `cpu` or `cuda` (default: cpu)
- `--output`: Output JSON file (default: node_access_results.json)
- `--plot`: Output plot file (default: node_access_plot.png)

**Features:**
- Tests 4 relative window sizes: **0.1%, 1%, 10%, 50%**
- Compares **Partitioning Model vs R* vs Guttman**
- Generates 1,000 random range queries per window size
- Tracks node accesses during tree traversal
- Produces publication-ready plots

**Quick Test (Small Dataset):**
```bash
python benchmark_node_access.py \
  --model trained_models/model_k3.pth \
  --dataset test_mixed_05000.csv \
  --max_entries 256 \
  --num_queries 100
```

**Example Output:**
```
Window Size     Partitioning    R*       Guttman
0.1%            38.71          58.0     84.0
1%              184.64         145.0    168.0
10%             461.09         232.0    252.0
50%             1135.70        319.0    336.0
```

**Full Reproduction (Both Datasets):**
```bash
# Twitter dataset
python benchmark_node_access.py \
  --model models_repro/range_model.pth \
  --dataset twitter_100k.csv \
  --max_entries 256 \
  --num_queries 1000 \
  --output results_repro/node_access_twitter.json \
  --plot plots_avg_query_time/node_access_twitter.png

# Crimes dataset
python benchmark_node_access.py \
  --model models_repro/range_model.pth \
  --dataset crimes_100k.csv \
  --max_entries 256 \
  --num_queries 1000 \
  --output results_repro/node_access_crimes.json \
  --plot plots_avg_query_time/node_access_crimes.png
```

## Output Files

### Query Time Benchmarks
- `results_repro/`: JSON files with benchmark results
- `plots_avg_query_time/`: Query latency plots
- `plots_build_time/`: Build time plots (log scale)

### Overlap Ratio Analysis
- `results_repro/overlap_ratios.json`: Summary of all overlap ratios by dataset and method

### Node Access Benchmarks
- `results_repro/node_access_*.json`: Node access counts for each window size
- `plots_avg_query_time/node_access_*.png`: Node access efficiency plots

### 5. Arizona State-of-the-Art Comparison

Compare your partitioning model against 7 state-of-the-art baselines (H4R, HR, PR, R*, STR, TGS, ACR) using precomputed baseline data from the paper.

**Script:** `benchmark_arizona.py`

**Requirements:**
1. Arizona dataset with **N=1,464,257 spatial objects**
2. Model parameters: **K=3, M=102**
3. Baseline data: `arizona_baselines.json` (precomputed from paper Table 3)

**Usage:**
```bash
python benchmark_arizona.py \
  --model models_repro/range_model.pth \
  --dataset arizona_1464257.csv \
  --max_entries 102 \
  --num_queries 1000 \
  --output results_repro/arizona_results.json \
  --plot plots_avg_query_time/arizona_comparison.png
```

**What it does:**
- Runs ONLY your partitioning model on Arizona dataset
- Tests with fixed window widths: **100m, 200m, 400m, 800m, 1600m**
- Compares against precomputed baseline data
- Generates comparison table and plot

**Arguments:**
- `--model`: Path to trained `.pth` model (required)
- `--dataset`: Path to Arizona CSV with 1,464,257 objects (required)
- `--max_entries`: Node capacity (use 102 for fair comparison)
- `--num_queries`: Queries per window width (default: 1000)
- `--device`: `cpu` or `cuda` (default: cpu)
- `--baselines`: Baseline JSON file (default: arizona_baselines.json)
- `--output`: Output JSON file
- `--plot`: Output plot file

**Example Output:**
```
Method      100m      200m      400m      800m      1600m
H4R         5.0e+07   1.2e+07   3.5e+06   8.0e+05   2.6e+05
HR          4.5e+07   1.1e+07   3.0e+06   7.5e+05   2.4e+05
PR          4.0e+07   1.0e+07   2.8e+06   7.0e+05   2.2e+05
R*          3.5e+07   9.0e+06   2.5e+06   6.5e+05   2.0e+05
STR         2.5e+07   7.0e+06   1.8e+06   5.0e+05   1.5e+05
TGS         2.0e+07   6.0e+06   1.5e+06   4.0e+05   1.2e+05
ACR         1.5e+07   3.5e+06   9.0e+05   2.5e+05   8.0e+04
Ours        [your results will appear here]
```

## Output Files (Updated)

### Query Time Benchmarks
- `results_repro/`: JSON files with benchmark results
- `plots_avg_query_time/`: Query latency plots
- `plots_build_time/`: Build time plots (log scale)

### Overlap Ratio Analysis
- `results_repro/overlap_ratios.json`: Summary of all overlap ratios

### Node Access Benchmarks  
- `results_repro/node_access_*.json`: Node access counts
- `plots_avg_query_time/node_access_*.png`: Node access plots

### Arizona Comparison
- `arizona_baselines.json`: Precomputed baseline data from paper
- `results_repro/arizona_results.json`: Your model's results
- `plots_arizona/arizona_comparison.png`: Comparison plot

### 6. Hyperparameter K Variation Study

Train models with different partition factors K and benchmark query latency.

**Script:** `benchmark_hyper_param_k.py`

**Usage:**
```bash
python benchmark_hyper_param_k.py \
  --train_dataset twitter_100k.csv \
  --test_dataset twitter_100k.csv \
  --k_values "2,3,4" \
  --max_entries 512 \
  --epochs 100
```

**Arguments:**
- `--train_dataset`, `--test_dataset`: CSV datasets (required)
- `--k_values`: Comma-separated K values (default: "2,3,4")
- `--max_entries`: Node capacity (default: 512)
- `--epochs`: Training epochs (default: 100)
- `--skip_training`: Use existing models

### 5. Arizona State-of-the-Art Comparison

Benchmark your partitioning model against state-of-the-art baselines (H4R, HR, PR, R*, STR, TGS, ACR) on the Arizona dataset with fixed window widths.

**Script:** `benchmark_arizona.py`

**Baseline Data:** `arizona_baselines.json` (precomputed comparison data from the paper)

**Usage:**
```bash
python benchmark_arizona.py \
  --model trained_models/arizona_model.pth \
  --dataset arizona_large.csv \
  --max_entries 102 \
  --num_queries 1000 \
  --output results_repro/arizona_results.json \
  --plot plots_avg_query_time/arizona_comparison.png
```

**Arguments:**
- `--model`: Path to trained `.pth` model (required)
- `--dataset`: Path to Arizona CSV dataset (required)
- `--max_entries`: Max entries per node (default: 102, as per paper protocol)
- `--num_queries`: Number of queries per window width (default: 1000)
- `--device`: Device to use - `cpu` or `cuda` (default: cpu)
- `--baselines`: Baseline data JSON file (default: arizona_baselines.json)
- `--output`: Output JSON file (default: arizona_results.json)
- `--plot`: Output plot file (default: arizona_comparison.png)

**Features:**
- Tests 5 fixed window widths: **100m, 200m, 400m, 800m, 1600m**
- Compares with **8 state-of-the-art methods** from the paper
- Uses **M=102** node capacity (paper protocol)
- Generates comparison table and log-scale plot

**Quick Test (Small Dataset):**
```bash
python benchmark_arizona.py \
  --model trained_models/model_k3.pth \
  --dataset test_mixed_05000.csv \
  --max_entries 102 \
  --num_queries 50
```

**Comparison Table Output:**
```
================================================================================
COMPARISON TABLE - Node Access (vs Window Width)
================================================================================
Method               100m         200m         400m         800m        1600m
--------------------------------------------------------------------------------
H4R                  5.0e+07      1.2e+07      3.5e+06      8.0e+05     2.6e+05
HR                   4.5e+07      1.1e+07      3.0e+06      7.5e+05     2.4e+05
PR                   4.0e+07      1.0e+07      2.8e+06      7.0e+05     2.2e+05
R*                   3.5e+07      9.0e+06      2.5e+06      6.5e+05     2.0e+05
STR                  2.5e+07      7.0e+06      1.8e+06      5.0e+05     1.5e+05
TGS                  2.0e+07      6.0e+06      1.5e+06      4.0e+05     1.2e+05
ACR                  1.5e+07      3.5e+06      9.0e+05      2.5e+05     8.0e+04
Ours (Paper)         4.7e+06      1.2e+06      3.1e+05      6.4e+04     2.4e+04
--------------------------------------------------------------------------------
Ours (Current)       [your results will appear here]
================================================================================
```

**Full Arizona Dataset Test:**
```bash
# Assuming you have arizona_large.csv (N=1,464,257)
python benchmark_arizona.py \
  --model models_repro/range_model.pth \
  --dataset arizona_large.csv \
  --max_entries 102 \
  --num_queries 1000 \
  --output results_repro/arizona_results.json \
  --plot plots_avg_query_time/arizona_comparison.png
```

## Output Files (Updated)

### Query Time Benchmarks
- `results_repro/`: JSON files with benchmark results
- `plots_avg_query_time/`: Query latency plots
- `plots_build_time/`: Build time plots (log scale)

### Overlap Ratio Analysis
- `results_repro/overlap_ratios.json`: Summary of all overlap ratios by dataset and method

### Node Access Benchmarks
- `results_repro/node_access_*.json`: Node access counts for each window size
- `plots_avg_query_time/node_access_*.png`: Node access efficiency plots

### Arizona State-of-the-Art Comparison
- `arizona_baselines.json`: Precomputed baseline data from paper (H4R, HR, PR, R*, STR, TGS, ACR)
- `results_repro/arizona_results.json`: Your model's results on Arizona dataset
- `plots_arizona/arizona_comparison.png`: Comparison plot with all methods

### 7. Impact of Dimensionality Study

Benchmark query latency across varying dimensions D (e.g., 2, 8, 20, 50, 100).
Reproduces the "Impact of Dimensionality" experiment.

**Scripts:**
- `benchmark_impact_of_dims.py` (Main orchestrator)
- `generate_synthetic_ndim.py` (Data generator)
- `train_ndim_model.py` (N-dim model trainer)

**Usage:**
```bash
python benchmark_impact_of_dims.py \
  --dims "2,8,20,50,100" \
  --output_dir ablation_dims \
  --dataset_size 20000 \
  --epochs 20
```

**Generate Specific N-Dim Data:**
```bash
python generate_synthetic_ndim.py --dim 50 --size 20000 --output data_50d.csv
```

**Train Specific N-Dim Model:**
```bash
python train_ndim_model.py --dataset data_50d.csv --output model_50d.pth --dim 50
```
