# WAHARP

[![Artifact checks](https://github.com/Rifat-Hossain49/Neural_Partitioning/actions/workflows/artifact-checks.yml/badge.svg)](https://github.com/Rifat-Hossain49/Neural_Partitioning/actions/workflows/artifact-checks.yml)

Official implementation and compact evaluation artifact for **WAHARP:
Amortizing Workload-Aware R-tree Packing with a Neural Policy**.

WAHARP is a static, workload-aware R-tree bulk loader. A frozen neural policy
scores 80 projection-and-page-quota actions while constructing each level of
the hierarchy. Exact quotas preserve every entry, enforce node capacity, and
produce equal-depth leaves without a fallback packing algorithm. Queries use
ordinary R-tree traversal; the neural model is used only during construction.

## Release status

The repository contains the selected checkpoint used by the paper's locked
evaluation:

```text
artifacts/model/selected.pt
```

Its SHA-256 digest, training protocol, and validation-only member selection are
recorded in `artifacts/model/FINAL_TRAINING_STATE.json` and
`artifacts/model/SHADOW_SELECTION.json`. The earlier 0.60/0.40 checkpoint,
which the paper compares against and the baseline runners evaluate, is in
`artifacts/model/previous_060_040/`. Other candidate checkpoints and temporary
training snapshots are intentionally excluded.

The locked construction and final query workloads are committed under
`artifacts/workloads/`, so any WAHARP result in the paper can be rebuilt from
the checkpoint and the raw dataset (see
[Reproducing the paper experiments](#reproducing-the-paper-experiments)).

A later validation-only teacher-cost sweep selected
`lambda_overlap = 1e-4` and `lambda_margin = 2e-4`. Its compact records are in
`artifacts/teacher_cost_ablation/`. That pair has **not** replaced the frozen
paper checkpoint: it requires fresh DAgger trajectories, full-domain training,
member selection, and locked evaluation first.

## Install and verify

Python 3.10-3.12 is recommended. CPU execution is sufficient for verification
and small custom datasets.

```bash
git clone https://github.com/Rifat-Hossain49/Neural_Partitioning.git
cd Neural_Partitioning
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python scripts/verify_artifacts.py
python scripts/make_analysis_appendix.py --check
python scripts/smoke_test.py
python -m unittest discover -s tests -v
```

These checks verify the checkpoint hash, feature and action contracts, result
manifests, workload hashes, query-correctness gates, exact page quotas, the
analysis appendix, and construction at capacities 128, 256, and 512.

## Run on a custom dataset

The interactive notebook is the easiest entry point:

[`notebooks/build_and_compare_custom_dataset.ipynb`](notebooks/build_and_compare_custom_dataset.ipynb)

It accepts:

- `.npy` arrays with shape `N x 2` for points or `N x 4` for rectangles;
- CSV files with `x,y`, longitude/latitude aliases, or
  `xmin,ymin,xmax,ymax` columns.

Coordinates are independently normalized to `[0,1]` on each axis. The
notebook generates one deterministic construction workload and one shared
evaluation workload, builds every requested method, measures construction
time and logical node accesses, and checks range, point, and kNN correctness.
It runs a synthetic demonstration when no data path is supplied.

The same workflow is available from the command line:

```bash
python scripts/custom_dataset_benchmark.py \
  --data /path/to/rectangles.npy \
  --capacity 128 \
  --methods WAHARP,STR,TGS \
  --output custom_benchmark_output
```

`Guttman` and `RStar` may also be added to `--methods`; their incremental
insertion can be slow on large inputs. The output directory contains:

```text
SUMMARY.csv             Build time, total accesses, ratios, and validity
BY_QUERY_FAMILY.csv     Range, point, and kNN access counts
PER_QUERY.csv           Per-query measurements and correctness signatures
RUN_CONFIG.json         Dataset normalization and complete run configuration
```

The custom-data workflow evaluates transfer of the released checkpoint. It
does not fine-tune WAHARP on the supplied dataset.

## Recorded results

The frozen all-domain model was evaluated on full Arizona, Crimes, and Twitter
datasets at `B in {128,256,512}`, with 6,000 paired queries per cell.

| Comparison | Geometric-mean change in logical node accesses |
|---|---:|
| WAHARP vs. PLATON | +2.56% |
| WAHARP vs. STR | -9.49% |
| WAHARP vs. TGS | -21.21% |

Positive values mean WAHARP used more accesses; negative values mean it used
fewer. All 54,000 WAHARP/PLATON query pairs passed the correctness checks.

[`artifacts/WAHARP_ANALYSIS_APPENDIX.md`](artifacts/WAHARP_ANALYSIS_APPENDIX.md)
is the companion appendix cited by the paper. It lists every reported value:
cell-level ratios with paired confidence intervals, construction times with
the PLATON MCTS/native split, the PLATON search-budget sweep, query-family and
internal/leaf access, paired outcomes, action usage, tree structure, and
leave-one-dataset-out transfer. The data behind each paper figure is in
`artifacts/paper_figures/`. Both are regenerated from the raw records by
`scripts/make_analysis_appendix.py`.

The leave-one-dataset-out experiment trains one policy on each pair of source
datasets and evaluates it zero-shot on the third. Its fold provenance,
confidence intervals, correctness records, and plot are under
`artifacts/generalization/`.

## Baseline scope

The custom notebook provides WAHARP, STR, TGS, Guttman, and R*-tree through the
same Python evaluation path. The paper's PLATON numbers use the authors'
implementation, its native `libspatialindex` build, 100 simulation steps, and
the recorded rollout budget. That third-party source bundle is not
redistributed here. Once it is supplied, the strict historical entry points in
`scripts/kaggle/` reproduce the PLATON protocol. A substitute reimplementation
should not be reported as the paper's PLATON baseline.

## Repository contents

```text
realtrain/                           WAHARP actions, features, training, and trees
rtreelib/                            Vendored R-tree implementation and baselines
native/                              Native evaluation additions used with PLATON
notebooks/                           Custom-data build and comparison notebook
scripts/custom_dataset_benchmark.py  Custom benchmark runner
scripts/reproduce_locked_cell.py     Rebuild one paper cell and compare with the record
scripts/make_analysis_appendix.py    Regenerate the analysis appendix and figure data
scripts/make_kaggle_bundles.py       Assemble code bundles for the historical scripts
scripts/kaggle/                      Historical experiment entry points and bundle markers
scripts/verify_artifacts.py          Compact-artifact integrity checks
artifacts/WAHARP_ANALYSIS_APPENDIX.md  Companion appendix cited by the paper
artifacts/paper_figures/             Data behind the paper's figures
artifacts/model/                     Frozen paper checkpoint, selection records, previous checkpoint
artifacts/workloads/                 Locked construction and final query arrays
artifacts/final_evaluation/          Locked evaluation results and construction times
artifacts/construction/              Per-cell build records and action counts
artifacts/tree_structure/            Per-cell tree metrics
artifacts/generalization/            Leave-one-dataset-out evaluation
artifacts/platon_budget/             PLATON rollout-budget sensitivity
artifacts/ablation/                  Validation-only loss-weight ablation
artifacts/teacher_cost_ablation/     Validation-only coefficient sweep
docs/                                Dataset and reproduction instructions
tests/                               Action, policy, teacher-cost, and benchmark unit tests
```

The internal package remains named `realtrain` to preserve direct
correspondence with the executed experiment scripts.

## Reproducing the paper experiments

With one raw dataset (see [`docs/DATASETS.md`](docs/DATASETS.md)), any locked
cell can be rebuilt on a CPU and checked against the recorded accesses,
decisions, action counts, and tree structure:

```bash
python scripts/reproduce_locked_cell.py --dataset arizona --capacity 512 \
  --data /path/to/arizona_buildings_n1464257_epsg26912.npy
```

[`docs/REPRODUCING.md`](docs/REPRODUCING.md) describes this check and the
ordered training and evaluation protocol: teacher states, loss ablation, final
training, baselines, locked evaluation, leave-one-dataset-out transfer, and
the PLATON budget sweep. It also covers the recorded environment.
Historical scripts deliberately reject missing, ambiguous, or
protocol-incompatible inputs so that partial runs cannot be mistaken for
locked results.

Raw datasets, credentials, per-query archives, caches, logs, temporary Kaggle
outputs, and superseded checkpoints are not committed. Cryptographic hashes
inside JSON state files are provenance checks, not scientific measurements.

## License and citation

The project is released under the MIT License. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for vendored code and dataset
notices. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
