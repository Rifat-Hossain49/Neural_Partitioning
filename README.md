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
`artifacts/model/SHADOW_SELECTION.json`. Candidate checkpoints and temporary
training snapshots are intentionally excluded.

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
python scripts/smoke_test.py
python -m unittest discover -s tests -v
```

These checks verify the checkpoint hash, feature and action contracts, result
manifests, query-correctness gates, exact page quotas, and construction at
capacities 128, 256, and 512.

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
Cell-level ratios, paired confidence intervals, query-family results, and
construction records are under `artifacts/final_evaluation/`,
`artifacts/construction/`, and `artifacts/tree_structure/`.

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
scripts/custom_dataset_benchmark.py Custom benchmark runner
scripts/kaggle/                      Historical experiment entry points
scripts/verify_artifacts.py          Compact-artifact integrity checks
artifacts/model/                     Frozen paper checkpoint and selection records
artifacts/final_evaluation/          Locked aggregate evaluation results
artifacts/generalization/            Leave-one-dataset-out evaluation
artifacts/teacher_cost_ablation/     Validation-only coefficient sweep
docs/                                Dataset and reproduction instructions
tests/                               Action and teacher-cost unit tests
```

The internal package remains named `realtrain` to preserve direct
correspondence with the executed experiment scripts.

## Reproducing the paper experiments

[`docs/REPRODUCING.md`](docs/REPRODUCING.md) describes the ordered training and
evaluation protocols. [`docs/DATASETS.md`](docs/DATASETS.md) lists the expected
dataset sources and representations. Historical scripts deliberately reject
missing, ambiguous, or protocol-incompatible inputs so that partial runs
cannot be mistaken for locked results.

Raw datasets, credentials, per-query archives, caches, logs, temporary Kaggle
outputs, and superseded checkpoints are not committed. Cryptographic hashes
inside JSON state files are provenance checks, not scientific measurements.

## License and citation

The project is released under the MIT License. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for vendored code and dataset
notices. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
