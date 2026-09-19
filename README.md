# WAHARP

[![Artifact checks](https://github.com/Rifat-Hossain49/Neural_Partitioning/actions/workflows/artifact-checks.yml/badge.svg)](https://github.com/Rifat-Hossain49/Neural_Partitioning/actions/workflows/artifact-checks.yml)

Official code and compact evaluation artifact for **WAHARP: Amortizing
Workload-Aware R-tree Packing with a Neural Policy**.

WAHARP is a static, workload-aware R-tree bulk loader. At each recursive
packing state, a neural policy scores 80 projection-and-page-quota actions.
The selected action partitions entries using exact page quotas, so every leaf
respects capacity without a post-processing repair or geometric fallback.

## Artifact status

This repository contains the implementation and the frozen model used for the
locked final evaluation. The selected model uses a 0.75 listwise and 0.25
classification loss mixture, chosen using validation data only. Final queries
were not read during loss-weight selection, training, or member selection.

The locked evaluation covers three datasets and capacities 128, 256, and 512:

| Comparison | Cell-geometric-mean change in logical accesses |
|---|---:|
| WAHARP vs. PLATON | +2.56% |
| WAHARP vs. STR | -9.49% |
| WAHARP vs. TGS | -21.21% |

Positive values mean WAHARP uses more logical accesses; negative values mean
fewer. Across all nine dataset-capacity cells, the evaluation used 6,000 fixed
queries per cell and reported zero correctness mismatches. The full cell-level
results and confidence intervals are in
[`artifacts/final_evaluation/`](artifacts/final_evaluation/).

### Leave-one-dataset-out generalization

A zero-shot leave-one-dataset-out experiment trains three additional policies,
each using only the other two datasets. Fold-specific DAgger states are
regenerated from the source domains; held-out data and final queries are not
read until after member selection and checkpoint freezing. Across the nine
held-out dataset-capacity cells, WAHARP uses 2.1% to 34.5% more logical node
accesses than PLATON and 0.3% to 28.5% more than the all-domain WAHARP model,
with zero correctness mismatches. The exact paired results, confidence
intervals, fold provenance, and plot are in
[`artifacts/generalization/`](artifacts/generalization/).

## What is included

```text
realtrain/                  Core WAHARP implementation
rtreelib/                   Vendored R-tree data structure and baselines
native/                     Native PLATON and dynamic R-tree evaluation code
scripts/smoke_test.py       Frozen-model construction smoke test
scripts/verify_artifacts.py Integrity and completeness checks
scripts/kaggle/             Exact historical training/evaluation entry points
artifacts/ablation/         Loss-weight selection records
artifacts/model/            Frozen checkpoint and member-selection records
artifacts/final_evaluation/ Locked aggregate query results
artifacts/generalization/   Leave-one-dataset-out zero-shot results
artifacts/construction/     Build-time and action-audit records
artifacts/tree_structure/   Topology and structural metrics for nine cells
docs/                       Data and reproduction instructions
```

The internal Python package remains named `realtrain` to preserve direct
correspondence with the executed experiment scripts.

## Quick start

Python 3.10-3.12 is recommended. A CPU-only installation is sufficient for the
checks below.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/smoke_test.py
python -m unittest discover -s tests -v
```

The smoke test loads `artifacts/model/selected.pt`, verifies the 498-feature,
80-action contract, and constructs valid trees at all three reported
capacities using deterministic synthetic rectangles and queries.

## Reproduction

There are three practical reproducibility levels:

1. **Artifact verification:** run the quick-start checks without downloading
   any benchmark data.
2. **Frozen-model evaluation:** obtain the datasets described in
   [`docs/DATASETS.md`](docs/DATASETS.md), then use the locked evaluation entry
   point and its documented input layout.
3. **Full retraining:** materialize the teacher-state bundle, run the six-point
   loss sweep, train three final members, select on validation access ratios,
   and finally execute the locked evaluation.

Exact commands and protocol boundaries are documented in
[`docs/REPRODUCING.md`](docs/REPRODUCING.md). The historical Kaggle entry
points intentionally fail when required inputs or protocol identifiers do not
match; this prevents accidental mixing of runs.

## Key artifact records

| Record | Purpose |
|---|---|
| `artifacts/ablation/FINAL_DECISION.json` | Selected loss weights and validation-only rule |
| `artifacts/model/FINAL_TRAINING_STATE.json` | Frozen model and training protocol |
| `artifacts/model/SHADOW_SELECTION.json` | Three-member selection statistics |
| `artifacts/final_evaluation/FINAL_STATE.json` | Locked evaluation completion gates |
| `artifacts/final_evaluation/ALL_PAIRWISE_SUMMARY.csv` | Nine cell-level comparisons |
| `artifacts/generalization/LODO_RESULTS.csv` | Leave-one-dataset-out paired comparisons |
| `artifacts/generalization/FOLD_TRAINING.csv` | Source domains and frozen fold checkpoints |
| `artifacts/construction/*_CONSTRUCTION_ALL.json` | Construction times and action audits |
| `artifacts/tree_structure/*_TREE_STRUCTURE.json` | Tree topology and occupancy |

Cryptographic hashes are retained inside machine-readable state files for
artifact integrity. They are provenance metadata, not scientific results.

## Data policy

Raw benchmark datasets, per-query dumps, temporary archives, caches, logs,
credentials, and superseded checkpoints are intentionally excluded. This
keeps the artifact reviewable and avoids redistributing third-party data.

## License and citation

The code is released under the MIT License. See `THIRD_PARTY_NOTICES.md` for
vendored code and dataset notices. Citation metadata is available in
`CITATION.cff`; publication venue and year should be added after acceptance.
