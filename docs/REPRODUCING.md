# Reproducing the WAHARP paper

There are three levels of reproduction, from minutes on a laptop to the full
multi-day Kaggle protocol. Paths below are relative to `code/`, and commands
are run from it. `bash run` (the Code Ocean Reproducible Run entry point)
performs levels 1 and 2 for the Arizona cells and writes its reports to
`../results/`.

| Level | What it shows | Needs | Time |
|---|---|---|---|
| 1. Verify the artifact | Checkpoint, records, tables, and figure data are consistent | This repository | ~1 minute |
| 2. Reproduce locked cells | The frozen checkpoint rebuilds the paper's trees and node-access counts exactly | Arizona is included; Twitter or Crimes must be downloaded | ~2 minutes per Arizona cell on a CPU |
| 3. Rerun the protocol | Teacher labels, training, selection, baselines, and evaluation from scratch | All datasets, PLATON author code, Kaggle-scale compute | Days |

## 1. Verify the artifact

```bash
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/make_kaggle_bundles.py --check
python scripts/make_analysis_appendix.py --check
python scripts/smoke_test.py
python -m unittest discover -s tests -v
```

`verify_artifacts.py` checks the checkpoint hash, protocol records, result-table
shapes, workload hashes, and correctness gates. `make_kaggle_bundles.py --check`
confirms that the code is identical to the code that ran the recorded
experiments. `make_analysis_appendix.py --check`
confirms that [`artifacts/WAHARP_ANALYSIS_APPENDIX.md`](../code/artifacts/WAHARP_ANALYSIS_APPENDIX.md)
and the figure data in `artifacts/paper_figures/` match the committed records.

## 2. Reproduce locked-evaluation cells

The locked construction workload (2,800 boxes) and the 6,000 final queries for
every dataset are committed under `artifacts/workloads/<dataset>/`. The same
arrays are used at all three capacities. The Arizona dataset is included in
`../data/`; Twitter and Crimes are listed in [`DATASETS.md`](DATASETS.md):

```bash
python scripts/reproduce_locked_cell.py --dataset arizona --capacity 512 \
    --data ../data/arizona_buildings_n1464257_epsg26912.npy
```

The script refuses to run if the normalized dataset hash differs from the
evaluated snapshot. It then builds the WAHARP tree with
`artifacts/model/selected.pt`, evaluates all 6,000 final queries, and compares
with the recorded artifact:

- total and per-workload logical node accesses;
- neural decision count and the full action-usage histogram;
- leaf-page count and tree structure (height, fill, sibling overlap).

All three Arizona cells (B = 128, 256, 512) reproduce exactly on CPU, on
Windows with Python 3.14 and PyTorch 2.13. Crimes and Twitter use the same
code path but take longer to build (Table 9 decision counts). The WAHARP side
of every cell can be reproduced this way. The PLATON side needs the authors'
code (Section 3).

## 3. Rerun the full protocol

The scripts in `scripts/kaggle/` are the entry points that produced the
recorded results. Each one searches `/kaggle/input` for a code bundle and
for the outputs of earlier steps, and it refuses missing, ambiguous, or
protocol-incompatible inputs, so a partial run cannot be mistaken for a locked
result. Build the code bundles with:

```bash
python scripts/make_kaggle_bundles.py --output kaggle_bundles_out
```

This writes `realtrain_v1/`, `final_fullscale_b128/`, and `final_capacity_sweep/`,
each containing the code and the recorded marker
file its scripts look for. The script also checks the repository code against the hashes of the code
that ran the experiments (recorded in
`scripts/kaggle/bundle_markers/realtrain_v1/EXECUTED_CODE_MANIFEST.csv`). The
`realtrain/`, `rtreelib/`, and `native/` code in this repository is identical
to that code.

Attach the datasets, the relevant bundle, and the previous step's output as
Kaggle inputs, then run the steps in order:

| Step | Command | Bundle | Protocol id | Output used by |
|---|---|---|---|---|
| 1. Workloads, 24,000 initial teacher states, bootstrap policy, 6,000 DAgger states (80 action costs each) | `python -m realtrain.pipeline` | `realtrain_v1` | `waharp_realtrain_v1_three_domain_pageio_20260829` | 2, 3, 6 |
| 2. Loss-weight ablation (6 mixtures x 3 seeds, validation only) | `scripts/kaggle/run_loss_weight_ablation.py` | `realtrain_v1` | `waharp_loss_weight_ablation_v1_20260902` | 3 |
| 3. Final training (3 members, 0.75/0.25) and validation-only member selection | `scripts/kaggle/run_final_train.py` | `realtrain_v1` | `waharp_loss075_final_train_v1_20260903` | 5, 6, 7 |
| 4a. B = 128 baselines (PLATON, STR, TGS), locked final queries | `scripts/kaggle/run_fullscale_b128.py --dataset <name>` | `final_fullscale_b128` | `waharp_final_fullscale_b128_member2_20260831` | 4b, 5 |
| 4b. B = 256/512 baselines on the same queries | `scripts/kaggle/run_fullscale.py --dataset <name> --capacity <256\|512>` | `final_capacity_sweep` | `waharp_final_capacity_sweep_member2_v1_20260901` | 5 |
| 5. Locked evaluation of the selected checkpoint (9 cells) | `scripts/kaggle/run_locked_eval.py --dataset <name>` | `final_capacity_sweep` | `waharp_loss075_locked_eval_v1_20260903` | paper Sections 7.2-7.9 |
| 6. Leave-one-dataset-out transfer | `scripts/kaggle/run_lodo_zero_shot.py` | `final_capacity_sweep` | `waharp_lodo_zero_shot_v1_20260919` | paper Section 7.10 |
| 7. PLATON search-budget sensitivity (rollouts 1/5/10/25, B = 128) | `scripts/kaggle/run_platon_budget_b128.py` | `final_capacity_sweep` | `waharp_platon_budget_b128_v1_20260907` | paper Section 7.3 |

Step 1 (`python -m realtrain.pipeline --input-root <inputs> --output-root <dir>`)
runs the complete V1 pipeline with the recorded configuration (stored in
`scripts/kaggle/bundle_markers/realtrain_v1/WAHARP_REALTRAIN_V1_PROTOCOL.json`).
It also trains, selects, and evaluates the earlier 0.60/0.40 model. Steps 2
and 3 only need its `teacher/initial_*.npz` and `teacher/dagger_*.npz` files.
If the PLATON code is not available, pass `--config` with a JSON file that sets
`run_platon`, `run_guttman_native`, and `run_rstar_native` to `false`. Those
flags only affect step 1's own baseline evaluation, not the teacher states.
Step 5's aggregation is
`python scripts/aggregate_waharp_loss075_locked_eval.py --twitter <dir> --crimes <dir> --arizona <dir> --training <dir> --output <dir>`.
The 0.60/0.40 checkpoint that
steps 4a and 4b evaluate is committed at
`artifacts/model/previous_060_040/member_2.pt`. `preflight_capacity_sweep.py`
is an optional pre-run check of the capacity-sweep bundle (`--bundle <dir>`).

Steps 4, 6, and 7 run the authors' PLATON code (`RtreeEnv`, `MCTS`, `Node`, and
the bundled libspatialindex 1.9.3) with 100 simulation steps, nominal node
capacity B/0.8 with fill factor 0.8, and 4,096-byte pages. That source is not
redistributed here. Obtain it from the PLATON authors and attach it as an input.
`native/` contains the additional native programs used to bulk-load and query
those trees. A substitute reimplementation should not be reported as the
paper's PLATON baseline.

## Protocol details

| Item | Locked value |
|---|---:|
| State feature dimension | 498 |
| Candidate actions | 80 |
| Teacher states | 30,000 (24,000 initial + 6,000 DAgger) |
| Action targets | 2,400,000 |
| Teacher cost weights (overlap / margin) | 1e-4 / 2e-5 |
| Loss mixtures tested | 6 |
| Paired seeds per mixture | 3 |
| Selected listwise/classification weights | 0.75 / 0.25 |
| Final ensemble candidates | 3 |
| Selected member | 2 |
| Capacities | 128 (trained), 256, 512 |
| Construction queries per dataset | 2,800 |
| Final queries per dataset-capacity cell | 6,000 |

The loss mixture minimizes the mean across paired seeds of
`worst-domain selected regret + 0.25 * mean selected regret`. Final-member
selection minimizes the worst-domain validation access ratio plus 0.25 times
the mean ratio. Neither step reads final-query results.

## Recorded environment

The recorded runs used Kaggle's Python 3.12 image. The selected member's final
record reports CPU execution without GPUs or DataParallel
(`artifacts/model/member_2.training.json`). Exact package versions were not
captured. The executed bundle required `numpy`, `scipy`,
`pandas`, `torch`, and `tqdm` without pins. `scipy` must be installed: without
it, the kNN-surrogate construction boxes fall back to a density estimate and
differ from the recorded workloads.

Construction time includes only method-specific building. Because the records
do not establish identical hardware, timing ratios retain their method-specific
definition (see Section 3 of the analysis appendix). Logical node accesses do
not depend on hardware.

## Published outputs

| Location | Contents |
|---|---|
| `artifacts/WAHARP_ANALYSIS_APPENDIX.md` | Companion appendix cited by the paper: every reported table and figure value |
| `artifacts/paper_figures/` | Data behind Figures 2-6 and the PLATON-budget comparison |
| `artifacts/model/` | Selected checkpoint, training and selection records, previous 0.60/0.40 checkpoint |
| `artifacts/workloads/` | Locked construction, validation, and final query arrays with hashes |
| `artifacts/final_evaluation/` | Locked results, confidence intervals, and construction times |
| `artifacts/construction/`, `artifacts/tree_structure/` | Per-cell build records, action counts, and tree metrics |
| `artifacts/generalization/` | Leave-one-dataset-out folds and results |
| `artifacts/platon_budget/` | PLATON rollout-budget results, its fresh workload, and construction times |
| `artifacts/ablation/` | Validation-only loss-weight ablation |

The 54,000-row per-method query traces are omitted because of their size. The
paired summaries and per-workload totals derived from them are included, and `reproduce_locked_cell.py` regenerates the WAHARP traces.
