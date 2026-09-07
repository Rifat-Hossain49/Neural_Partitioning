# Reproducing the WAHARP artifact

## 1. Verify the compact release

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/smoke_test.py
python -m unittest discover -s tests -v
```

This verifies the frozen checkpoint, protocol records, result-table shape,
action space, feature dimension, exact page quotas, and tree capacity.

## 2. Historical experiment entry points

The scripts under `scripts/kaggle/` are the entry points used by the recorded
experiments. They preserve their original `/kaggle/input` discovery and strict
protocol checks:

```text
run_loss_weight_ablation.py  Six loss mixtures, three paired seeds
run_final_train.py           Three final members with 0.75/0.25 weights
run_locked_eval.py           Nine locked dataset-capacity cells
run_fullscale.py             Capacity-sweep baseline materialization
```

Run them in this order:

1. Materialize the 30,000 teacher states: 10,000 from each of Twitter, Crimes,
   and Arizona, with 80 action targets per state.
2. Attach that teacher-state bundle and run `run_loss_weight_ablation.py`.
3. Attach the completed ablation output and run `run_final_train.py`.
4. Attach the three datasets, frozen final-training output, fixed workload
   manifests, and baseline per-query outputs; then run `run_locked_eval.py`.
5. Aggregate the resulting dataset outputs with
   `scripts/aggregate_waharp_loss075_locked_eval.py`.

The scripts reject missing, ambiguous, or protocol-incompatible inputs. This
is deliberate: a partial run must not be mistaken for the locked experiment.

## 3. Protocol details

| Item | Locked value |
|---|---:|
| State feature dimension | 498 |
| Candidate actions | 80 |
| Teacher states | 30,000 |
| Action targets | 2,400,000 |
| Loss mixtures tested | 6 |
| Paired seeds per mixture | 3 |
| Selected listwise/classification weights | 0.75 / 0.25 |
| Final ensemble candidates | 3 |
| Selected member | 2 |
| Capacities | 128, 256, 512 |
| Final queries per dataset-capacity cell | 6,000 |

The loss mixture minimizes the mean across paired seeds of
`worst-domain selected regret + 0.25 * mean selected regret`. Final-member
selection minimizes the worst-domain validation access ratio plus 0.25 times
the mean ratio. Neither step reads final-query results.

## 4. Published compact outputs

The release includes aggregate and cell-level result tables, construction
records, and tree metrics. It omits the 54,000-row per-method query traces
because they are large and redundant with the included paired summaries. The
included `FINAL_STATE.json` records zero correctness mismatches and zero
baseline rebuilds in the locked pass.
