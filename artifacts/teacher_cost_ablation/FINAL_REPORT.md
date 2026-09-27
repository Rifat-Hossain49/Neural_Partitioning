# WAHARP Teacher-Cost Coefficient Ablation

Protocol: `waharp_teacher_cost_ablation_v1_20260926`

This is a validation-only, paired ablation. It does not read final queries or PLATON results.
All configurations use the same 30,000 states, three training seeds, 0.75/0.25 listwise/best-action loss, shadow objects, and validation queries.
The initial states and historical-policy DAgger trajectories are held fixed across coefficient pairs; only the teacher costs and resulting targets are recomputed.

## Grid

- Overlap coefficients: `(0.0, 1e-05, 0.0001, 0.001)`
- Margin coefficients: `(0.0, 2e-06, 2e-05, 0.0002)`
- Historical pair: `0.0001` / `2e-05`

## Validation results

| Overlap | Margin | Score | Twitter ratio | Crimes ratio | Arizona ratio |
|---:|---:|---:|---:|---:|---:|
| 1e-04 | 2e-04 | 1.311754 +/- 0.046195 | 1.014334 | 1.053224 | 1.010401 |
| 1e-04 | 2e-05 | 1.317059 +/- 0.033522 | 1.009403 | 1.058953 | 1.028917 |
| 1e-03 | 2e-05 | 1.329668 +/- 0.029077 | 1.045728 | 1.039297 | 1.031812 |
| 1e-03 | 2e-04 | 1.330389 +/- 0.068124 | 1.032958 | 1.046372 | 1.048378 |
| 0e+00 | 2e-05 | 1.342666 +/- 0.083538 | 1.038486 | 1.078922 | 1.047513 |
| 1e-05 | 2e-04 | 1.348241 +/- 0.063899 | 1.031783 | 1.076898 | 1.055065 |
| 1e-05 | 2e-05 | 1.348858 +/- 0.082638 | 1.035648 | 1.074774 | 1.045150 |
| 0e+00 | 2e-04 | 1.355287 +/- 0.060826 | 1.029209 | 1.086293 | 1.051730 |
| 1e-04 | 2e-06 | 1.366735 +/- 0.047490 | 1.028569 | 1.102410 | 1.040919 |
| 1e-03 | 0e+00 | 1.376778 +/- 0.050492 | 1.029918 | 1.112322 | 1.031239 |
| 1e-03 | 2e-06 | 1.389050 +/- 0.049176 | 1.022538 | 1.116458 | 1.047908 |
| 1e-05 | 0e+00 | 1.401636 +/- 0.069477 | 1.044809 | 1.130666 | 1.076158 |
| 0e+00 | 2e-06 | 1.403468 +/- 0.046641 | 1.098543 | 1.125814 | 1.107491 |
| 1e-04 | 0e+00 | 1.404010 +/- 0.051636 | 1.040304 | 1.135919 | 1.040861 |
| 1e-05 | 2e-06 | 1.414415 +/- 0.024588 | 1.073198 | 1.139181 | 1.089948 |
| 0e+00 | 0e+00 | 1.461910 +/- 0.042233 | 1.113090 | 1.171883 | 1.129654 |

## Selection

Selected overlap coefficient: **1e-04**.
Selected margin coefficient: **2e-04**.
Selected validation score: **1.311754 +/- 0.046195**.
Historical validation score: **1.317059 +/- 0.033522**.
Paired selected-minus-historical score differences: `[0.005786384674207001, -0.0838729958118416, 0.06216909269132609]`.

Selection minimizes the mean across three paired seeds of the worst-domain validation-access ratio plus 0.25 times the mean domain ratio. Ratios are normalized per seed and domain against the best grid configuration.

## Interpretation boundary

This experiment selects teacher-cost coefficients using only validation query node accesses. It does not support a final-test claim by itself.
Because the paired study uses a fixed historical-policy trajectory bank, a selected non-historical pair must be rerun through its own full DAgger/final-training pipeline and locked final evaluation before changing the paper's main results.

## Label sensitivity

| Overlap | Margin | Changed vs hit-only | Changed vs historical |
|---:|---:|---:|---:|
| 0e+00 | 0e+00 | 0.0000% | 30.8100% |
| 0e+00 | 2e-06 | 14.5567% | 23.5200% |
| 0e+00 | 2e-05 | 30.2833% | 12.9200% |
| 0e+00 | 2e-04 | 35.9600% | 15.0267% |
| 1e-05 | 0e+00 | 9.5800% | 25.2600% |
| 1e-05 | 2e-06 | 17.6733% | 19.3500% |
| 1e-05 | 2e-05 | 30.0700% | 11.3733% |
| 1e-05 | 2e-04 | 35.9033% | 14.8233% |
| 1e-04 | 0e+00 | 19.0900% | 16.2733% |
| 1e-04 | 2e-06 | 23.7033% | 12.4033% |
| 1e-04 | 2e-05 | 30.8100% | 0.0000% |
| 1e-04 | 2e-04 | 35.3733% | 12.8833% |
| 1e-03 | 0e+00 | 26.1267% | 19.9300% |
| 1e-03 | 2e-06 | 27.1733% | 17.5333% |
| 1e-03 | 2e-05 | 30.3133% | 12.2100% |
| 1e-03 | 2e-04 | 33.9133% | 8.5200% |
