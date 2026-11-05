# WAHARP Leave-One-Dataset-Out Generalization

Protocol: `waharp_lodo_zero_shot_v1_20260919`

Three source-only models were trained. For each fold, the held-out dataset was excluded from initial states, bootstrap training, fold-specific DAgger, final training, and member selection. The checkpoint was frozen before held-out data and final queries were loaded.

## Overall logical node accesses

| Held-out dataset | B | LODO / PLATON | 95% CI | LODO / all-domain | Wins / ties / losses vs PLATON |
|---|---:|---:|---:|---:|---:|
| Twitter | 128 | 1.0206 | [1.0193, 1.0221] | 1.0029 | 538 / 551 / 4911 |
| Twitter | 256 | 1.0250 | [1.0233, 1.0266] | 1.0068 | 774 / 1000 / 4226 |
| Twitter | 512 | 1.0348 | [1.0329, 1.0370] | 1.0134 | 712 / 1050 / 4238 |
| Crimes | 128 | 1.1484 | [1.0935, 1.2678] | 1.1270 | 128 / 4261 / 1611 |
| Crimes | 256 | 1.2292 | [1.1460, 1.4555] | 1.1957 | 157 / 4366 / 1477 |
| Crimes | 512 | 1.0870 | [1.0637, 1.1373] | 1.0684 | 152 / 4401 / 1447 |
| Arizona | 128 | 1.0617 | [1.0555, 1.0684] | 1.0347 | 578 / 618 / 4804 |
| Arizona | 256 | 1.0829 | [1.0744, 1.0919] | 1.0446 | 592 / 779 / 4629 |
| Arizona | 512 | 1.3445 | [1.3107, 1.3816] | 1.2846 | 692 / 894 / 4414 |

## Integrity

- Dual-T4 DataParallel was enforced for training.
- Reused all-domain DAgger states: **No**.
- Correctness mismatches across all cells: **0**.
- Frozen folds: **3/3**.
- Each cell uses the exact paper workload of 6,000 paired queries and the existing baseline trace.
