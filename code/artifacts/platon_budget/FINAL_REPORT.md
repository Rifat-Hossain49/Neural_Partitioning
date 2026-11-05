# WAHARP vs. PLATON MCTS Budget at B=128

Protocol: `waharp_platon_budget_b128_v1_20260907`

WAHARP used the existing frozen model; no training was performed. PLATON used the paper's author implementation with 100 simulation steps. The PLATON-25 cut lists were reused only after protocol, data-hash, and construction-query-hash verification. Evaluation used a fresh held-out 6,000-query workload per dataset, shared by every method.

## Logical node accesses

| Dataset | Rollouts | WAHARP | PLATON | WAHARP / PLATON | W / T / L |
|---|---:|---:|---:|---:|---:|
| arizona | 1 | 571,142 | 630,122 | 0.906399 | 2948 / 1014 / 2038 |
| arizona | 5 | 571,142 | 560,701 | 1.018621 | 1139 / 1142 / 3719 |
| arizona | 10 | 571,142 | 557,585 | 1.024314 | 933 / 1102 / 3965 |
| arizona | 25 | 571,142 | 557,490 | 1.024488 | 875 / 1158 / 3967 |
| crimes | 1 | 383,905 | 380,859 | 1.007998 | 352 / 4266 / 1382 |
| crimes | 5 | 383,905 | 379,344 | 1.012023 | 249 / 4371 / 1380 |
| crimes | 10 | 383,905 | 379,081 | 1.012726 | 244 / 4348 / 1408 |
| crimes | 25 | 383,905 | 378,933 | 1.013121 | 246 / 4321 / 1433 |
| twitter | 1 | 4,364,532 | 4,968,015 | 0.878526 | 3130 / 778 / 2092 |
| twitter | 5 | 4,364,532 | 4,298,786 | 1.015294 | 899 / 783 / 4318 |
| twitter | 10 | 4,364,532 | 4,294,097 | 1.016403 | 764 / 887 / 4349 |
| twitter | 25 | 4,364,532 | 4,292,702 | 1.016733 | 734 / 840 / 4426 |

## Construction time

| Dataset | Method | MCTS s | Native build s | Total s |
|---|---|---:|---:|---:|
| arizona | PLATON-1 | 73.789 | 12.293 | 86.082 |
| arizona | PLATON-10 | 698.893 | 16.715 | 715.608 |
| arizona | PLATON-25 | 1572.816 | 12.327 | 1585.142 |
| arizona | PLATON-5 | 300.697 | 11.950 | 312.647 |
| arizona | WAHARP | - | - | 62.672 |
| crimes | PLATON-1 | 475.599 | 117.611 | 593.210 |
| crimes | PLATON-10 | 3823.460 | 104.717 | 3928.177 |
| crimes | PLATON-25 | 8487.503 | 112.379 | 8599.882 |
| crimes | PLATON-5 | 1887.108 | 110.790 | 1997.898 |
| crimes | WAHARP | - | - | 306.615 |
| twitter | PLATON-1 | 879.657 | 254.924 | 1134.580 |
| twitter | PLATON-10 | 6978.556 | 257.983 | 7236.538 |
| twitter | PLATON-25 | 15684.028 | 247.488 | 15931.516 |
| twitter | PLATON-5 | 3427.561 | 258.787 | 3686.348 |
| twitter | WAHARP | - | - | 662.145 |

## Correctness

Total semantic mismatches across completed comparisons: **0**.
Range and point queries require exact result count, ID sum, and ID XOR. kNN queries require equivalent kth distance because the native engine may return additional boundary ties.

Detailed query-family and correctness results are in `QUERY_FAMILY_RESULTS.csv` and `CORRECTNESS.csv`.
