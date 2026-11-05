# WAHARP analysis appendix

Companion to the WAHARP paper. Every table below is regenerated from committed files under `artifacts/` by `scripts/make_analysis_appendix.py`; the CSV data behind each paper figure is written to `artifacts/paper_figures/`. The evaluated model is the frozen checkpoint `artifacts/model/selected.pt` (0.75/0.25 loss weights, member 2). Ratios are WAHARP / baseline logical node accesses over 6,000 paired final queries; values below 1 favor WAHARP. Confidence intervals are pointwise paired-bootstrap 95% intervals over queries (2,000 draws) and describe fixed trees and workloads, not training-seed variability.

## 1. Aggregate node-access ratios (Section 7.2, Figure 2)

| Dataset | B | vs. PLATON [95% CI] | vs. STR [95% CI] | vs. TGS [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Arizona | 128 | 1.026083 [1.0230, 1.0297] | 0.857685 [0.8453, 0.8688] | 0.890839 [0.8786, 0.9012] |
| Arizona | 256 | 1.036651 [1.0324, 1.0413] | 0.821886 [0.8069, 0.8351] | 0.785914 [0.7664, 0.8023] |
| Arizona | 512 | 1.046619 [1.0417, 1.0522] | 0.789816 [0.7747, 0.8034] | 0.703155 [0.6828, 0.7222] |
| Crimes | 128 | 1.019015 [1.0134, 1.0323] | 0.973004 [0.9411, 0.9855] | 1.019789 [1.0143, 1.0323] |
| Crimes | 256 | 1.028010 [1.0197, 1.0489] | 0.959594 [0.9216, 0.9769] | 1.024241 [1.0157, 1.0445] |
| Crimes | 512 | 1.017395 [1.0087, 1.0329] | 0.920055 [0.8678, 0.9454] | 1.000955 [0.9906, 1.0167] |
| Twitter | 128 | 1.017604 [1.0164, 1.0190] | 0.962999 [0.9609, 0.9650] | 0.727856 [0.7162, 0.7386] |
| Twitter | 256 | 1.018066 [1.0166, 1.0195] | 0.949823 [0.9473, 0.9522] | 0.636735 [0.6235, 0.6493] |
| Twitter | 512 | 1.021116 [1.0195, 1.0228] | 0.931852 [0.9285, 0.9350] | 0.490301 [0.4768, 0.5052] |
| **Geometric mean** |  | **1.025574** | **0.905106** | **0.787856** |

Geometric-mean change: +2.56% vs. PLATON, -9.49% vs. STR, -21.21% vs. TGS. A CI that contains 1 means that comparison is inconclusive at the query level.

| Baseline ratio | Geometric mean over nine cells |
| --- | ---: |
| STR / PLATON | 1.133097 |
| TGS / PLATON | 1.301728 |

## 2. Capacity transfer (Section 7.7)

The model was trained at B = 128 only; B = 256 and 512 reuse the same weights.

| B | WAHARP/PLATON | WAHARP/STR | WAHARP/TGS |
| --- | ---: | ---: | ---: |
| 128 | 1.020894 | 0.929728 | 0.871201 |
| 256 | 1.027548 | 0.908198 | 0.800286 |
| 512 | 1.028295 | 0.878137 | 0.701417 |

## 3. Construction time (Section 7.3, Figure 3)

Seconds of method-specific construction only: WAHARP = policy inference plus Python tree assembly; PLATON = summed MCTS decision time plus native bulk loading. Dataset loading, final queries, and idle resumable-run time are excluded. Because the records do not establish identical hardware, timing ratios retain their method-specific definition.

| Dataset | B | WAHARP | STR | TGS | PLATON MCTS | PLATON native | PLATON total | MCTS decisions | MCTS share | PLATON/WAHARP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arizona | 128 | 69.208 | 10.153 | 23.307 | 1,572.816 | 13.427 | 1,586.243 | 11,439 | 99.2% | 22.92x |
| Arizona | 256 | 42.129 | 9.659 | 18.980 | 935.019 | 13.109 | 948.127 | 5,719 | 98.6% | 22.51x |
| Arizona | 512 | 29.614 | 9.991 | 18.444 | 602.914 | 11.698 | 614.612 | 2,859 | 98.1% | 20.75x |
| Crimes | 128 | 350.132 | 49.492 | 115.075 | 8,487.503 | 108.481 | 8,595.985 | 60,139 | 98.7% | 24.55x |
| Crimes | 256 | 224.231 | 54.339 | 113.006 | 7,031.731 | 115.792 | 7,147.523 | 30,069 | 98.4% | 31.88x |
| Crimes | 512 | 159.326 | 52.081 | 113.321 | 4,438.264 | 108.006 | 4,546.270 | 15,034 | 97.6% | 28.53x |
| Twitter | 128 | 694.233 | 96.440 | 244.714 | 15,684.028 | 254.943 | 15,938.971 | 111,425 | 98.4% | 22.96x |
| Twitter | 256 | 441.316 | 98.901 | 254.321 | 12,568.095 | 270.976 | 12,839.071 | 55,712 | 97.9% | 29.09x |
| Twitter | 512 | 332.384 | 94.093 | 208.818 | 8,053.600 | 263.920 | 8,317.520 | 27,856 | 96.8% | 25.02x |

PLATON/WAHARP: 20.75-31.88x (arithmetic mean 25.36x). MCTS share of PLATON time: 96.8-99.2%.

### PLATON search-budget sensitivity (Section 7.3)

B = 128, 100 simulation steps, frozen WAHARP, and one fresh 6,000-query workload per dataset (`artifacts/platon_budget/`). Zero correctness mismatches.

| Dataset | Rollouts | WAHARP acc. | PLATON acc. | WAHARP/PLATON | WAHARP s | PLATON s | PLATON/WAHARP time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arizona | 1 | 571,142 | 630,122 | 0.906399 | 62.7 | 86.1 | 1.37x |
| Arizona | 5 | 571,142 | 560,701 | 1.018621 | 62.7 | 312.6 | 4.99x |
| Arizona | 10 | 571,142 | 557,585 | 1.024314 | 62.7 | 715.6 | 11.42x |
| Arizona | 25 | 571,142 | 557,490 | 1.024488 | 62.7 | 1,585.1 | 25.29x |
| Crimes | 1 | 383,905 | 380,859 | 1.007998 | 306.6 | 593.2 | 1.93x |
| Crimes | 5 | 383,905 | 379,344 | 1.012023 | 306.6 | 1,997.9 | 6.52x |
| Crimes | 10 | 383,905 | 379,081 | 1.012726 | 306.6 | 3,928.2 | 12.81x |
| Crimes | 25 | 383,905 | 378,933 | 1.013121 | 306.6 | 8,599.9 | 28.05x |
| Twitter | 1 | 4,364,532 | 4,968,015 | 0.878526 | 662.1 | 1,134.6 | 1.71x |
| Twitter | 5 | 4,364,532 | 4,298,786 | 1.015294 | 662.1 | 3,686.3 | 5.57x |
| Twitter | 10 | 4,364,532 | 4,294,097 | 1.016403 | 662.1 | 7,236.5 | 10.93x |
| Twitter | 25 | 4,364,532 | 4,292,702 | 1.016733 | 662.1 | 15,931.5 | 24.06x |

## 4. Query families (Section 7.4, Figure 4)

WAHARP/PLATON access ratio by query family (range 4,200, point 300, kNN 1,500 queries per cell).

| Dataset | Family | B=128 | B=256 | B=512 |
| --- | ---: | ---: | ---: | ---: |
| Arizona | range | 1.0227 | 1.0321 | 1.0425 |
| Arizona | point | 1.3793 | 1.3069 | 1.1159 |
| Arizona | knn | 1.1743 | 1.1674 | 1.1347 |
| Crimes | range | 1.0064 | 1.0063 | 0.9996 |
| Crimes | point | 1.2896 | 1.4769 | 1.2457 |
| Crimes | knn | 1.2369 | 1.3101 | 1.1781 |
| Twitter | range | 1.0166 | 1.0170 | 1.0200 |
| Twitter | point | 1.2005 | 1.1859 | 1.1405 |
| Twitter | knn | 1.2604 | 1.2200 | 1.1613 |

## 5. Paired query outcomes vs. PLATON (Section 7.5, Figure 5)

| Dataset | B | WAHARP fewer | Tie | WAHARP more | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arizona | 128 | 14.77% | 17.87% | 67.37% | 0 |
| Arizona | 256 | 18.00% | 21.38% | 60.62% | 0 |
| Arizona | 512 | 19.83% | 28.33% | 51.83% | 0 |
| Crimes | 128 | 3.58% | 72.58% | 23.83% | 0 |
| Crimes | 256 | 4.18% | 75.08% | 20.73% | 0 |
| Crimes | 512 | 3.78% | 77.08% | 19.13% | 0 |
| Twitter | 128 | 11.67% | 13.58% | 74.75% | 0 |
| Twitter | 256 | 13.95% | 20.43% | 65.62% | 0 |
| Twitter | 512 | 16.60% | 25.20% | 58.20% | 0 |

## 6. Comparison with the previous checkpoint (Section 7.6)

Selected 0.75/0.25 checkpoint vs. the earlier 0.60/0.40 checkpoint on the same locked queries. Hyperparameters were fixed before the locked test, so the selected checkpoint was retained regardless of these outcomes.

| Dataset | B | New/old | 95% CI | New accesses | Old accesses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arizona | 128 | 0.986966 | [0.985052, 0.988740] | 535,047 | 542,113 |
| Arizona | 256 | 0.980103 | [0.976924, 0.982859] | 288,953 | 294,819 |
| Arizona | 512 | 0.956588 | [0.951587, 0.961203] | 162,026 | 169,379 |
| Crimes | 128 | 1.001647 | [0.999060, 1.004234] | 295,544 | 295,058 |
| Crimes | 256 | 1.002198 | [0.999598, 1.005447] | 155,910 | 155,568 |
| Crimes | 512 | 0.990919 | [0.987338, 0.993653] | 86,970 | 87,767 |
| Twitter | 128 | 1.010298 | [1.009557, 1.011056] | 4,303,335 | 4,259,472 |
| Twitter | 256 | 1.008213 | [1.007132, 1.009249] | 2,186,312 | 2,168,502 |
| Twitter | 512 | 1.007266 | [1.005889, 1.008608] | 1,128,627 | 1,120,486 |

Cell geometric mean 0.993663; pooled ratio 1.005450.

## 7. Neural action usage (Section 7.8, Figure 6)

Share of recorded split decisions at object and parent levels. Action id a maps to angle index j = a // 5 (j * 11.25 degrees) and page fraction index k = a mod 5. Decision counts for all 16 angles and 5 fractions are in `paper_figures/fig6_action_usage.csv`.

| Setting | Axis-aligned (0/90 deg) | Other angles | f=0.25 | f=0.375 | f=0.5 | f=0.625 | f=0.75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AZ 128 | 96.54% | 3.46% | 84.1% | 3.0% | 4.5% | 2.5% | 6.0% |
| AZ 256 | 94.84% | 5.16% | 80.4% | 2.9% | 6.2% | 3.3% | 7.2% |
| AZ 512 | 92.18% | 7.82% | 72.0% | 3.8% | 8.8% | 4.3% | 11.1% |
| CR 128 | 92.17% | 7.83% | 82.2% | 8.1% | 3.0% | 2.0% | 4.6% |
| CR 256 | 92.67% | 7.33% | 79.6% | 8.3% | 3.9% | 2.7% | 5.4% |
| CR 512 | 93.88% | 6.12% | 78.6% | 7.7% | 4.5% | 3.1% | 6.0% |
| TW 128 | 96.75% | 3.25% | 82.2% | 6.6% | 2.5% | 2.6% | 6.1% |
| TW 256 | 96.58% | 3.42% | 78.5% | 7.1% | 3.2% | 3.4% | 7.9% |
| TW 512 | 96.04% | 3.96% | 74.2% | 6.8% | 4.6% | 4.5% | 9.9% |

## 8. Tree structure (Section 7.9, Table 9)

| Dataset | B | Height | Leaves | Internal | Decisions | Distinct actions | Mean fill | Sibling overlap burden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arizona | 128 | 3 | 11,440 | 91 | 11,528 | 37 | 0.999877 | 0.051959 |
| Arizona | 256 | 3 | 5,720 | 24 | 5,741 | 38 | 0.999684 | 0.053171 |
| Arizona | 512 | 3 | 2,860 | 7 | 2,864 | 34 | 0.999468 | 0.076188 |
| Crimes | 128 | 4 | 60,140 | 475 | 60,611 | 65 | 0.999963 | 0.028956 |
| Crimes | 256 | 3 | 30,070 | 119 | 30,186 | 59 | 0.999951 | 0.029082 |
| Crimes | 512 | 3 | 15,035 | 31 | 15,063 | 55 | 0.999882 | 0.025627 |
| Twitter | 128 | 4 | 111,426 | 879 | 112,301 | 47 | 0.999985 | 18.197437 |
| Twitter | 256 | 3 | 55,713 | 219 | 55,929 | 45 | 0.999990 | 0.013659 |
| Twitter | 512 | 3 | 27,857 | 56 | 27,910 | 44 | 0.999928 | 0.024771 |

## 9. Leave-one-dataset-out transfer (Section 7.10, Table 10)

| Comparison | Cell geometric mean | Pooled | Cells below 1 |
| --- | ---: | ---: | ---: |
| LODO / PLATON | 1.1105 | 1.0416 | 0 of 9 |
| LODO / WAHARP_ALL_DOMAIN | 1.0828 | 1.0212 | 0 of 9 |
| LODO / STR | 0.9800 | 0.9603 | 6 of 9 |
| LODO / TGS | 0.8531 | 0.6980 | 6 of 9 |
| LODO / PLATON, range | 1.0586 |  |  |
| LODO / PLATON, point | 2.1444 |  |  |
| LODO / PLATON, knn | 1.9019 |  |  |

Per-cell ratios, intervals, and correctness are in `artifacts/generalization/`.

## Reproducing one cell

`scripts/reproduce_locked_cell.py` rebuilds any dataset-capacity cell from the frozen checkpoint and the committed queries in `artifacts/workloads/`, then compares total and per-workload accesses, policy decisions, action counts, and tree structure with these records.
