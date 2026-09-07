# WAHARP 0.75/0.25 Final Locked Evaluation

Training protocol: `waharp_loss075_final_train_v1_20260903`
Evaluation protocol: `waharp_loss075_locked_eval_v1_20260903`
Selected member: `2`
Checkpoint SHA-256: `c158a7e365dc5fbad01de6126e8665a99fc942f13e362d359cb49be301f6b4f7`

Training and member selection read no final queries. The final pass reused all baseline per-query outputs and rebuilt only the new neural trees.

## New vs. previous neural model

| Dataset | Capacity | New/old | 95% CI | Change | New accesses | Old accesses |
|---|---:|---:|---:|---:|---:|---:|
| Arizona | 128 | 0.986966 | [0.985052, 0.988740] | -1.30% | 535,047 | 542,113 |
| Arizona | 256 | 0.980103 | [0.976924, 0.982859] | -1.99% | 288,953 | 294,819 |
| Arizona | 512 | 0.956588 | [0.951587, 0.961203] | -4.34% | 162,026 | 169,379 |
| Crimes | 128 | 1.001647 | [0.999060, 1.004234] | +0.16% | 295,544 | 295,058 |
| Crimes | 256 | 1.002198 | [0.999598, 1.005447] | +0.22% | 155,910 | 155,568 |
| Crimes | 512 | 0.990919 | [0.987338, 0.993653] | -0.91% | 86,970 | 87,767 |
| Twitter | 128 | 1.010298 | [1.009557, 1.011056] | +1.03% | 4,303,335 | 4,259,472 |
| Twitter | 256 | 1.008213 | [1.007132, 1.009249] | +0.82% | 2,186,312 | 2,168,502 |
| Twitter | 512 | 1.007266 | [1.005889, 1.008608] | +0.73% | 1,128,627 | 1,120,486 |

## Aggregate controls

| Control | Micro new/control | Cell-geomean new/control | Micro change | Cell-geomean change |
|---|---:|---:|---:|---:|
| OldNeural_060_040 | 1.005450 | 0.993663 | +0.55% | -0.63% |
| PLATON | 1.019954 | 1.025574 | +2.00% | +2.56% |
| STR | 0.940325 | 0.905106 | -5.97% | -9.49% |
| TGS | 0.683543 | 0.787856 | -31.65% | -21.21% |

## Gates

- Nine dataset-capacity cells completed.
- Each cell used 6,000 locked final queries: 4,200 range, 300 point, and 1,500 kNN.
- Correctness mismatches: 0.
- Baseline reconstructions: 0.
- The same frozen checkpoint was used in every cell.
