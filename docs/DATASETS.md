# Dataset preparation

WAHARP's locked evaluation uses three real spatial datasets. The Arizona array
is included in `data/` under the Open Database License. Twitter and Crimes are
not redistributed because their licenses and access terms belong to their
original providers.

| Dataset | Expected source | Representation used by WAHARP |
|---|---|---|
| Twitter | [UCI Twitter Geospatial Data](https://doi.org/10.24432/C5RS5J) (CC BY 4.0) | Longitude/latitude points converted to small rectangles and normalized to `[0,1]^2` |
| Crimes | [Crimes In Chicago (2001 to 2023), Kaggle, version 1 of 29 April 2023](https://www.kaggle.com/datasets/utkarshx27/crimes-2001-to-present) (listed as U.S. Government Works) | Longitude/latitude points converted to small rectangles and normalized to `[0,1]^2` |
| Arizona | [Geofabrik OpenStreetMap US extracts](https://download.geofabrik.de/north-america/us.html) | 1,464,257 building rectangles, projected and normalized to `[0,1]^2` |

The loaders in `code/realtrain/data.py` discover the source files and record the
normalized object-array hash in each dataset manifest. For exact reproduction,
use the same source snapshot and compare that manifest before evaluating. The
evaluated snapshots are recorded in `code/artifacts/workloads/<dataset>/DATASET.json`:

| Dataset | Valid objects | Rejected rows | Normalized object SHA-256 |
|---|---:|---:|---|
| Arizona | 1,464,257 | 0 | `58eb17e00bbed2a46790f0e373e85198cfc4d2bb92a2355afa5b8db8f83ac5c6` |
| Crimes | 7,697,816 | 86,848 | `e229e88e15541ed945211487216c013cd14eebd55178f05749071983b290cd4b` |
| Twitter | 14,262,517 | 0 | `1ba4e74bffee233024d3ba5c400b7008d3e47b2fb32101b477e5a322b47dc3ca` |

`code/scripts/reproduce_locked_cell.py` checks this hash before building a tree.

The Arizona array is included in [`data/`](../data/README.md) under the Open
Database License (© OpenStreetMap contributors). Twitter and Crimes must be
obtained from the sources above.

Training capped Twitter and Crimes at their first 1,000,000 valid rows and
used all Arizona objects. Evaluation used every valid object.

Expected source names are:

```text
twitter.csv
Crimes_-_2001_to_Present.csv
arizona_buildings_n1464257_epsg26912.npy
```

The Arizona cache stores columns as `xmin,xmax,ymin,ymax`; the loader converts
them to the internal `xmin,ymin,xmax,ymax` convention. Twitter and Crimes CSV
column names are detected case-insensitively from common longitude/latitude
aliases.
