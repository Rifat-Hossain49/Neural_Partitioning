# Dataset preparation

WAHARP's locked evaluation uses three real spatial datasets. They are not
redistributed because their licenses and access terms belong to their original
providers.

| Dataset | Expected source | Representation used by WAHARP |
|---|---|---|
| Twitter | [UCI Twitter Geospatial Data](https://archive.ics.uci.edu/dataset/1050/twitter+geospatial+data) | Longitude/latitude points converted to small rectangles and normalized to `[0,1]^2` |
| Crimes | [Chicago Crimes, 2001 to Present](https://www.kaggle.com/datasets/utkarshx27/crimes-2001-to-present) | Longitude/latitude points converted to small rectangles and normalized to `[0,1]^2` |
| Arizona | [Geofabrik OpenStreetMap US extracts](https://download.geofabrik.de/north-america/us.html) | 1,464,257 building rectangles, projected and normalized to `[0,1]^2` |

The loaders in `realtrain/data.py` discover the source files and record the
normalized object-array hash in each dataset manifest. For exact reproduction,
use the same source snapshot and compare that manifest before evaluating.

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
