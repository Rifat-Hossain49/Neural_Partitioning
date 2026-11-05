# Third-party notices

The vendored `code/rtreelib/` package is derived from the open-source `RTree`
project by Sergey Krilov and is distributed under the MIT License. Its original
copyright and permission notice are retained in `code/rtreelib/LICENSE`.

Dataset licenses and terms remain with their respective providers.

`data/arizona_buildings_n1464257_epsg26912.npy` is derived from OpenStreetMap
building data (Geofabrik Arizona extract). It is © OpenStreetMap contributors
and is made available under the Open Database License (ODbL) 1.0:
<https://opendatacommons.org/licenses/odbl/1-0/>. See
<https://www.openstreetmap.org/copyright>.

The Twitter and Chicago Crimes datasets are not redistributed. The query
workloads in `code/artifacts/` contain small samples of each dataset: the point
and kNN query locations are object centers drawn from the dataset and
normalized to [0, 1] (a few hundred to about two thousand per dataset). These
samples remain under the corresponding dataset's terms:

- Twitter: N. E. Helwig, Y. Gao, S. Wang, and P. Ma, "Twitter Geospatial
  Data," UCI Machine Learning Repository, 2015,
  <https://doi.org/10.24432/C5RS5J>, licensed under CC BY 4.0
  (<https://creativecommons.org/licenses/by/4.0/>).
- Crimes: U. Singh, "Crimes in Chicago (2001 to 2023)," Kaggle, 2023,
  <https://www.kaggle.com/datasets/utkarshx27/crimes-2001-to-present>, from
  the City of Chicago crime data; listed on Kaggle under "U.S. Government
  Works".
