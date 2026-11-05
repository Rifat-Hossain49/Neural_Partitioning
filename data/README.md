# Data

`arizona_buildings_n1464257_epsg26912.npy` is the Arizona dataset used in the
paper: 1,464,257 OpenStreetMap building rectangles from the Geofabrik Arizona
extract, projected to EPSG:26912. It is a `float32` array of shape
`1464257 x 4` with columns `xmin,xmax,ymin,ymax`. The loader in
`code/realtrain/data.py` reorders and normalizes it. The normalized array's
SHA-256 is `58eb17e00bbed2a46790f0e373e85198cfc4d2bb92a2355afa5b8db8f83ac5c6`,
which `code/scripts/reproduce_locked_cell.py` checks before building a tree.

This data is © OpenStreetMap contributors and is available under the Open
Database License (ODbL) 1.0: <https://opendatacommons.org/licenses/odbl/1-0/>.
Source: <https://www.openstreetmap.org/copyright>, via the Geofabrik extract
cited in the paper.

The Twitter and Crimes datasets are not redistributed. See
[`docs/DATASETS.md`](../docs/DATASETS.md) for their sources and expected hashes.
