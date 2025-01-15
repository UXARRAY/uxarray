import uxarray as ux
import os

import pytest

from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

ds_path = current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc'

AGGS = ["topological_mean",
        "topological_max",
        "topological_min",
        "topological_prod",
        "topological_sum",
        "topological_std",
        "topological_std",
        "topological_var",
        "topological_median",
        "topological_all",
        "topological_any"]


def test_node_to_face_aggs():
    uxds = ux.open_dataset(ds_path, ds_path)

    for agg_func in AGGS:
        grid_reduction = getattr(uxds['areaTriangle'], agg_func)(destination='face')

        assert 'n_face' in grid_reduction.dims


def test_node_to_edge_aggs():
    uxds = ux.open_dataset(ds_path, ds_path)

    for agg_func in AGGS:
        grid_reduction = getattr(uxds['areaTriangle'], agg_func)(destination='edge')

        assert 'n_edge' in grid_reduction.dims


def test_edge_to_face_aggs():
    grid_path = '/Users/aaronzedwick/uxarray/test/meshfiles/mpas/QU/mesh.QU.1920km.151026.nc'

    uxds = ux.open_dataset(grid_path, grid_path)

    uxds = uxds['cellsOnEdge'].subset.nearest_neighbor(k=3, center_coord=[0, 0])

    uxda_edge_face_agg = uxds.topological_mean(destination="node")

    print(uxda_edge_face_agg)
