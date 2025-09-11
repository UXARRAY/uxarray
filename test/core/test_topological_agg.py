import uxarray as ux
import os

import pytest

from pathlib import Path



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

def test_node_to_face_aggs(gridpath):
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))

    for agg_func in AGGS:
        grid_reduction = getattr(uxds['areaTriangle'], agg_func)(destination='face')

        assert 'n_face' in grid_reduction.dims

def test_node_to_edge_aggs(gridpath):
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))

    for agg_func in AGGS:
        grid_reduction = getattr(uxds['areaTriangle'], agg_func)(destination='edge')

        assert 'n_edge' in grid_reduction.dims
