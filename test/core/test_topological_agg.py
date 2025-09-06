import uxarray as ux
import os

import pytest

from pathlib import Path

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *

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
    uxds = ux.open_dataset(MPAS_OCEAN_MESH, MPAS_OCEAN_MESH)

    for agg_func in AGGS:
        grid_reduction = getattr(uxds['areaTriangle'], agg_func)(destination='face')

        assert 'n_face' in grid_reduction.dims

def test_node_to_edge_aggs():
    uxds = ux.open_dataset(MPAS_OCEAN_MESH, MPAS_OCEAN_MESH)

    for agg_func in AGGS:
        grid_reduction = getattr(uxds['areaTriangle'], agg_func)(destination='edge')

        assert 'n_edge' in grid_reduction.dims
