import glob
import os
from pathlib import Path

import numpy.testing as nt
import pytest

import uxarray as ux

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *




@pytest.mark.parametrize("path_or_obj", [FESOM_MESH_DIAG, FESOM_PI_PATH, FESOM_SOUFFLET_NETCDF_GRID])
def test_open_grid(path_or_obj, ):
    uxgrid = ux.open_grid(path_or_obj)
    uxgrid.validate()


def test_compare_ascii_to_netcdf():
    uxgrid_a = ux.open_grid(FESOM_MESH_DIAG)
    uxgrid_b = ux.open_grid(FESOM_PI_PATH)

    assert uxgrid_a.n_face == uxgrid_b.n_face
    assert uxgrid_a.n_node == uxgrid_b.n_node

    nt.assert_array_equal(uxgrid_a.face_node_connectivity.values,
                          uxgrid_b.face_node_connectivity.values)


@pytest.mark.parametrize("grid_path", [FESOM_MESH_DIAG, FESOM_PI_PATH])
def test_open_dataset(grid_path):
    data_path = FESOM_PI_PATH / "data" / "sst.fesom.1985.nc"
    uxds = ux.open_dataset(grid_path, data_path)

    assert "n_node" in uxds.dims
    assert len(uxds) == 1


@pytest.mark.parametrize("grid_path", [FESOM_MESH_DIAG, FESOM_PI_PATH])
def test_open_mfdataset(grid_path):
    data_path = glob.glob(str(FESOM_PI_PATH / "data" / "*.nc"))
    uxds = ux.open_mfdataset(grid_path, data_path)
    assert "n_node" in uxds.dims
    assert "n_face" in uxds.dims
    assert len(uxds) == 3
