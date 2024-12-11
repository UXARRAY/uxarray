import glob
import os
from pathlib import Path

import numpy.testing as nt
import pytest

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

fesom_ugrid_diag_file = current_path / "meshfiles" / "ugrid" / "fesom" / "fesom.mesh.diag.nc"

fesom_ascii_path = current_path / "meshfiles" / "fesom" / "pi"

fesom_netcdf_path = current_path / "meshfiles" / "fesom" / "soufflet-netcdf" / "grid.nc"


@pytest.mark.parametrize("path_or_obj", [fesom_ugrid_diag_file, fesom_ascii_path, fesom_netcdf_path])
def test_open_grid(path_or_obj, ):
    uxgrid = ux.open_grid(path_or_obj)
    uxgrid.validate()


def test_compare_ascii_to_netcdf():
    uxgrid_a = ux.open_grid(fesom_ugrid_diag_file)
    uxgrid_b = ux.open_grid(fesom_ascii_path)

    assert uxgrid_a.n_face == uxgrid_b.n_face
    assert uxgrid_a.n_node == uxgrid_b.n_node

    nt.assert_array_equal(uxgrid_a.face_node_connectivity.values,
                          uxgrid_b.face_node_connectivity.values)


@pytest.mark.parametrize("grid_path", [fesom_ugrid_diag_file, fesom_ascii_path])
def test_open_dataset(grid_path):
    data_path = fesom_ascii_path / "data" / "sst.fesom.1985.nc"
    uxds = ux.open_dataset(grid_path, data_path)

    assert "n_node" in uxds.dims
    assert len(uxds) == 1


@pytest.mark.parametrize("grid_path", [fesom_ugrid_diag_file, fesom_ascii_path])
def test_open_mfdataset(grid_path):
    data_path = glob.glob(str(fesom_ascii_path / "data" / "*.nc"))
    uxds = ux.open_mfdataset(grid_path, data_path)
    assert "n_node" in uxds.dims
    assert "n_face" in uxds.dims
    assert len(uxds) == 3
