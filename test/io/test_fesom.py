import glob
import os
from pathlib import Path

import numpy.testing as nt
import pytest

import uxarray as ux




def test_open_grid_mesh_diag(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "fesom", "fesom.mesh.diag.nc"))
    uxgrid.validate()

def test_open_grid_pi_path(test_data_dir):
    uxgrid = ux.open_grid(test_data_dir / "fesom" / "pi")
    uxgrid.validate()

def test_open_grid_soufflet_netcdf(gridpath):
    uxgrid = ux.open_grid(gridpath("fesom", "soufflet-netcdf", "grid.nc"))
    uxgrid.validate()


def test_compare_ascii_to_netcdf(gridpath, test_data_dir):
    uxgrid_a = ux.open_grid(gridpath("ugrid", "fesom", "fesom.mesh.diag.nc"))
    uxgrid_b = ux.open_grid(test_data_dir / "fesom" / "pi")

    assert uxgrid_a.n_face == uxgrid_b.n_face
    assert uxgrid_a.n_node == uxgrid_b.n_node

    nt.assert_array_equal(uxgrid_a.face_node_connectivity.values,
                          uxgrid_b.face_node_connectivity.values)


def test_open_dataset_mesh_diag(gridpath, test_data_dir):
    data_path = test_data_dir / "fesom" / "pi" / "data" / "sst.fesom.1985.nc"
    uxds = ux.open_dataset(gridpath("ugrid", "fesom", "fesom.mesh.diag.nc"), data_path)

    assert "n_node" in uxds.dims
    assert len(uxds) == 1

def test_open_dataset_pi_path(test_data_dir):
    grid_path = test_data_dir / "fesom" / "pi"
    data_path = test_data_dir / "fesom" / "pi" / "data" / "sst.fesom.1985.nc"
    uxds = ux.open_dataset(grid_path, data_path)

    assert "n_node" in uxds.dims
    assert len(uxds) == 1


def test_open_mfdataset_mesh_diag(gridpath, test_data_dir):
    data_path = glob.glob(str(test_data_dir / "fesom" / "pi" / "data" / "*.nc"))
    uxds = ux.open_mfdataset(gridpath("ugrid", "fesom", "fesom.mesh.diag.nc"), data_path)
    assert "n_node" in uxds.dims
    assert "n_face" in uxds.dims
    assert len(uxds) == 3

def test_open_mfdataset_pi_path(test_data_dir):
    grid_path = test_data_dir / "fesom" / "pi"
    data_path = glob.glob(str(test_data_dir / "fesom" / "pi" / "data" / "*.nc"))
    uxds = ux.open_mfdataset(grid_path, data_path)
    assert "n_node" in uxds.dims
    assert "n_face" in uxds.dims
    assert len(uxds) == 3
