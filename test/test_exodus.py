import os
import numpy as np
from pathlib import Path
import pytest
import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_exo_ne8 = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
gridfile_exo_mixed = current_path / "meshfiles" / "exodus" / "mixed" / "mixed.exo"
gridfile_ugrid_csne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"


def test_read_exodus():
    """Read an exodus file and writes a exodus file."""
    uxgrid = ux.open_grid(gridfile_exo_ne8)
    assert uxgrid.n_node == 386
    assert uxgrid.n_face == 384


def test_init_verts():
    """Create a uxarray grid from vertices and saves a 1 face exodus file."""
    verts = [[[0, 0], [2, 0], [0, 2], [2, 2]]]
    uxgrid = ux.open_grid(verts)
    assert uxgrid.n_node == 4
    assert uxgrid.n_face == 1


def test_encode_exodus(tmp_path):
    """Read a UGRID dataset and encode that as an Exodus format."""
    uxgrid = ux.open_grid(gridfile_ugrid_csne30)  # A ugrid file

    outfile = tmp_path / "test.exo"

    exodus_ds = uxgrid.to_xarray("Exodus")
    exodus_ds.to_netcdf(outfile)

    reopened_grid = ux.open_grid(outfile)
    reopened_grid.validate()


def test_mixed_exodus(tmp_path):
    """Read/write an exodus file with two types of faces (triangle and
    quadrilaterals) and writes a ugrid file."""
    uxgrid = ux.open_grid(gridfile_exo_mixed)

    outfile_ugrid = tmp_path / "mixed.nc"
    ugrid_ds = uxgrid.to_xarray("UGRID")
    ugrid_ds.to_netcdf(outfile_ugrid)
    reopened_ugrid = ux.open_grid(outfile_ugrid)
    reopened_ugrid.validate()

    outfile_exodus = tmp_path / "mixed.exo"
    exodus_ds = uxgrid.to_xarray("Exodus")
    exodus_ds.to_netcdf(outfile_exodus)
    reopened_exodus = ux.open_grid(outfile_exodus)
    reopened_exodus.validate()


def test_standardized_dtype_and_fill():
    """Test to see if Mesh2_Face_Nodes uses the expected integer datatype and expected fill value as set in constants.py."""
    uxgrid = ux.open_grid(gridfile_exo_mixed)

    assert uxgrid.face_node_connectivity.dtype == INT_DTYPE
    assert uxgrid.face_node_connectivity._FillValue == INT_FILL_VALUE
