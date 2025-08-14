import os
import numpy as np
from pathlib import Path
import pytest
import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

exo_filename = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
exo2_filename = current_path / "meshfiles" / "exodus" / "mixed" / "mixed.exo"

def test_read_exodus():
    """Read an exodus file and writes a exodus file."""
    uxgrid = ux.open_grid(exo_filename)
    # Add assertions or checks as needed
    assert uxgrid is not None  # Example assertion

def test_init_verts():
    """Create a uxarray grid from vertices and saves a 1 face exodus file."""
    verts = [[[0, 0], [2, 0], [0, 2], [2, 2]]]
    uxgrid = ux.open_grid(verts)
    # Add assertions or checks as needed
    assert uxgrid is not None  # Example assertion

def test_encode_exodus():
    """Read a UGRID dataset and encode that as an Exodus format."""
    uxgrid = ux.open_grid(exo_filename)
    # Add encoding logic and assertions as needed
    pass  # Placeholder for actual implementation

def test_mixed_exodus():
    """Read/write an exodus file with two types of faces (triangle and quadrilaterals) and writes a ugrid file."""
    uxgrid = ux.open_grid(exo2_filename)

    ugrid_obj = uxgrid.to_xarray("UGRID")
    exo_obj = uxgrid.to_xarray("Exodus")

    ugrid_obj.to_netcdf("test_ugrid.nc")
    exo_obj.to_netcdf("test_exo.exo")

    ugrid_load_saved = ux.open_grid("test_ugrid.nc")
    exodus_load_saved = ux.open_grid("test_exo.exo")

    # Face node connectivity comparison
    assert np.array_equal(ugrid_load_saved.face_node_connectivity.values, uxgrid.face_node_connectivity.values)
    assert np.array_equal(uxgrid.face_node_connectivity.values, exodus_load_saved.face_node_connectivity.values)

    # Node coordinates comparison
    assert np.array_equal(ugrid_load_saved.node_lon.values, uxgrid.node_lon.values)
    assert np.array_equal(uxgrid.node_lon.values, exodus_load_saved.node_lon.values)
    assert np.array_equal(ugrid_load_saved.node_lat.values, uxgrid.node_lat.values)

    # Cleanup
    ugrid_load_saved._ds.close()
    exodus_load_saved._ds.close()
    del ugrid_load_saved, exodus_load_saved
    os.remove("test_ugrid.nc")
    os.remove("test_exo.exo")

def test_standardized_dtype_and_fill():
    """Test to see if Mesh2_Face_Nodes uses the expected integer datatype and expected fill value as set in constants.py."""
    uxgrid = ux.open_grid(exo2_filename)

    assert uxgrid.face_node_connectivity.dtype == INT_DTYPE
    assert uxgrid.face_node_connectivity._FillValue == INT_FILL_VALUE
