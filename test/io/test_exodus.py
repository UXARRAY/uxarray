import os
import numpy as np
import pytest
import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


def test_read_exodus(gridpath):
    """Read an exodus file and writes a exodus file."""
    uxgrid = ux.open_grid(gridpath("exodus", "outCSne8", "outCSne8.g"))
    # Add assertions or checks as needed
    assert uxgrid is not None  # Example assertion

def test_init_verts():
    """Create a uxarray grid from vertices and saves a 1 face exodus file."""
    verts = [[[0, 0], [2, 0], [0, 2], [2, 2]]]
    uxgrid = ux.open_grid(verts)
    # Add assertions or checks as needed
    assert uxgrid is not None  # Example assertion

def test_encode_exodus(gridpath):
    """Read a UGRID dataset and encode that as an Exodus format."""
    uxgrid = ux.open_grid(gridpath("exodus", "outCSne8", "outCSne8.g"))
    exo_ds = uxgrid.to_xarray("Exodus")

    # A uniform quad mesh belongs in exactly one block, typed for a quad
    blocks = [v for v in exo_ds.data_vars if v.startswith("connect")]
    assert blocks == ["connect1"]
    assert exo_ds["connect1"].attrs["elem_type"] == "SHELL4"
    assert exo_ds["connect1"].shape == (uxgrid.n_face, 4)

def test_encode_exodus_mixed_blocks():
    """Faces of different sizes go into separate, correctly typed blocks.

    Exodus element blocks are homogeneous, so a mixed mesh has to be split by
    face size. Getting the fill value wrong collapses everything into one
    max-width block and writes the padding out as a node index.
    """
    face_node_connectivity = np.array([
        [0, 1, 2, 3],
        [1, 4, 2, INT_FILL_VALUE],
        [0, 3, 4, INT_FILL_VALUE],
    ])
    uxgrid = ux.Grid.from_topology(
        node_lon=np.array([0.0, 10.0, 10.0, 0.0, 20.0]),
        node_lat=np.array([0.0, 0.0, 10.0, 10.0, 0.0]),
        face_node_connectivity=face_node_connectivity,
        fill_value=INT_FILL_VALUE,
    )

    exo_ds = uxgrid.to_xarray("Exodus")

    blocks = sorted(v for v in exo_ds.data_vars if v.startswith("connect"))
    assert blocks == ["connect1", "connect2"]

    by_type = {exo_ds[b].attrs["elem_type"]: exo_ds[b] for b in blocks}
    assert set(by_type) == {"TRI", "SHELL4"}
    assert by_type["TRI"].shape == (2, 3)
    assert by_type["SHELL4"].shape == (1, 4)

    # Blocks are written grouped by type, so the original face order has to be
    # recorded or face-centered data silently misaligns on the way back in.
    assert "elem_num_map" in exo_ds
    assert sorted(exo_ds["elem_num_map"].values.tolist()) == [1, 2, 3]

def test_mixed_exodus(gridpath):
    """Read/write an exodus file with two types of faces (triangle and quadrilaterals) and writes a ugrid file."""
    uxgrid = ux.open_grid(gridpath("exodus", "mixed", "mixed.exo"))

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
