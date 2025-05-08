import os
import numpy as np
from pathlib import Path
import pytest
import xarray as xr
import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

exo_filename = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
exo2_filename = current_path / "meshfiles" / "exodus" / "mixed" / "mixed.exo"



def test_read_exodus():
    """Read an Exodus file and verify basic UGRID properties."""
    grid = ux.open_grid(exo_filename)

    # must have at least one face and one node
    assert grid.n_face > 0
    assert grid.n_node > 0

    # connectivity shape matches n_face
    conn = grid.face_node_connectivity.values
    assert conn.ndim == 2
    assert conn.shape[0] == grid.n_face

    # node coordinates must exist
    assert "node_lon" in grid._ds
    assert "node_lat" in grid._ds


def test_encode_exodus():
    """Take a tiny triangular grid → encode_as('Exodus') → inspect output Dataset."""
    verts = [[[0, 0], [1, 0], [0, 1]]]
    grid = ux.open_grid(verts)

    exo_ds = grid.encode_as("Exodus")
    assert isinstance(exo_ds, xr.Dataset)

    # must have the num_elem dimension matching grid.n_face
    assert "num_elem" in exo_ds.dims
    assert exo_ds.dims["num_elem"] == grid.n_face

    # metadata attributes should be present
    for attr in ("version", "api_version", "floating_point_word_size"):
        assert attr in exo_ds.attrs


def test_mixed_exodus():
    """Read a mixed‐element Exodus → encode_as both 'UGRID' and 'Exodus' → inspect."""
    grid = ux.open_grid(exo2_filename)

    # UGRID encoding
    ds_ugrid = grid.encode_as("UGRID")
    assert isinstance(ds_ugrid, xr.Dataset)
    assert "face_node_connectivity" in ds_ugrid

    # Exodus encoding
    ds_exo = grid.encode_as("Exodus")
    assert isinstance(ds_exo, xr.Dataset)
    assert "num_elem" in ds_exo.dims
    for attr in ("version", "api_version"):
        assert attr in ds_exo.attrs


def test_standardized_dtype_and_fill():
    """Check that face_node_connectivity uses INT_DTYPE and INT_FILL_VALUE."""
    grid = ux.open_grid(exo2_filename)
    conn = grid.face_node_connectivity

    assert conn.dtype == INT_DTYPE
    fill = conn.attrs.get("_FillValue", None)
    assert fill == INT_FILL_VALUE
