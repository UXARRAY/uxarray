import os
import numpy as np

from unittest import TestCase
from pathlib import Path

import xarray as xr
import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestExodus(TestCase):

    def test_read_exodus(self):
        """Read an exodus file and writes a exodus file."""

        exo2_filename = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
        xr_exo_ds = xr.open_dataset(exo2_filename)
        tgrid = ux.Grid(xr_exo_ds)

    def test_init_verts(self):
        """Create a uxarray grid from vertices and saves a 1 face exodus
        file."""
        verts = [[[0, 0], [2, 0], [0, 2], [2, 2]]]
        vgrid = ux.Grid(verts)

    def test_encode_exodus(self):
        """Read a UGRID dataset and encode that as an Exodus format."""

    def test_mixed_exodus(self):
        """Read/write an exodus file with two types of faces (triangle and
        quadrilaterals) and writes a ugrid file."""

        exo2_filename = current_path / "meshfiles" / "exodus" / "mixed" / "mixed.exo"
        xr_exo_ds = xr.open_dataset(exo2_filename)
        tgrid = ux.Grid(xr_exo_ds)
        outfile = current_path / "write_test_mixed.ug"
        tgrid.encode_as("ugrid")
        outfile = current_path / "write_test_mixed.exo"
        tgrid.encode_as("exodus")

    def test_standardized_dtype_and_fill(self):
        """Test to see if Mesh2_Face_Nodes uses the expected integer datatype
        and expected fill value as set in constants.py."""

        exo2_filename = current_path / "meshfiles" / "exodus" / "mixed" / "mixed.exo"
        xr_exo_ds = xr.open_dataset(exo2_filename)
        ux_grid = ux.Grid(xr_exo_ds)

        assert ux_grid.Mesh2_face_nodes.dtype == INT_DTYPE
        assert ux_grid.Mesh2_face_nodes._FillValue == INT_FILL_VALUE
