import os
import numpy as np
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestExodus(TestCase):

    def test_read_exodus(self):
        """Read an exodus file and writes a exodus file."""

        exo2_filename = current_path / "meshfiles" / "outCSne8.g"
        tgrid = ux.open_dataset(str(exo2_filename))
        outfile = current_path / "write_test_outCSne8.g"
        tgrid.write(str(outfile))

    def test_init_verts(self):
        """Create a uxarray grid from vertices and saves a 1 face exodus
        file."""
        verts = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        vgrid = ux.Grid(verts)

        face_filename = current_path / "meshfiles" / "1face.g"
        vgrid.write(face_filename)

    def test_mixed_exodus(self):
        """Read/write an exodus file with two types of faces (triangle and
        quadrilaterals) and writes a ugrid file."""

        exo2_filename = current_path / "meshfiles" / "mixed.exo"
        tgrid = ux.open_dataset(str(exo2_filename))
        outfile = current_path / "write_test_mixed.ug"
        tgrid.write(str(outfile))
        outfile = current_path / "write_test_mixed.exo"
        tgrid.write(str(outfile))
