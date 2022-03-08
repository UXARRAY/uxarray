import os
import numpy as np
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestGrid(TestCase):

    def test_saveas(self):
        """Rename a file."""

        exo_filename = current_path / "meshfiles" / "outCSne8.g"
        # grid object expects a string argument for loading a file
        tgrid = ux.Grid(str(exo_filename))

        new_filename = "new_outCSne8.g"
        new_filepath = current_path / "meshfiles" / new_filename
        tgrid.saveas_file(new_filepath)

        assert (tgrid.filepath == str(new_filepath))

    def test_read_exodus(self):
        """Read an exodus file and writes a ugrid file."""

        exo2_filename = current_path / "meshfiles" / "outCSne8.g"
        tgrid = ux.Grid(str(exo2_filename))
        outfile = current_path / "write_test_outCSne8.ug"
        tgrid.write(str(outfile))

    def test_write_exodus(self):
        """Read a ugrid file and write exodus."""

        filename = current_path / "meshfiles" / "outCSne8.g"
        tgrid = ux.Grid(str(filename))
        outfile = current_path / "ouCSne8_uxarray.exo"
        tgrid.write(str(outfile), ".exo")

    def test_read_scrip(self):
        """Reads a scrip file and write ugrid file."""

        scrip_filename = current_path / "meshfiles" / "outCSne8.nc"
        tgrid = ux.Grid(str(scrip_filename))

    def test_read_ugrid(self):
        """Reads a ugrid file."""

        ug_filename1 = current_path / "meshfiles" / "outCSne30.ug"
        ug_filename2 = current_path / "meshfiles" / "outRLL1deg.ug"
        ug_filename3 = current_path / "meshfiles" / "ov_RLL10deg_CSne4.ug"

        tgrid1 = ux.Grid(str(ug_filename1))
        tgrid2 = ux.Grid(str(ug_filename2))
        tgrid3 = ux.Grid(str(ug_filename3))

    # use external package to read?
    # https://gis.stackexchange.com/questions/113799/how-to-read-a-shapefile-in-python
    def test_read_shpfile(self):
        """Reads a shape file and write ugrid file."""

        shp_filename = current_path / "meshfiles" / "grid_fire.shp"
        tgrid = ux.Grid(str(shp_filename))

    def test_init_verts(self):
        """Create a uxarray grid from vertices and saves an exodus file."""
        verts = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        vgrid = ux.Grid(verts)

        face_filename = current_path / "meshfiles" / "1face.g"
        vgrid.write(face_filename)

    def test_mixed_exodus(self):
        """Read/write an exodus file with two types of faces (triangle and
        quadrilaterals) and writes a ugrid file."""

        exo2_filename = current_path / "meshfiles" / "mixed.exo"
        tgrid = ux.Grid(str(exo2_filename))
        outfile = current_path / "write_test_mixed.ug"
        tgrid.write(str(outfile))
        outfile = current_path / "write_test_mixed.exo"
        tgrid.write(str(outfile))
