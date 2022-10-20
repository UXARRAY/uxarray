import os
import numpy as np
import xarray as xr

from unittest import TestCase
from pathlib import Path

import xarray as xr
import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestGrid(TestCase):

    def test_read_ugrid_write_exodus(self):
        """Reads a ugrid file and writes and exodus file."""

        ug_filename1 = current_path / "meshfiles" / "outCSne30.ug"
        ug_filename2 = current_path / "meshfiles" / "outRLL1deg.ug"
        ug_filename3 = current_path / "meshfiles" / "ov_RLL10deg_CSne4.ug"

        ug_outfile1 = current_path / "meshfiles" / "outCSne30.exo"
        ug_outfile2 = current_path / "meshfiles" / "outRLL1deg.g"
        ug_outfile3 = current_path / "meshfiles" / "ov_RLL10deg_CSne4.g"

        xr_ds1 = xr.open_dataset(ug_filename1)
        xr_ds2 = xr.open_dataset(ug_filename2)
        xr_ds3 = xr.open_dataset(ug_filename3)
        tgrid1 = ux.Grid(xr_ds1)
        tgrid2 = ux.Grid(xr_ds2)
        tgrid3 = ux.Grid(xr_ds3)

        tgrid1.write(str(ug_outfile1), "exodus")
        tgrid2.write(str(ug_outfile2), "exodus")
        tgrid3.write(str(ug_outfile3), "exodus")

    def test_read_ugrid_write_scrip(self):
        """Reads in augrid file and writes to a scrip file."""
        ug_filename1 = current_path / "meshfiles" / "outCSne30.ug"
        ug_filename2 = current_path / "meshfiles" / "outRLL1deg.ug"
        ug_filename3 = current_path / "meshfiles" / "ov_RLL10deg_CSne4.ug"

        ug_outfile1 = current_path / "meshfiles" / "outCSne30.nc"
        ug_outfile2 = current_path / "meshfiles" / "outRLL1deg.nc"
        ug_outfile3 = current_path / "meshfiles" / "ov_RLL10deg_CSne4.nc"

        xr_ds1 = xr.open_dataset(ug_filename1)
        xr_ds2 = xr.open_dataset(ug_filename2)
        xr_ds3 = xr.open_dataset(ug_filename3)
        tgrid1 = ux.Grid(xr_ds1)
        tgrid2 = ux.Grid(xr_ds2)
        tgrid3 = ux.Grid(xr_ds3)

        tgrid1.write(str(ug_outfile1), "scrip")
        tgrid2.write(str(ug_outfile2), "scrip")
        tgrid3.write(str(ug_outfile3), "scrip")

    def test_init_verts(self):
        """Create a uxarray grid from vertices and saves a ugrid file.

        Also, test kwargs for grid initialization
        """

        verts = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        vgrid = ux.Grid(verts, vertices=True, islatlon=True, concave=False)

        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.source_datasets is None)

        face_filename = current_path / "meshfiles" / "1face.ug"
        vgrid.write(face_filename, "ugrid")

    def test_init_grid_var_attrs(self):
        """Tests to see if accessing variables through set attributes is equal
        to using the dict."""
        # Dataset with Variables in UGRID convention
        path = current_path / "meshfiles" / "outCSne30.ug"
        xr_grid = xr.open_dataset(path)
        grid = ux.Grid(xr_grid)
        xr.testing.assert_equal(grid.Mesh2_node_x,
                                grid.ds[grid.ds_var_names["Mesh2_node_x"]])
        xr.testing.assert_equal(grid.Mesh2_node_y,
                                grid.ds[grid.ds_var_names["Mesh2_node_y"]])
        xr.testing.assert_equal(grid.Mesh2_face_nodes,
                                grid.ds[grid.ds_var_names["Mesh2_face_nodes"]])

        # Dataset with Variables NOT in UGRID convention
        path = current_path / "meshfiles" / "mesh.nc"
        xr_grid = xr.open_dataset(path)
        grid = ux.Grid(xr_grid)
        xr.testing.assert_equal(grid.Mesh2_node_x,
                                grid.ds[grid.ds_var_names["Mesh2_node_x"]])
        xr.testing.assert_equal(grid.Mesh2_node_y,
                                grid.ds[grid.ds_var_names["Mesh2_node_y"]])
        xr.testing.assert_equal(grid.Mesh2_face_nodes,
                                grid.ds[grid.ds_var_names["Mesh2_face_nodes"]])


# TODO: Move to test_shpfile/scrip when implemented
# use external package to read?
# https://gis.stackexchange.com/questions/113799/how-to-read-a-shapefile-in-python

    def test_read_shpfile(self):
        """Reads a shape file and write ugrid file."""
        with self.assertRaises(RuntimeError):
            shp_filename = current_path / "meshfiles" / "grid_fire.shp"
            tgrid = ux.Grid(str(shp_filename))

    def test_read_scrip(self):
        """Reads a scrip file and write ugrid file."""

        scrip_8 = current_path / "meshfiles" / "outCSne8.nc"
        ug_30 = current_path / "meshfiles" / "outCSne30.ug"

        # Test read from scrip and from ugrid for grid class
        xr_grid_s8 = xr.open_dataset(scrip_8)
        ux_grid_s8 = ux.Grid(xr_grid_s8)  # tests from scrip

        xr_grid_u30 = xr.open_dataset(ug_30)
        ux_grid_u30 = ux.Grid(xr_grid_u30)  # tests from ugrid
