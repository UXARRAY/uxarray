import os
import numpy as np
import xarray as xr

from unittest import TestCase
from pathlib import Path

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

        tgrid1 = ux.open_dataset(str(ug_filename1))
        tgrid2 = ux.open_dataset(str(ug_filename2))
        tgrid3 = ux.open_dataset(str(ug_filename3))

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

        tgrid1 = ux.open_dataset(str(ug_filename1))
        tgrid2 = ux.open_dataset(str(ug_filename2))
        tgrid3 = ux.open_dataset(str(ug_filename3))

        tgrid1.write(str(ug_outfile1), "scrip")
        tgrid2.write(str(ug_outfile2), "scrip")
        tgrid3.write(str(ug_outfile3), "scrip")

    def test_write_to_netcdf(self):
        """Tests that the writer functions create a correctly formatted file
        when correct arguments are used in writer call."""
        # Read in ugrid file to be used in writer functions
        ug_filename1 = current_path / "meshfiles" / "outCSne30.ug"
        tgrid1 = ux.open_dataset(str(ug_filename1))

        # Create paths for new scrip and exodus files to be stored
        scrip_outfile = current_path / "meshfiles" / "test_scrip_to_netcdf.nc"
        exodus_outfile = current_path / "meshfiles" / "test_exodus_to_netcdf.nc"
        ugrid_outfile = current_path / "meshfiles" / "test_ugrid_to_netcdf.nc"

        # User writer function with encoder argument
        tgrid1.write(str(scrip_outfile), "scrip", 'netcdf',
                     mode='w')  # test kwargs work
        tgrid1.write(str(exodus_outfile), "exodus", "netcdf")
        tgrid1.write(str(ugrid_outfile), "ugrid", "netcdf")

    def test_write_to_zarr(self):
        """Tests that the writer functions create a correctly formatted file
        when correct arguments are used in writer call."""
        # Read in ugrid file to be used in writer functions
        ug_filename1 = current_path / "meshfiles" / "outCSne30.ug"
        tgrid1 = ux.open_dataset(str(ug_filename1))

        # Create paths for new scrip and exodus files to be stored
        scrip_outfile = current_path / "meshfiles" / "test_scrip_to_zarr.zarr"
        exodus_outfile = current_path / "meshfiles" / "test_exodus_to_zarr.zarr"
        ugrid_outfile = current_path / "meshfiles" / "test_ugrid_to_zarr.zarr"

        # User writer function with encoder argument
        tgrid1.write(str(scrip_outfile), "scrip", "zarr",
                     store=scrip_outfile)  # test kwargs work
        tgrid1.write(str(exodus_outfile), "exodus", "zarr")
        tgrid1.write(str(ugrid_outfile), "ugrid", "zarr")

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
        grid = ux.open_dataset(path)
        xr.testing.assert_equal(grid.Mesh2_node_x,
                                grid.ds[grid.ds_var_names["Mesh2_node_x"]])
        xr.testing.assert_equal(grid.Mesh2_node_y,
                                grid.ds[grid.ds_var_names["Mesh2_node_y"]])
        xr.testing.assert_equal(grid.Mesh2_face_nodes,
                                grid.ds[grid.ds_var_names["Mesh2_face_nodes"]])

        # Dataset with Variables NOT in UGRID convention
        path = current_path / "meshfiles" / "mesh.nc"
        grid = ux.open_dataset(path)
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
            tgrid = ux.open_dataset(str(shp_filename))

    def test_read_scrip(self):
        """Reads a scrip file and write ugrid file."""

        scrip_8 = current_path / "meshfiles" / "outCSne8.nc"
        ug_30 = current_path / "meshfiles" / "outCSne30.ug"

        # Test read from scrip and from ugrid for grid class
        ux.open_dataset(str(scrip_8))  # tests from scrip

        ux.open_dataset(str(ug_30))  # tests from ugrid
