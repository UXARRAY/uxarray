import os
import numpy as np
import xarray as xr

from unittest import TestCase
from pathlib import Path

import xarray as xr
import uxarray as ux
import numpy.testing as nt

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestGrid(TestCase):
    ug_filename1 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    ug_filename2 = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
    ug_filename3 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"

    xr_ds1 = xr.open_dataset(ug_filename1)
    xr_ds2 = xr.open_dataset(ug_filename2)
    xr_ds3 = xr.open_dataset(ug_filename3)
    tgrid1 = ux.Grid(xr_ds1)
    tgrid2 = ux.Grid(xr_ds2)
    tgrid3 = ux.Grid(xr_ds3)

    def test_encode_as(self):
        """Reads a ugrid file and encodes it as `xarray.Dataset` in various
        types."""

        self.tgrid1.encode_as("ugrid")
        self.tgrid2.encode_as("ugrid")
        self.tgrid3.encode_as("ugrid")

        self.tgrid1.encode_as("exodus")
        self.tgrid2.encode_as("exodus")
        self.tgrid3.encode_as("exodus")

    def test_open_non_mesh2_write_exodus(self):
        """Loads grid files of different formats using uxarray's open_dataset
        call."""

        path = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
        xr_grid = xr.open_dataset(path)
        grid = ux.Grid(xr_grid)

        grid.encode_as("exodus")

    def test_init_verts(self):
        """Create a uxarray grid from vertices and saves a ugrid file.

        Also, test kwargs for grid initialization
        """

        verts = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        vgrid = ux.Grid(verts, vertices=True, islatlon=True, concave=False)

        assert (vgrid.source_grid == "From vertices")

        vgrid.encode_as("ugrid")

    def test_init_grid_var_attrs(self):
        """Tests to see if accessing variables through set attributes is equal
        to using the dict."""

        # Dataset with standard UGRID variable names
        # Coordinates
        xr.testing.assert_equal(
            self.tgrid1.Mesh2_node_x,
            self.tgrid1.ds[self.tgrid1.ds_var_names["Mesh2_node_x"]])
        xr.testing.assert_equal(
            self.tgrid1.Mesh2_node_y,
            self.tgrid1.ds[self.tgrid1.ds_var_names["Mesh2_node_y"]])
        # Variables
        xr.testing.assert_equal(
            self.tgrid1.Mesh2_face_nodes,
            self.tgrid1.ds[self.tgrid1.ds_var_names["Mesh2_face_nodes"]])

        # Dimensions
        n_nodes = self.tgrid1.Mesh2_node_x.shape[0]
        n_faces, n_face_nodes = self.tgrid1.Mesh2_face_nodes.shape

        self.assertEqual(n_nodes, self.tgrid1.nMesh2_node)
        self.assertEqual(n_faces, self.tgrid1.nMesh2_face)
        self.assertEqual(n_face_nodes, self.tgrid1.nMaxMesh2_face_nodes)

        # xr.testing.assert_equal(
        #     self.tgrid1.nMesh2_node,
        #     self.tgrid1.ds[self.tgrid1.ds_var_names["nMesh2_node"]])
        # xr.testing.assert_equal(
        #     self.tgrid1.nMesh2_face,
        #     self.tgrid1.ds[self.tgrid1.ds_var_names["nMesh2_face"]])

        # Dataset with non-standard UGRID variable names
        path = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
        xr_grid = xr.open_dataset(path)
        grid = ux.Grid(xr_grid)
        xr.testing.assert_equal(grid.Mesh2_node_x,
                                grid.ds[grid.ds_var_names["Mesh2_node_x"]])
        xr.testing.assert_equal(grid.Mesh2_node_y,
                                grid.ds[grid.ds_var_names["Mesh2_node_y"]])
        # Variables
        xr.testing.assert_equal(grid.Mesh2_face_nodes,
                                grid.ds[grid.ds_var_names["Mesh2_face_nodes"]])
        # Dimensions
        n_nodes = grid.Mesh2_node_x.shape[0]
        n_faces, n_face_nodes = grid.Mesh2_face_nodes.shape

        self.assertEqual(n_nodes, grid.nMesh2_node)
        self.assertEqual(n_faces, grid.nMesh2_face)
        self.assertEqual(n_face_nodes, grid.nMaxMesh2_face_nodes)

    # def test_init_dimension_attrs(self):

    # TODO: Move to test_shpfile/scrip when implemented
    # use external package to read?
    # https://gis.stackexchange.com/questions/113799/how-to-read-a-shapefile-in-python

    def test_read_shpfile(self):
        """Reads a shape file and write ugrid file."""
        with self.assertRaises(RuntimeError):
            shp_filename = current_path / "meshfiles" / "shp" / "grid_fire.shp"
            tgrid = ux.Grid(str(shp_filename))

    def test_read_scrip(self):
        """Reads a scrip file."""

        scrip_8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
        ug_30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"

        # Test read from scrip and from ugrid for grid class
        xr_grid_s8 = xr.open_dataset(scrip_8)
        ux_grid_s8 = ux.Grid(xr_grid_s8)  # tests from scrip

        xr_grid_u30 = xr.open_dataset(ug_30)
        ux_grid_u30 = ux.Grid(xr_grid_u30)  # tests from ugrid


class TestIntegrate(TestCase):
    mesh_file30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    data_file30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"
    data_file30_v2 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"

    def test_calculate_total_face_area_triangle(self):
        """Create a uxarray grid from vertices and saves an exodus file."""
        verts = np.array([[0.57735027, -5.77350269e-01, -0.57735027],
                          [0.57735027, 5.77350269e-01, -0.57735027],
                          [-0.57735027, 5.77350269e-01, -0.57735027]])
        vgrid = ux.Grid(verts)

        # get node names for each grid object
        x_var = vgrid.ds_var_names["Mesh2_node_x"]
        y_var = vgrid.ds_var_names["Mesh2_node_y"]
        z_var = vgrid.ds_var_names["Mesh2_node_z"]

        vgrid.ds[x_var].attrs["units"] = "m"
        vgrid.ds[y_var].attrs["units"] = "m"
        vgrid.ds[z_var].attrs["units"] = "m"

        area_gaussian = vgrid.calculate_total_face_area(
            quadrature_rule="gaussian", order=5)
        nt.assert_almost_equal(area_gaussian, constants.TRI_AREA, decimal=3)

        area_triangular = vgrid.calculate_total_face_area(
            quadrature_rule="triangular", order=4)
        nt.assert_almost_equal(area_triangular, constants.TRI_AREA, decimal=1)

    def test_calculate_total_face_area_file(self):
        """Create a uxarray grid from vertices and saves an exodus file."""

        xr_grid = xr.open_dataset(str(self.mesh_file30))
        grid = ux.Grid(xr_grid)

        area = grid.calculate_total_face_area()

        nt.assert_almost_equal(area, constants.MESH30_AREA, decimal=3)

    def test_integrate(self):
        xr_grid = xr.open_dataset(self.mesh_file30)
        xr_psi = xr.open_dataset(self.data_file30)
        xr_v2 = xr.open_dataset(self.data_file30_v2)

        u_grid = ux.Grid(xr_grid)

        integral_psi = u_grid.integrate(xr_psi)
        integral_var2 = u_grid.integrate(xr_v2)

        nt.assert_almost_equal(integral_psi, constants.PSI_INTG, decimal=3)
        nt.assert_almost_equal(integral_var2, constants.VAR2_INTG, decimal=3)


class TestFaceAreas(TestCase):

    def test_compute_face_areas_geoflow_small(self):
        """Checks if the GeoFlow Small can generate a face areas output."""
        geoflow_small_grid = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
        grid_1_ds = xr.open_dataset(geoflow_small_grid)
        grid_1 = ux.Grid(grid_1_ds)
        grid_1.compute_face_areas()

    def test_compute_face_areas_fesom(self):
        """Checks if the FESOM PI-Grid Output can generate a face areas
        output."""

        fesom_grid_small = current_path / "meshfiles" / "ugrid" / "fesom" / "fesom.mesh.diag.nc"
        grid_2_ds = xr.open_dataset(fesom_grid_small)
        grid_2 = ux.Grid(grid_2_ds)
        grid_2.compute_face_areas()


class TestPopulateCoordinates(TestCase):

    def test_populate_cartesian_xyz_coord(self):
        # The following testcases are generated through the matlab cart2sph/sph2cart functions
        # These points correspond to the eight vertices of a cube.
        lon_deg = [
            45.0001052295749, 45.0001052295749, -45.0001052295749,
            -45.0001052295749, 135.000315688725, 135.000315688725,
            -135.000315688725, -135.000315688725
        ]
        lat_deg = [
            35.2655522903022, -35.2655522903022, 35.2655522903022,
            -35.2655522903022, 35.2655522903022, -35.2655522903022,
            35.2655522903022, -35.2655522903022
        ]
        cart_x = [
            0.577340924821405, 0.577340924821405, 0.577340924821405,
            0.577340924821405, -0.577345166204668, -0.577345166204668,
            -0.577345166204668, -0.577345166204668
        ]
        cart_y = [
            0.577343045516932, 0.577343045516932, -0.577343045516932,
            -0.577343045516932, 0.577338804118089, 0.577338804118089,
            -0.577338804118089, -0.577338804118089
        ]
        cart_z = [
            0.577366836872017, -0.577366836872017, 0.577366836872017,
            -0.577366836872017, 0.577366836872017, -0.577366836872017,
            0.577366836872017, -0.577366836872017
        ]
        verts_degree = np.stack((lon_deg, lat_deg), axis=1)
        vgrid = ux.Grid(verts_degree)
        vgrid._populate_cartesian_xyz_coord()
        for i in range(0, vgrid.nMesh2_node):
            nt.assert_almost_equal(vgrid.ds["Mesh2_node_cart_x"].values[i],
                                   cart_x[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid.ds["Mesh2_node_cart_y"].values[i],
                                   cart_y[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid.ds["Mesh2_node_cart_z"].values[i],
                                   cart_z[i],
                                   decimal=12)

    def test_populate_lonlat_coord(self):
        # The following testcases are generated through the matlab cart2sph/sph2cart functions
        # These points correspond to the eight vertices of a cube.
        lon_deg = [
            45.0001052295749, 45.0001052295749, 360 - 45.0001052295749,
            360 - 45.0001052295749, 135.000315688725, 135.000315688725,
            360 - 135.000315688725, 360 - 135.000315688725
        ]
        lat_deg = [
            35.2655522903022, -35.2655522903022, 35.2655522903022,
            -35.2655522903022, 35.2655522903022, -35.2655522903022,
            35.2655522903022, -35.2655522903022
        ]
        cart_x = [
            0.577340924821405, 0.577340924821405, 0.577340924821405,
            0.577340924821405, -0.577345166204668, -0.577345166204668,
            -0.577345166204668, -0.577345166204668
        ]
        cart_y = [
            0.577343045516932, 0.577343045516932, -0.577343045516932,
            -0.577343045516932, 0.577338804118089, 0.577338804118089,
            -0.577338804118089, -0.577338804118089
        ]
        cart_z = [
            0.577366836872017, -0.577366836872017, 0.577366836872017,
            -0.577366836872017, 0.577366836872017, -0.577366836872017,
            0.577366836872017, -0.577366836872017
        ]

        verts_cart = np.stack((cart_x, cart_y, cart_z), axis=1)
        vgrid = ux.Grid(verts_cart)
        vgrid.ds.Mesh2_node_x.attrs["units"] = "m"
        vgrid.ds.Mesh2_node_y.attrs["units"] = "m"
        vgrid.ds.Mesh2_node_z.attrs["units"] = "m"
        vgrid._populate_lonlat_coord()
        for i in range(0, vgrid.nMesh2_node):
            nt.assert_almost_equal(vgrid.ds["Mesh2_node_x"].values[i],
                                   lon_deg[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid.ds["Mesh2_node_y"].values[i],
                                   lat_deg[i],
                                   decimal=12)
