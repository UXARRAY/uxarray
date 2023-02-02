import os
import numpy as np
import numpy.testing as nt
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_CSne8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
gridfile_RLL1deg = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
gridfile_RLL10deg_CSne4 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"
gridfile_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
gridfile_fesom = current_path / "meshfiles" / "ugrid" / "fesom" / "fesom.mesh.diag.nc"
gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"

dsfile_vortex_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"
dsfile_var2_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"

shp_filename = current_path / "meshfiles" / "shp" / "grid_fire.shp"


class TestGrid(TestCase):

    grid_CSne30 = ux.open_grid(gridfile_CSne30)
    grid_RLL1deg = ux.open_grid(gridfile_RLL1deg)
    grid_RLL10deg_CSne4 = ux.open_grid(gridfile_RLL10deg_CSne4)

    def test_encode_as(self):
        """Reads a ugrid file and encodes it as `xarray.Dataset` in various
        types."""

        self.grid_CSne30.encode_as("ugrid")
        self.grid_RLL1deg.encode_as("ugrid")
        self.grid_RLL10deg_CSne4.encode_as("ugrid")

        self.grid_CSne30.encode_as("exodus")
        self.grid_RLL1deg.encode_as("exodus")
        self.grid_RLL10deg_CSne4.encode_as("exodus")

    def test_open_non_mesh2_write_exodus(self):
        """Loads grid files of different formats using uxarray's open_dataset
        call."""

        grid_geoflow = ux.open_grid(gridfile_CSne30)

        grid_geoflow.encode_as("exodus")

    def test_init_verts(self):
        """Create a uxarray grid from vertices and saves a ugrid file.

        Also, test kwargs for grid initialization
        """

        verts = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        vgrid = ux.open_grid(verts, islatlon=True, isconcave=False)

        assert (vgrid.source_grid == "From vertices")

        vgrid.encode_as("ugrid")

    def test_init_grid_var_attrs(self):
        """Tests to see if accessing variables through set attributes is equal
        to using the dict."""

        # Dataset with standard UGRID variable names
        # Coordinates
        xr.testing.assert_equal(
            self.grid_CSne30.Mesh2_node_x, self.grid_CSne30._ds[
                self.grid_CSne30.grid_var_names["Mesh2_node_x"]])
        xr.testing.assert_equal(
            self.grid_CSne30.Mesh2_node_y, self.grid_CSne30._ds[
                self.grid_CSne30.grid_var_names["Mesh2_node_y"]])
        # Variables
        xr.testing.assert_equal(
            self.grid_CSne30.Mesh2_face_nodes, self.grid_CSne30._ds[
                self.grid_CSne30.grid_var_names["Mesh2_face_nodes"]])

        # Dimensions
        n_nodes = self.grid_CSne30.Mesh2_node_x.shape[0]
        n_faces, n_face_nodes = self.grid_CSne30.Mesh2_face_nodes.shape

        self.assertEqual(n_nodes, self.grid_CSne30.nMesh2_node)
        self.assertEqual(n_faces, self.grid_CSne30.nMesh2_face)
        self.assertEqual(n_face_nodes, self.grid_CSne30.nMaxMesh2_face_nodes)

        # xr.testing.assert_equal(
        #     self.tgrid1.nMesh2_node,
        #     self.tgrid1._ds[self.tgrid1.grid_var_names["nMesh2_node"]])
        # xr.testing.assert_equal(
        #     self.tgrid1.nMesh2_face,
        #     self.tgrid1._ds[self.tgrid1.grid_var_names["nMesh2_face"]])

        # Dataset with non-standard UGRID variable names
        grid_geoflow = ux.open_grid(gridfile_geoflow)

        xr.testing.assert_equal(
            grid_geoflow.Mesh2_node_x,
            grid_geoflow._ds[grid_geoflow.grid_var_names["Mesh2_node_x"]])
        xr.testing.assert_equal(
            grid_geoflow.Mesh2_node_y,
            grid_geoflow._ds[grid_geoflow.grid_var_names["Mesh2_node_y"]])
        # Variables
        xr.testing.assert_equal(
            grid_geoflow.Mesh2_face_nodes,
            grid_geoflow._ds[grid_geoflow.grid_var_names["Mesh2_face_nodes"]])
        # Dimensions
        n_nodes = grid_geoflow.Mesh2_node_x.shape[0]
        n_faces, n_face_nodes = grid_geoflow.Mesh2_face_nodes.shape

        self.assertEqual(n_nodes, grid_geoflow.nMesh2_node)
        self.assertEqual(n_faces, grid_geoflow.nMesh2_face)
        self.assertEqual(n_face_nodes, grid_geoflow.nMaxMesh2_face_nodes)

    # def test_init_dimension_attrs(self):

    # TODO: Move to test_shpfile/scrip when implemented
    # use external package to read?
    # https://gis.stackexchange.com/questions/113799/how-to-read-a-shapefile-in-python

    def test_read_shpfile(self):
        """Reads a shape file and write ugrid file."""
        with self.assertRaises(ValueError):
            grid_shp = ux.open_grid(shp_filename)

    def test_read_scrip(self):
        """Reads a scrip file."""

        # Test read from scrip and from ugrid for grid class
        grid_CSne8 = ux.open_grid(gridfile_CSne8)  # tests from scrip


class TestIntegrate(TestCase):

    grid_CSne30 = ux.open_grid(gridfile_CSne30)

    def test_calculate_total_face_area_triangle(self):
        """Create a uxarray grid from vertices and saves an exodus file."""
        verts = np.array([[0.57735027, -5.77350269e-01, -0.57735027],
                          [0.57735027, 5.77350269e-01, -0.57735027],
                          [-0.57735027, 5.77350269e-01, -0.57735027]])

        grid_verts = ux.open_grid(verts)

        # get node names for each grid object
        x_var = grid_verts.grid_var_names["Mesh2_node_x"]
        y_var = grid_verts.grid_var_names["Mesh2_node_y"]
        z_var = grid_verts.grid_var_names["Mesh2_node_z"]

        grid_verts._ds[x_var].attrs["units"] = "m"
        grid_verts._ds[y_var].attrs["units"] = "m"
        grid_verts._ds[z_var].attrs["units"] = "m"

        area_gaussian = grid_verts.calculate_total_face_area(
            quadrature_rule="gaussian", order=5)
        nt.assert_almost_equal(area_gaussian, constants.TRI_AREA, decimal=3)

        area_triangular = grid_verts.calculate_total_face_area(
            quadrature_rule="triangular", order=4)
        nt.assert_almost_equal(area_triangular, constants.TRI_AREA, decimal=1)

    def test_calculate_total_face_area_file(self):
        """Create a uxarray grid from vertices and saves an exodus file."""

        area = self.grid_CSne30.calculate_total_face_area()

        nt.assert_almost_equal(area, constants.MESH30_AREA, decimal=3)

    def test_integrate(self):
        xr_psi = xr.open_dataset(dsfile_vortex_CSne30)
        xr_v2 = xr.open_dataset(dsfile_var2_CSne30)

        integral_psi = self.grid_CSne30.integrate(xr_psi)
        integral_var2 = self.grid_CSne30.integrate(xr_v2)

        nt.assert_almost_equal(integral_psi, constants.PSI_INTG, decimal=3)
        nt.assert_almost_equal(integral_var2, constants.VAR2_INTG, decimal=3)


class TestFaceAreas(TestCase):

    def test_compute_face_areas_geoflow_small(self):
        """Checks if the GeoFlow Small can generate a face areas output."""
        grid_geoflow = ux.open_grid(gridfile_geoflow)

        grid_geoflow.compute_face_areas()

    def test_compute_face_areas_fesom(self):
        """Checks if the FESOM PI-Grid Output can generate a face areas
        output."""
        grid_fesom = ux.open_grid(gridfile_fesom)

        grid_fesom.compute_face_areas()


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

        vgrid = ux.open_grid(verts_degree)
        vgrid._populate_cartesian_xyz_coord()

        for i in range(0, vgrid.nMesh2_node):
            nt.assert_almost_equal(vgrid._ds["Mesh2_node_cart_x"].values[i],
                                   cart_x[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid._ds["Mesh2_node_cart_y"].values[i],
                                   cart_y[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid._ds["Mesh2_node_cart_z"].values[i],
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
        vgrid._ds.Mesh2_node_x.attrs["units"] = "m"
        vgrid._ds.Mesh2_node_y.attrs["units"] = "m"
        vgrid._ds.Mesh2_node_z.attrs["units"] = "m"
        vgrid._populate_lonlat_coord()
        for i in range(0, vgrid.nMesh2_node):
            nt.assert_almost_equal(vgrid._ds["Mesh2_node_x"].values[i],
                                   lon_deg[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid._ds["Mesh2_node_y"].values[i],
                                   lat_deg[i],
                                   decimal=12)
