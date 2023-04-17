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
        """Create a uxarray grid from multiple face vertices with duplicate
        nodes and saves a ugrid file.

        Also, test kwargs for grid initialization

        The input cartesian coordinates represents 8 vertices on a cube
             7---------6
            /|        /|
           / |       / |
          3---------2  |
          |  |      |  |
          |  4------|--5
          | /       | /
          |/        |/
          0---------1
        """
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

        # The order of the vertexes is irrelevant, the following indexing is just for forming a face matrix
        face_vertices = [
            [0, 1, 2, 3],  # front face
            [1, 5, 6, 2],  # right face
            [5, 4, 7, 6],  # back face
            [4, 0, 3, 7],  # left face
            [3, 2, 6, 7],  # top face
            [4, 5, 1, 0]  # bottom face
        ]

        # Pack the cart_x/y/z into the face matrix using the index from face_vertices
        faces_coords = []
        for face in face_vertices:
            face_coords = []
            for vertex_index in face:
                x, y, z = cart_x[vertex_index], cart_y[vertex_index], cart_z[
                    vertex_index]
                face_coords.append([x, y, z])
            faces_coords.append(face_coords)

        # Now consturct the grid using the faces_coords
        verts_cart = np.array(faces_coords)
        vgrid = ux.Grid(verts_cart,
                        vertices=True,
                        islatlon=False,
                        concave=False)

        # Since the read-in data are cartesian coordinates, we should change the unit to m
        vgrid._ds.Mesh2_node_x.attrs["units"] = "m"
        vgrid._ds.Mesh2_node_y.attrs["units"] = "m"
        vgrid._ds.Mesh2_node_z.attrs["units"] = "m"
        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 6)
        assert (vgrid.nMesh2_node == 8)
        vgrid.encode_as("ugrid")

        # Test the case when user created a nested one-face grid
        faces_verts_one = np.array([
            np.array([[150, 10], [160, 20], [150, 30], [135, 30], [125, 20],
                      [135, 10]])
        ])
        vgrid = ux.Grid(faces_verts_one,
                        vertices=True,
                        islatlon=True,
                        concave=False)
        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 1)
        assert (vgrid.nMesh2_node == 6)
        vgrid.encode_as("ugrid")

        # Test the case when user created a one-face grid
        faces_verts_single_face = np.array([[150, 10], [160, 20], [150, 30],
                                            [135, 30], [125, 20], [135, 10]])

        vgrid = ux.Grid(faces_verts_single_face,
                        vertices=True,
                        islatlon=True,
                        concave=False)
        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 1)
        assert (vgrid.nMesh2_node == 6)
        vgrid.encode_as("ugrid")

    def test_init_verts_different_input_datatype(self):
        """Create a uxarray grid from multiple face vertices with different
        datatypes(ndarray, list, tuple) and saves a ugrid file.

        Also, test kwargs for grid initialization
        """

        # Test initializing Grid from ndarray
        faces_verts_ndarray = np.array([
            np.array([[150, 10], [160, 20], [150, 30], [135, 30], [125, 20],
                      [135, 10]]),
            np.array([[125, 20], [135, 30], [125, 60], [110, 60], [100, 30],
                      [105, 20]]),
            np.array([[95, 10], [105, 20], [100, 30], [85, 30], [75, 20],
                      [85, 10]]),
        ])
        vgrid = ux.open_grid(faces_verts_ndarray,
                             vertices=True,
                             islatlon=True,
                             isconcave=False)

        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 3)
        assert (vgrid.nMesh2_node == 14)
        vgrid.encode_as("ugrid")

        # Test initializing Grid from list
        faces_verts_list = [[[150, 10], [160, 20], [150, 30], [135, 30],
                             [125, 20], [135, 10]],
                            [[125, 20], [135, 30], [125, 60], [110, 60],
                             [100, 30], [105, 20]],
                            [[95, 10], [105, 20], [100, 30], [85, 30], [75, 20],
                             [85, 10]]]
        vgrid = ux.Grid(faces_verts_list,
                        vertices=True,
                        islatlon=False,
                        concave=False)
        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 3)
        assert (vgrid.nMesh2_node == 14)
        vgrid.encode_as("ugrid")

        # Test initializing Grid from tuples
        faces_verts_tuples = [
            ((150, 10), (160, 20), (150, 30), (135, 30), (125, 20), (135, 10)),
            ((125, 20), (135, 30), (125, 60), (110, 60), (100, 30), (105, 20)),
            ((95, 10), (105, 20), (100, 30), (85, 30), (75, 20), (85, 10))
        ]
        vgrid = ux.Grid(faces_verts_tuples,
                        vertices=True,
                        islatlon=False,
                        concave=False)
        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 3)
        assert (vgrid.nMesh2_node == 14)
        vgrid.encode_as("ugrid")

    def test_init_verts_fill_values(self):
        faces_verts_filled_values = [[[150, 10], [160, 20], [150, 30],
                                      [135, 30], [125, 20], [135, 10]],
                                     [[125, 20], [135, 30], [125, 60],
                                      [110, 60], [100, 30],
                                      [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]],
                                     [[95, 10], [105, 20], [100, 30], [85, 30],
                                      [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
                                      [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]]
        vgrid = ux.Grid(faces_verts_filled_values,
                        vertices=True,
                        islatlon=False,
                        concave=False)
        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 3)
        assert (vgrid.nMesh2_node == 12)

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


class TestOperators(TestCase):
    grid_CSne30_01 = ux.open_grid(gridfile_CSne30)
    grid_CSne30_02 = ux.open_grid(gridfile_CSne30)
    grid_RLL1deg = ux.open_grid(gridfile_RLL1deg)

    def test_eq(self):
        """Test Equals ('==') operator."""
        assert self.grid_CSne30_01 == self.grid_CSne30_02

    def test_ne(self):
        """Test Not Equals ('!=') operator."""
        assert self.grid_CSne30_01 != self.grid_RLL1deg


class TestIntegrate(TestCase):

    grid_CSne30 = ux.open_grid(gridfile_CSne30)

    def test_calculate_total_face_area_triangle(self):
        """Create a uxarray grid from vertices and saves an exodus file."""

        verts = [[[0.57735027, -5.77350269e-01, -0.57735027],
                  [0.57735027, 5.77350269e-01, -0.57735027],
                  [-0.57735027, 5.77350269e-01, -0.57735027]]]

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
            45.0001052295749, 45.0001052295749, 360 - 45.0001052295749,
            360 - 45.0001052295749
        ]
        lat_deg = [
            35.2655522903022, -35.2655522903022, 35.2655522903022,
            -35.2655522903022
        ]
        cart_x = [
            0.577340924821405, 0.577340924821405, 0.577340924821405,
            0.577340924821405
        ]

        cart_y = [
            0.577343045516932, 0.577343045516932, -0.577343045516932,
            -0.577343045516932
        ]

        cart_z = [
            -0.577366836872017, 0.577366836872017, -0.577366836872017,
            0.577366836872017
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
        # These points correspond to the 4 vertexes on a cube.

        lon_deg = [
            45.0001052295749, 45.0001052295749, 360 - 45.0001052295749,
            360 - 45.0001052295749
        ]
        lat_deg = [
            35.2655522903022, -35.2655522903022, 35.2655522903022,
            -35.2655522903022
        ]
        cart_x = [
            0.577340924821405, 0.577340924821405, 0.577340924821405,
            0.577340924821405
        ]
        cart_y = [
            0.577343045516932, 0.577343045516932, -0.577343045516932,
            -0.577343045516932
        ]
        cart_z = [
            0.577366836872017, -0.577366836872017, 0.577366836872017,
            -0.577366836872017
        ]

        verts_cart = np.stack((cart_x, cart_y, cart_z), axis=1)

        vgrid = ux.Grid(verts_cart)
        vgrid._ds.Mesh2_node_x.attrs["units"] = "m"
        vgrid._ds.Mesh2_node_y.attrs["units"] = "m"
        vgrid._ds.Mesh2_node_z.attrs["units"] = "m"
        vgrid._populate_lonlat_coord()
        # The connectivity in `__from_vert__()` will be formed in a reverse order
        lon_deg, lat_deg = zip(*reversed(list(zip(lon_deg, lat_deg))))
        for i in range(0, vgrid.nMesh2_node):
            nt.assert_almost_equal(vgrid._ds["Mesh2_node_x"].values[i],
                                   lon_deg[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid._ds["Mesh2_node_y"].values[i],
                                   lat_deg[i],
                                   decimal=12)
