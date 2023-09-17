import os
import numpy as np
import numpy.testing as nt
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux

from uxarray.grid.connectivity import _build_edge_node_connectivity, _build_face_edges_connectivity

from uxarray.grid.coordinates import _populate_cartesian_xyz_coord, _populate_lonlat_coord

from uxarray.grid.neighbors import BallTree

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
gridfile_mpas = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'

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

        self.grid_CSne30.encode_as("UGRID")
        self.grid_RLL1deg.encode_as("UGRID")
        self.grid_RLL10deg_CSne4.encode_as("UGRID")

        self.grid_CSne30.encode_as("Exodus")
        self.grid_RLL1deg.encode_as("Exodus")
        self.grid_RLL10deg_CSne4.encode_as("Exodus")

    def test_open_non_mesh2_write_exodus(self):
        """Loads grid files of different formats using uxarray's open_dataset
        call."""

        grid_geoflow = ux.open_grid(gridfile_CSne30)

        exods = grid_geoflow.encode_as("Exodus")
        # Remove the _FillValue attribute from the variable's attributes
        if '_FillValue' in grid_geoflow._ds['Mesh2_face_nodes'].attrs:
            del grid_geoflow._ds['Mesh2_face_nodes'].attrs['_FillValue']

        exods.to_netcdf("grid_geoflow.exo")

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
        vgrid = ux.open_grid(verts_cart, latlon=False)

        assert (vgrid.nMesh2_face == 6)
        assert (vgrid.nMesh2_node == 8)
        vgrid.encode_as("UGRID")

        # Test the case when user created a nested one-face grid
        faces_verts_one = np.array([
            np.array([[150, 10], [160, 20], [150, 30], [135, 30], [125, 20],
                      [135, 10]])
        ])
        vgrid = ux.open_grid(faces_verts_one, latlon=True)
        assert (vgrid.nMesh2_face == 1)
        assert (vgrid.nMesh2_node == 6)
        vgrid.encode_as("UGRID")

        # Test the case when user created a one-face grid
        faces_verts_single_face = np.array([[150, 10], [160, 20], [150, 30],
                                            [135, 30], [125, 20], [135, 10]])

        vgrid = ux.open_grid(faces_verts_single_face, latlon=True)
        assert (vgrid.nMesh2_face == 1)
        assert (vgrid.nMesh2_node == 6)
        vgrid.encode_as("UGRID")

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
        vgrid = ux.open_grid(faces_verts_ndarray, latlon=True)
        assert (vgrid.nMesh2_face == 3)
        assert (vgrid.nMesh2_node == 14)
        vgrid.encode_as("UGRID")

        # Test initializing Grid from list
        faces_verts_list = [[[150, 10], [160, 20], [150, 30], [135, 30],
                             [125, 20], [135, 10]],
                            [[125, 20], [135, 30], [125, 60], [110, 60],
                             [100, 30], [105, 20]],
                            [[95, 10], [105, 20], [100, 30], [85, 30], [75, 20],
                             [85, 10]]]
        vgrid = ux.open_grid(faces_verts_list, latlon=False)
        assert (vgrid.nMesh2_face == 3)
        assert (vgrid.nMesh2_node == 14)
        vgrid.encode_as("UGRID")

        # Test initializing Grid from tuples
        faces_verts_tuples = [
            ((150, 10), (160, 20), (150, 30), (135, 30), (125, 20), (135, 10)),
            ((125, 20), (135, 30), (125, 60), (110, 60), (100, 30), (105, 20)),
            ((95, 10), (105, 20), (100, 30), (85, 30), (75, 20), (85, 10))
        ]
        vgrid = ux.open_grid(faces_verts_tuples, latlon=False)
        assert (vgrid.nMesh2_face == 3)
        assert (vgrid.nMesh2_node == 14)
        vgrid.encode_as("UGRID")

    def test_init_verts_fill_values(self):
        faces_verts_filled_values = [[[150, 10], [160, 20], [150, 30],
                                      [135, 30], [125, 20], [135, 10]],
                                     [[125, 20], [135, 30], [125, 60],
                                      [110, 60], [100, 30],
                                      [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]],
                                     [[95, 10], [105, 20], [100, 30], [85, 30],
                                      [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
                                      [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]]
        vgrid = ux.open_grid(
            faces_verts_filled_values,
            latlon=False,
        )
        assert (vgrid.nMesh2_face == 3)
        assert (vgrid.nMesh2_node == 12)

    def test_grid_properties(self):
        """Tests to see if accessing variables through set properties is equal
        to using the dict."""

        # Dataset with standard UGRID variable names
        # Coordinates
        xr.testing.assert_equal(self.grid_CSne30.Mesh2_node_x,
                                self.grid_CSne30._ds["Mesh2_node_x"])
        xr.testing.assert_equal(self.grid_CSne30.Mesh2_node_y,
                                self.grid_CSne30._ds["Mesh2_node_y"])
        # Variables
        xr.testing.assert_equal(self.grid_CSne30.Mesh2_face_nodes,
                                self.grid_CSne30._ds["Mesh2_face_nodes"])

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

        xr.testing.assert_equal(grid_geoflow.Mesh2_node_x,
                                grid_geoflow._ds["Mesh2_node_x"])
        xr.testing.assert_equal(grid_geoflow.Mesh2_node_y,
                                grid_geoflow._ds["Mesh2_node_y"])
        # Variables
        xr.testing.assert_equal(grid_geoflow.Mesh2_face_nodes,
                                grid_geoflow._ds["Mesh2_face_nodes"])
        # Dimensions
        n_nodes = grid_geoflow.Mesh2_node_x.shape[0]
        n_faces, n_face_nodes = grid_geoflow.Mesh2_face_nodes.shape

        self.assertEqual(n_nodes, grid_geoflow.nMesh2_node)
        self.assertEqual(n_faces, grid_geoflow.nMesh2_face)
        self.assertEqual(n_face_nodes, grid_geoflow.nMaxMesh2_face_nodes)

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


class TestFaceAreas(TestCase):

    grid_CSne30 = ux.open_grid(gridfile_CSne30)

    def test_calculate_total_face_area_triangle(self):
        """Create a uxarray grid from vertices and saves an exodus file."""

        verts = [[[0.57735027, -5.77350269e-01, -0.57735027],
                  [0.57735027, 5.77350269e-01, -0.57735027],
                  [-0.57735027, 5.77350269e-01, -0.57735027]]]

        grid_verts = ux.open_grid(verts,
                                  vertices=True,
                                  islatlon=False,
                                  isconcave=False)

        #calculate area
        area_gaussian = grid_verts.calculate_total_face_area(
            quadrature_rule="gaussian", order=5)
        nt.assert_almost_equal(area_gaussian, constants.TRI_AREA, decimal=3)

        area_triangular = grid_verts.calculate_total_face_area(
            quadrature_rule="triangular", order=4)
        nt.assert_almost_equal(area_triangular, constants.TRI_AREA, decimal=1)

    def test_calculate_total_face_area_file(self):
        """Create a uxarray grid from vertices and saves an exodus file."""

        # = self.grid_CSne30.calculate_total_face_area()
        area = ux.open_grid(gridfile_CSne30).calculate_total_face_area()

        nt.assert_almost_equal(area, constants.MESH30_AREA, decimal=3)

    def test_calculate_total_face_area_sphere(self):
        """Computes the total face area of an MPAS mesh that lies on a unit
        sphere, with an expected total face area of 4pi."""
        mpas_grid_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'

        primal_grid = ux.open_grid(mpas_grid_path, use_dual=False)
        dual_grid = ux.open_grid(mpas_grid_path, use_dual=True)

        primal_face_area = primal_grid.calculate_total_face_area()
        dual_face_area = dual_grid.calculate_total_face_area()

        nt.assert_almost_equal(primal_face_area,
                               constants.UNIT_SPHERE_AREA,
                               decimal=3)

        nt.assert_almost_equal(dual_face_area,
                               constants.UNIT_SPHERE_AREA,
                               decimal=3)

    # TODO: Will depend on the decision for whether to provide integrate function
    # from within `Grid` as well as UxDataset
    # def test_integrate(self):
    #     xr_psi = xr.open_dataset(dsfile_vortex_CSne30)
    #     xr_v2 = xr.open_dataset(dsfile_var2_CSne30)
    #
    #     integral_psi = self.grid_CSne30.integrate(xr_psi)
    #     integral_var2 = self.grid_CSne30.integrate(xr_v2)
    #
    #     nt.assert_almost_equal(integral_psi, constants.PSI_INTG, decimal=3)
    #     nt.assert_almost_equal(integral_var2, constants.VAR2_INTG, decimal=3)

    def test_compute_face_areas_geoflow_small(self):
        """Checks if the GeoFlow Small can generate a face areas output."""
        grid_geoflow = ux.open_grid(gridfile_geoflow)

        grid_geoflow.compute_face_areas()

    # TODO: Add this test after fix to tranposed face nodes
    # def test_compute_face_areas_fesom(self):
    #     """Checks if the FESOM PI-Grid Output can generate a face areas
    #     output."""
    #     grid_fesom = ux.open_grid(gridfile_fesom)
    #
    #     grid_fesom.compute_face_areas()

    def test_verts_calc_area(self):
        faces_verts_ndarray = np.array([
            np.array([[150, 10, 0], [160, 20, 0], [150, 30, 0], [135, 30, 0],
                      [125, 20, 0], [135, 10, 0]]),
            np.array([[125, 20, 0], [135, 30, 0], [125, 60, 0], [110, 60, 0],
                      [100, 30, 0], [105, 20, 0]]),
            np.array([[95, 10, 0], [105, 20, 0], [100, 30, 0], [85, 30, 0],
                      [75, 20, 0], [85, 10, 0]]),
        ])
        # load our vertices into a UXarray Grid object
        verts_grid = ux.open_grid(faces_verts_ndarray, latlon=True)

        face_verts_areas = verts_grid.face_areas

        nt.assert_almost_equal(face_verts_areas.sum(),
                               constants.FACE_VERTS_AREA,
                               decimal=3)


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

        vgrid = ux.open_grid(verts_degree, latlon=True)
        #_populate_cartesian_xyz_coord(vgrid)

        for i in range(0, vgrid.nMesh2_node):
            nt.assert_almost_equal(vgrid.Mesh2_node_cart_x.values[i],
                                   cart_x[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid.Mesh2_node_cart_y.values[i],
                                   cart_y[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid.Mesh2_node_cart_z.values[i],
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

        vgrid = ux.open_grid(verts_cart, latlon=False)
        _populate_lonlat_coord(vgrid)
        # The connectivity in `__from_vert__()` will be formed in a reverse order
        lon_deg, lat_deg = zip(*reversed(list(zip(lon_deg, lat_deg))))
        for i in range(0, vgrid.nMesh2_node):
            nt.assert_almost_equal(vgrid._ds["Mesh2_node_x"].values[i],
                                   lon_deg[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid._ds["Mesh2_node_y"].values[i],
                                   lat_deg[i],
                                   decimal=12)


class TestConnectivity(TestCase):
    mpas_filepath = current_path / "meshfiles" / "mpas" / "QU" / "mesh.QU.1920km.151026.nc"
    exodus_filepath = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
    ugrid_filepath_01 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    ugrid_filepath_02 = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
    ugrid_filepath_03 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"

    grid_mpas = ux.open_grid(mpas_filepath)
    grid_exodus = ux.open_grid(exodus_filepath)
    grid_ugrid = ux.open_grid(ugrid_filepath_01)

    # used from constructing vertices
    f0_deg = [[120, -20], [130, -10], [120, 0], [105, 0], [95, -10], [105, -20]]
    f1_deg = [[120, 0], [120, 10], [115, 0],
              [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
              [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
              [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]
    f2_deg = [[115, 0], [120, 10], [100, 10], [105, 0],
              [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
              [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]
    f3_deg = [[95, -10], [105, 0], [95, 30], [80, 30], [70, 0], [75, -10]]
    f4_deg = [[65, -20], [75, -10], [70, 0], [55, 0], [45, -10], [55, -20]]
    f5_deg = [[70, 0], [80, 30], [70, 30], [60, 0],
              [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
              [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]
    f6_deg = [[60, 0], [70, 30], [40, 30], [45, 0],
              [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
              [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]

    # Helper function
    def _revert_edges_conn_to_face_nodes_conn(
            self, edge_nodes_connectivity: np.ndarray,
            face_edges_connectivity: np.ndarray,
            original_face_nodes_connectivity: np.ndarray):
        """utilize the edge_nodes_connectivity and face_edges_connectivity to
        generate the res_face_nodes_connectivity in the counter-clockwise
        order. The counter-clockwise order will be enforced by the passed in
        original_face_edges_connectivity. We will only use the first two nodes
        in the original_face_edges_connectivity. The order of these two nodes
        will provide a correct counter-clockwise order to build our
        res_face_nodes_connectivity. A ValueError will be raised if the first
        two nodes in the res_face_nodes_connectivity and the
        original_face_nodes_connectivity are not the same elements (The order
        doesn't matter here).

        Parameters
        ----------
        edge_nodes_connectivity : np.ndarray
            The edge_nodes_connectivity array
        face_edges_connectivity : np.ndarray
            The face_edges_connectivity array
        original_face_nodes_connectivity : np.ndarray
            The original face_nodes_connectivity array

        Returns
        -------
        res_face_nodes_connectivity : np.ndarray
            The face_nodes_connectivity array in the counter-clockwise order

        Raises
        ------
        ValueError
            if the first two nodes in the res_face_nodes_connectivity are not the same as the first two nodes in the
            original_face_nodes_connectivity
        """

        # Create a dictionary to store the face indices for each edge
        face_nodes_dict = {}

        # Loop through each face and edge to build the dictionary
        for face_idx, face_edges in enumerate(face_edges_connectivity):
            for edge_idx in face_edges:
                if edge_idx != ux.INT_FILL_VALUE:
                    edge = edge_nodes_connectivity[edge_idx]
                    if face_idx not in face_nodes_dict:
                        face_nodes_dict[face_idx] = []
                    face_nodes_dict[face_idx].append(edge[0])
                    face_nodes_dict[face_idx].append(edge[1])

        # Make sure the face_nodes_dict is in the counter-clockwise order and remove duplicate nodes
        for face_idx, face_nodes in face_nodes_dict.items():
            # First need to re-position the first two nodes position according to the original face_nodes_connectivity
            first_edge_correct = np.array([
                original_face_nodes_connectivity[face_idx][0],
                original_face_nodes_connectivity[face_idx][1]
            ])
            first_edge = np.array([face_nodes[0], face_nodes[1]])

            first_edge_correct_copy = first_edge_correct.copy()
            first_edge_copy = first_edge.copy()
            self.assertTrue(
                np.array_equal(first_edge_correct_copy.sort(),
                               first_edge_copy.sort()))
            face_nodes[0] = first_edge_correct[0]
            face_nodes[1] = first_edge_correct[1]

            i = 2
            while i < len(face_nodes):
                if face_nodes[i] != face_nodes[i - 1]:
                    # swap the order
                    old = face_nodes[i]
                    face_nodes[i] = face_nodes[i - 1]
                    face_nodes[i + 1] = old
                i += 2

            after_swapped = face_nodes

            after_swapped_remove = [after_swapped[0]]

            for i in range(1, len(after_swapped) - 1):
                if after_swapped[i] != after_swapped[i - 1]:
                    after_swapped_remove.append(after_swapped[i])

            face_nodes_dict[face_idx] = after_swapped_remove

        # Convert the dictionary to a list
        res_face_nodes_connectivity = []
        for face_idx in range(len(face_edges_connectivity)):
            res_face_nodes_connectivity.append(face_nodes_dict[face_idx])
            while len(res_face_nodes_connectivity[face_idx]
                     ) < original_face_nodes_connectivity.shape[1]:
                res_face_nodes_connectivity[face_idx].append(ux.INT_FILL_VALUE)

        return np.array(res_face_nodes_connectivity)

    def test_build_nNodes_per_face(self):
        """Tests the construction of the ``nNodes_per_face`` variable."""

        # test on grid constructed from sample datasets
        grids = [self.grid_mpas, self.grid_exodus, self.grid_ugrid]

        for grid in grids:
            # highest possible dimension dimension for a face
            max_dimension = grid.nMaxMesh2_face_nodes

            # face must be at least a triangle
            min_dimension = 3

            assert grid.nNodes_per_face.min() >= min_dimension
            assert grid.nNodes_per_face.max() <= max_dimension

        # test on grid constructed from vertices
        verts = [
            self.f0_deg, self.f1_deg, self.f2_deg, self.f3_deg, self.f4_deg,
            self.f5_deg, self.f6_deg
        ]
        grid_from_verts = ux.open_grid(verts)

        # number of non-fill-value nodes per face
        expected_nodes_per_face = np.array([6, 3, 4, 6, 6, 4, 4], dtype=int)
        nt.assert_equal(grid_from_verts.nNodes_per_face.values,
                        expected_nodes_per_face)

    def test_edge_nodes_euler(self):
        """Verifies that (``nMesh2_edge``) follows euler's formula."""
        grid_paths = [
            self.exodus_filepath, self.ugrid_filepath_01,
            self.ugrid_filepath_02, self.ugrid_filepath_03
        ]

        for grid_path in grid_paths:
            grid_ux = ux.open_grid(grid_path)

            n_face = grid_ux.nMesh2_face
            n_node = grid_ux.nMesh2_node
            n_edge = grid_ux.nMesh2_edge

            # euler's formula (n_face = n_edges - n_nodes + 2)
            assert (n_face == n_edge - n_node + 2)

    def test_build_face_edges_connectivity_mpas(self):
        """Tests the construction of (``Mesh2_edge_nodes``) on an MPAS grid
        with known edge nodes."""

        # grid with known edge node connectivity
        mpas_grid_ux = ux.open_grid(self.mpas_filepath)
        edge_nodes_expected = mpas_grid_ux._ds['Mesh2_edge_nodes'].values

        # arrange edge nodes in the same manner as Grid._build_edge_node_connectivity
        edge_nodes_expected.sort(axis=1)
        edge_nodes_expected = np.unique(edge_nodes_expected, axis=0)

        # construct edge nodes
        _build_edge_node_connectivity(mpas_grid_ux, repopulate=True)
        edge_nodes_output = mpas_grid_ux._ds['Mesh2_edge_nodes'].values

        self.assertTrue(np.array_equal(edge_nodes_expected, edge_nodes_output))

        # euler's formula (n_face = n_edges - n_nodes + 2)
        n_face = mpas_grid_ux.nMesh2_node
        n_node = mpas_grid_ux.nMesh2_face
        n_edge = edge_nodes_output.shape[0]

        assert (n_face == n_edge - n_node + 2)

    def test_build_face_edges_connectivity(self):
        """Generates Grid.Mesh2_edge_nodes from Grid.Mesh2_face_nodes."""
        ug_filename_list = [
            self.ugrid_filepath_01, self.ugrid_filepath_02,
            self.ugrid_filepath_03
        ]
        for ug_file_name in ug_filename_list:
            tgrid = ux.open_grid(ug_file_name)

            mesh2_face_nodes = tgrid._ds["Mesh2_face_nodes"]

            _build_face_edges_connectivity(tgrid)
            mesh2_face_edges = tgrid._ds.Mesh2_face_edges
            mesh2_edge_nodes = tgrid._ds.Mesh2_edge_nodes

            # Assert if the mesh2_face_edges sizes are correct.
            self.assertEqual(mesh2_face_edges.sizes["nMesh2_face"],
                             mesh2_face_nodes.sizes["nMesh2_face"])
            self.assertEqual(mesh2_face_edges.sizes["nMaxMesh2_face_edges"],
                             mesh2_face_nodes.sizes["nMaxMesh2_face_nodes"])

            # Assert if the mesh2_edge_nodes sizes are correct.
            # Euler formular for determining the edge numbers: n_face = n_edges - n_nodes + 2
            num_edges = mesh2_face_edges.sizes["nMesh2_face"] + tgrid._ds[
                "Mesh2_node_x"].sizes["nMesh2_node"] - 2
            size = mesh2_edge_nodes.sizes["nMesh2_edge"]
            self.assertEqual(mesh2_edge_nodes.sizes["nMesh2_edge"], num_edges)

            original_face_nodes_connectivity = tgrid._ds.Mesh2_face_nodes.values

            reverted_mesh2_edge_nodes = self._revert_edges_conn_to_face_nodes_conn(
                edge_nodes_connectivity=mesh2_edge_nodes.values,
                face_edges_connectivity=mesh2_face_edges.values,
                original_face_nodes_connectivity=original_face_nodes_connectivity
            )

            for i in range(len(reverted_mesh2_edge_nodes)):
                self.assertTrue(
                    np.array_equal(reverted_mesh2_edge_nodes[i],
                                   original_face_nodes_connectivity[i]))

    def test_build_face_edges_connectivity_mpas(self):
        tgrid = ux.open_grid(self.mpas_filepath)

        mesh2_face_nodes = tgrid._ds["Mesh2_face_nodes"]

        _build_face_edges_connectivity(tgrid)
        mesh2_face_edges = tgrid._ds.Mesh2_face_edges
        mesh2_edge_nodes = tgrid._ds.Mesh2_edge_nodes

        # Assert if the mesh2_face_edges sizes are correct.
        self.assertEqual(mesh2_face_edges.sizes["nMesh2_face"],
                         mesh2_face_nodes.sizes["nMesh2_face"])
        self.assertEqual(mesh2_face_edges.sizes["nMaxMesh2_face_edges"],
                         mesh2_face_nodes.sizes["nMaxMesh2_face_nodes"])

        # Assert if the mesh2_edge_nodes sizes are correct.
        # Euler formular for determining the edge numbers: n_face = n_edges - n_nodes + 2
        num_edges = mesh2_face_edges.sizes["nMesh2_face"] + tgrid._ds[
            "Mesh2_node_x"].sizes["nMesh2_node"] - 2
        size = mesh2_edge_nodes.sizes["nMesh2_edge"]
        self.assertEqual(mesh2_edge_nodes.sizes["nMesh2_edge"], num_edges)

    def test_build_face_edges_connectivity_fillvalues(self):
        verts = [
            self.f0_deg, self.f1_deg, self.f2_deg, self.f3_deg, self.f4_deg,
            self.f5_deg, self.f6_deg
        ]
        uds = ux.open_grid(verts)
        _build_face_edges_connectivity(uds)
        n_face = len(uds._ds["Mesh2_face_edges"].values)
        n_node = uds.nMesh2_node
        n_edge = len(uds._ds["Mesh2_edge_nodes"].values)

        self.assertEqual(7, n_face)
        self.assertEqual(21, n_node)
        self.assertEqual(28, n_edge)

        # We will utilize the edge_nodes_connectivity and face_edges_connectivity to generate the
        # res_face_nodes_connectivity and compare it with the uds._ds["Mesh2_face_nodes"].values
        edge_nodes_connectivity = uds._ds["Mesh2_edge_nodes"].values
        face_edges_connectivity = uds._ds["Mesh2_face_edges"].values
        face_nodes_connectivity = uds._ds["Mesh2_face_nodes"].values

        res_face_nodes_connectivity = self._revert_edges_conn_to_face_nodes_conn(
            edge_nodes_connectivity, face_edges_connectivity,
            face_nodes_connectivity)

        # Compare the res_face_nodes_connectivity with the uds._ds["Mesh2_face_nodes"].values
        self.assertTrue(
            np.array_equal(res_face_nodes_connectivity,
                           uds._ds["Mesh2_face_nodes"].values))

    def test_node_face_connectivity_from_verts(self):
        """Test generating Grid.Mesh2_node_faces from array input."""

        # We used the following codes to generate the testing face_nodes_connectivity in lonlat,
        # The index of the nodes here is just for generation purpose and ensure the topology.
        # This nodes list is only for vertices creation purposes and the nodes' order will not be used the
        # same in the Grid object; i.e. the Grid object instantiation will instead use the below
        # `face_nodes_conn_lonlat_degree`  connectivity variable and determine the actual node orders by itself.
        face_nodes_conn_lonlat_degree = [[162., 30], [216., 30], [70., 30],
                                         [162., -30], [216., -30], [70., -30]]

        # This index variable will only be used to determine the face-node lon-lat values that are
        # represented by `face_nodes_conn_lonlat`  below, which is the actual data that is used
        # by `Grid.__from_vert__()` during the creation of the grid topology.
        face_nodes_conn_index = np.array([[3, 4, 5, ux.INT_FILL_VALUE],
                                          [3, 0, 2, 5], [3, 4, 1, 0],
                                          [0, 1, 2, ux.INT_FILL_VALUE]])
        face_nodes_conn_lonlat = np.full(
            (face_nodes_conn_index.shape[0], face_nodes_conn_index.shape[1], 2),
            ux.INT_FILL_VALUE)

        for i, face_nodes_conn_index_row in enumerate(face_nodes_conn_index):
            for j, node_index in enumerate(face_nodes_conn_index_row):
                if node_index != ux.INT_FILL_VALUE:
                    face_nodes_conn_lonlat[
                        i, j] = face_nodes_conn_lonlat_degree[node_index]

        # Now we don't need the face_nodes_conn_index anymore.
        del face_nodes_conn_index

        vgrid = ux.Grid.from_face_vertices(
            face_nodes_conn_lonlat,
            latlon=True,
        )

        # We eyeballed the `Grid._face_nodes_connectivity` and wrote the following expected result
        expected = np.array([
            np.array([0, 1, ux.INT_FILL_VALUE]),
            np.array([1, 3, ux.INT_FILL_VALUE]),
            np.array([0, 1, 2]),
            np.array([1, 2, 3]),
            np.array([0, 2, ux.INT_FILL_VALUE]),
            np.array([2, 3, ux.INT_FILL_VALUE])
        ])

        self.assertTrue(np.array_equal(vgrid.Mesh2_node_faces.values, expected))

    def test_node_face_connectivity_from_files(self):
        """Test generating Grid.Mesh2_node_faces from file input."""
        grid_paths = [
            self.exodus_filepath, self.ugrid_filepath_01,
            self.ugrid_filepath_02, self.ugrid_filepath_03
        ]

        for grid_path in grid_paths:
            grid_xr = xr.open_dataset(grid_path)
            grid_ux = ux.Grid.from_dataset(grid_xr)

            # use the dictionary method to build the node_face_connectivity
            node_face_connectivity = {}
            nNodes_per_face = grid_ux.nNodes_per_face.values
            face_nodes = grid_ux._ds["Mesh2_face_nodes"].values
            for face_idx, max_nodes in enumerate(nNodes_per_face):
                cur_face_nodes = face_nodes[face_idx, 0:max_nodes]
                for j in cur_face_nodes:
                    if j not in node_face_connectivity:
                        node_face_connectivity[j] = []
                    node_face_connectivity[j].append(face_idx)

            # compare the two methods
            for i in range(grid_ux.nMesh2_node):
                face_index_from_sparse_matrix = grid_ux.Mesh2_node_faces.values[
                    i]
                valid_face_index_from_sparse_matrix = face_index_from_sparse_matrix[
                    face_index_from_sparse_matrix !=
                    grid_ux.Mesh2_node_faces.attrs["_FillValue"]]
                valid_face_index_from_sparse_matrix.sort()
                face_index_from_dict = node_face_connectivity[i]
                face_index_from_dict.sort()
                self.assertTrue(
                    np.array_equal(valid_face_index_from_sparse_matrix,
                                   face_index_from_dict))


class TestClassMethods(TestCase):

    gridfile_ugrid = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
    gridfile_mpas = current_path / "meshfiles" / "mpas" / "QU" / "mesh.QU.1920km.151026.nc"
    gridfile_exodus = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
    gridfile_scrip = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"

    def test_from_dataset(self):

        # UGRID
        xrds = xr.open_dataset(self.gridfile_ugrid)
        uxgrid = ux.Grid.from_dataset(xrds)

        # MPAS
        xrds = xr.open_dataset(self.gridfile_mpas)
        uxgrid = ux.Grid.from_dataset(xrds, use_dual=False)
        uxgrid = ux.Grid.from_dataset(xrds, use_dual=True)

        # Exodus
        xrds = xr.open_dataset(self.gridfile_exodus)
        uxgrid = ux.Grid.from_dataset(xrds)

        # SCRIP
        xrds = xr.open_dataset(self.gridfile_scrip)
        uxgrid = ux.Grid.from_dataset(xrds)

        pass

    def test_from_face_vertices(self):
        single_face_latlon = [(0.0, 90.0), (-180, 0.0), (0.0, -90)]
        uxgrid = ux.Grid.from_face_vertices(single_face_latlon, latlon=True)

        multi_face_latlon = [[(0.0, 90.0), (-180, 0.0), (0.0, -90)],
                             [(0.0, 90.0), (180, 0.0), (0.0, -90)]]
        uxgrid = ux.Grid.from_face_vertices(multi_face_latlon, latlon=True)

        single_face_cart = [(0.0,)]


class TestBallTree(TestCase):

    corner_grid_files = [gridfile_CSne30, gridfile_mpas]
    center_grid_files = [gridfile_mpas]

    def test_construction_from_nodes(self):
        """Tests the construction of the ball tree on nodes and performs a
        sample query."""

        for grid_file in self.corner_grid_files:
            uxgrid = ux.open_grid(grid_file)

            # performs a sample query
            d, ind = uxgrid.get_ball_tree(tree_type="nodes").query([3.0, 3.0],
                                                                   k=3)

    def test_construction_from_face_centers(self):
        """Tests the construction of the ball tree on center nodes and performs
        a sample query."""

        for grid_file in self.center_grid_files:
            uxgrid = ux.open_grid(grid_file)

            # performs a sample query
            d, ind = uxgrid.get_ball_tree(tree_type="face centers").query(
                [3.0, 3.0], k=3)

    def test_construction_from_both_sequentially(self):
        """Tests the construction of the ball tree on center nodes and performs
        a sample query."""

        for grid_file in self.center_grid_files:
            uxgrid = ux.open_grid(grid_file)

            # performs a sample query
            d, ind = uxgrid.get_ball_tree(tree_type="nodes").query([3.0, 3.0],
                                                                   k=3)
            d_centers, ind_centers = uxgrid.get_ball_tree(
                tree_type="face centers").query([3.0, 3.0], k=3)

    def test_antimeridian_distance_nodes(self):
        """Verifies nearest neighbor search across Antimeridian."""

        # single triangle with point on antimeridian
        verts = [(0.0, 90.0), (-180, 0.0), (0.0, -90)]

        uxgrid = ux.open_grid(verts)

        # point on antimeridian, other side of grid
        d, ind = uxgrid.get_ball_tree(tree_type="nodes").query([180.0, 0.0],
                                                               k=1)

        # distance across antimeridian is approx zero
        assert np.isclose(d, 0.0)

        # index should point to the 0th (x, y) pair (-180, 0.0)
        assert ind == 0

        # point on antimeridian, other side of grid, slightly larger than 90 due to floating point calcs
        d, ind = uxgrid.get_ball_tree(tree_type="nodes").query_radius(
            [-180, 0.0], r=90.01)

        expected_d = np.array([0.0, 90.0, 90.0])

        assert np.allclose(a=d, b=expected_d, atol=1e-03)

    def test_antimeridian_distance_face_centers(self):
        """TODO: Write addition tests once construction and representation of face centers is implemented."""
        pass
