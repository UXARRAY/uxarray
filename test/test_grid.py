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
        vgrid = ux.open_grid(verts_cart,
                             vertices=True,
                             islatlon=False,
                             concave=False)

        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 6)
        assert (vgrid.nMesh2_node == 8)
        vgrid.encode_as("ugrid")

        # Test the case when user created a nested one-face grid
        faces_verts_one = np.array([
            np.array([[150, 10], [160, 20], [150, 30], [135, 30], [125, 20],
                      [135, 10]])
        ])
        vgrid = ux.open_grid(faces_verts_one,
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

        vgrid = ux.open_grid(faces_verts_single_face,
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
        vgrid = ux.open_grid(faces_verts_list,
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
        vgrid = ux.open_grid(faces_verts_tuples,
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
        vgrid = ux.open_grid(faces_verts_filled_values,
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

        grid_verts = ux.open_grid(verts,
                                  vertices=True,
                                  islatlon=False,
                                  concave=False)

        #calculate area
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

    def test_calculate_total_face_area_sphere(self):
        """Computes the total face area of an MPAS mesh that lies on a unit
        sphere, with an expected total face area of 4pi."""
        mpas_grid_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'

        ds = xr.open_dataset(mpas_grid_path)
        primal_grid = ux.Grid(ds, use_dual=False)
        dual_grid = ux.Grid(ds, use_dual=True)

        primal_face_area = primal_grid.calculate_total_face_area()
        dual_face_area = dual_grid.calculate_total_face_area()

        nt.assert_almost_equal(primal_face_area,
                               constants.UNIT_SPHERE_AREA,
                               decimal=3)

        nt.assert_almost_equal(dual_face_area,
                               constants.UNIT_SPHERE_AREA,
                               decimal=3)

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

    # removed test until fix to tranposed face nodes
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
        verts_grid = ux.Grid(faces_verts_ndarray,
                             vertices=True,
                             islatlon=True,
                             concave=False)

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

        vgrid = ux.open_grid(verts_degree, islatlon=False)
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

        vgrid = ux.open_grid(verts_cart, islatlon=False)
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


class TestConnectivity(TestCase):
    mpas_filepath = current_path / "meshfiles" / "mpas" / "QU" / "mesh.QU.1920km.151026.nc"
    exodus_filepath = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
    ugrid_filepath_01 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    ugrid_filepath_02 = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
    ugrid_filepath_03 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"

    xrds_maps = xr.open_dataset(mpas_filepath)
    xrds_exodus = xr.open_dataset(exodus_filepath)
    xrds_ugrid = xr.open_dataset(ugrid_filepath_01)

    grid_mpas = ux.Grid(xrds_maps)
    grid_exodus = ux.Grid(xrds_exodus)
    grid_ugrid = ux.Grid(xrds_ugrid)

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
        grid_from_verts = ux.Grid(verts)

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
            grid_xr = xr.open_dataset(grid_path)
            grid_ux = ux.Grid(grid_xr)

            n_face = grid_ux.nMesh2_face
            n_node = grid_ux.nMesh2_node
            n_edge = grid_ux.nMesh2_edge

            # euler's formula (n_face = n_edges - n_nodes + 2)
            assert (n_face == n_edge - n_node + 2)

    def test_build_face_edges_connectivity_mpas(self):
        """Tests the construction of (``Mesh2_edge_nodes``) on an MPAS grid
        with known edge nodes."""

        # grid with known edge node connectivity
        mpas_grid_xr = xr.open_dataset(self.mpas_filepath)
        mpas_grid_ux = ux.Grid(mpas_grid_xr)
        edge_nodes_expected = mpas_grid_ux._ds['Mesh2_edge_nodes'].values

        # arrange edge nodes in the same manner as Grid._build_edge_node_connectivity
        edge_nodes_expected.sort(axis=1)
        edge_nodes_expected = np.unique(edge_nodes_expected, axis=0)

        # construct edge nodes
        mpas_grid_ux._build_edge_node_connectivity(repopulate=True)
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
            xr_ds = xr.open_dataset(ug_file_name)
            tgrid = ux.Grid(xr_ds)

            mesh2_face_nodes = tgrid._ds["Mesh2_face_nodes"]

            tgrid._build_face_edges_connectivity()
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
        xr_ds = xr.open_dataset(self.mpas_filepath)
        tgrid = ux.Grid(xr_ds)

        mesh2_face_nodes = tgrid._ds["Mesh2_face_nodes"]

        tgrid._build_face_edges_connectivity()
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
        uds = ux.Grid(verts)
        uds._build_face_edges_connectivity()
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
