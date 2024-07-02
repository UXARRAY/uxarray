import os
import numpy as np
import numpy.testing as nt
import random
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux

from uxarray.grid.connectivity import _replace_fill_values
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _normalize_xyz, _xyz_to_lonlat_rad
from uxarray.grid.arcs import point_within_gca, _angle_of_2_vectors, in_between
from uxarray.grid.utils import _get_cartesian_face_edge_nodes, _get_lonlat_rad_face_edge_nodes
from uxarray.grid.geometry import _pole_point_inside_polygon

try:
    import constants
except ImportError:
    from . import constants

# Data files
current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_exo_CSne8 = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
gridfile_scrip_CSne8 = current_path / 'meshfiles' / "scrip" / "outCSne8" / 'outCSne8.nc'
gridfile_geoflowsmall_grid = current_path / 'meshfiles' / "ugrid" / "geoflow-small" / 'grid.nc'
gridfile_geoflowsmall_var = current_path / 'meshfiles' / "ugrid" / "geoflow-small" / 'v1.nc'

err_tolerance = 1.0e-12


class TestIntegrate(TestCase):

    def test_face_area_coords(self):
        """Test function for helper function get_all_face_area_from_coords."""
        # Note: currently only testing one face, but this can be used to get area of multiple faces
        x = np.array([0.57735027, 0.57735027, -0.57735027])
        y = np.array([-5.77350269e-01, 5.77350269e-01, 5.77350269e-01])
        z = np.array([-0.57735027, -0.57735027, -0.57735027])

        face_nodes = np.array([[0, 1, 2]]).astype(INT_DTYPE)
        face_dimension = np.array([3], dtype=INT_DTYPE)

        area, jacobian = ux.grid.area.get_all_face_area_from_coords(
            x, y, z, face_nodes, face_dimension, 3, coords_type="cartesian")

        nt.assert_almost_equal(area, constants.TRI_AREA, decimal=1)

    def test_calculate_face_area(self):
        """Test function for helper function calculate_face_area - only one face."""
        # Note: currently only testing one face, but this can be used to get area of multiple faces
        # Also note, this does not need face_nodes, assumes nodes are in counterclockwise orientation
        x = np.array([0.57735027, 0.57735027, -0.57735027])
        y = np.array([-5.77350269e-01, 5.77350269e-01, 5.77350269e-01])
        z = np.array([-0.57735027, -0.57735027, -0.57735027])

        area, jacobian = ux.grid.area.calculate_face_area(
            x, y, z, "gaussian", 5, "cartesian")

        nt.assert_almost_equal(area, constants.TRI_AREA, decimal=3)

    def test_quadrature(self):
        order = 1
        dG, dW = ux.grid.area.get_tri_quadratureDG(order)
        G = np.array([[0.33333333, 0.33333333, 0.33333333]])
        W = np.array([1.0])

        np.testing.assert_array_almost_equal(G, dG)
        np.testing.assert_array_almost_equal(W, dW)

        dG, dW = ux.grid.area.get_gauss_quadratureDG(order)

        G = np.array([[0.5]])
        W = np.array([1.0])

        np.testing.assert_array_almost_equal(G, dG)
        np.testing.assert_array_almost_equal(W, dW)


class TestGridCenter(TestCase):

    def test_grid_center(self):
        """Calculates if the calculated center point of a grid box is the same
        as a given value for the same dataset."""
        ds_scrip_CSne8 = xr.open_dataset(gridfile_scrip_CSne8)

        # select actual center_lat/lon
        scrip_center_lon = ds_scrip_CSne8['grid_center_lon']
        scrip_center_lat = ds_scrip_CSne8['grid_center_lat']

        # Calculate the center_lat/lon using same dataset's corner_lat/lon
        calc_center = ux.io._scrip.grid_center_lat_lon(ds_scrip_CSne8)
        calc_lat = calc_center[0]
        calc_lon = calc_center[1]

        # Test that calculated center_lat/lon is the same as actual center_lat/lon
        np.testing.assert_array_almost_equal(scrip_center_lat, calc_lat)
        np.testing.assert_array_almost_equal(scrip_center_lon, calc_lon)


class TestCoordinatesConversion(TestCase):

    def test_normalize_in_place(self):
        x, y, z = _normalize_xyz(
            random.random(), random.random(),
             random.random())

        self.assertLessEqual(np.absolute(np.sqrt(x * x + y * y + z * z) - 1),
                             err_tolerance)

    def test_node_xyz_to_lonlat_rad(self):
        x, y, z = _normalize_xyz(*[
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ])

        lon, lat = _xyz_to_lonlat_rad(x, y, z)
        new_x, new_y, new_z =_lonlat_rad_to_xyz(lon, lat)

        self.assertLessEqual(np.absolute(new_x - x), err_tolerance)
        self.assertLessEqual(np.absolute(new_y - y), err_tolerance)
        self.assertLessEqual(np.absolute(new_z - z), err_tolerance)

    def test_node_latlon_rad_to_xyz(self):
        [lon, lat] = [
            random.uniform(0, 2 * np.pi),
            random.uniform(-0.5 * np.pi, 0.5 * np.pi)
        ]

        x, y, z = _lonlat_rad_to_xyz(lon, lat)

        new_lon, new_lat = _xyz_to_lonlat_rad(x, y, z)

        self.assertLessEqual(np.absolute(new_lon - lon), err_tolerance)
        self.assertLessEqual(np.absolute(new_lat - lat), err_tolerance)


class TestConstants(TestCase):
    # DTYPE as set in constants.py
    expected_int_dtype = INT_DTYPE

    # INT_FILL_VALUE as set in constants.py
    fv = INT_FILL_VALUE

    def test_invalid_indexing(self):
        """Tests if the current INT_DTYPE and INT_FILL_VALUE throw the correct
        errors when indexing."""
        dummy_data = np.array([1, 2, 3, 4])

        invalid_indices = np.array([self.fv, self.fv], dtype=INT_DTYPE)
        invalid_index = self.fv

        # invalid index/indices should throw an Index Error
        with self.assertRaises(IndexError):
            dummy_data[invalid_indices]
            dummy_data[invalid_index]

    def test_replace_fill_values(self):
        """Tests _replace_fill_values() helper function across multiple
        different dtype arrays used as face_nodes."""

        # expected output from _replace_fill_values()
        face_nodes_gold = np.array(
            [[1, 2, self.fv], [self.fv, self.fv, self.fv]], dtype=INT_DTYPE)

        # test different datatypes for face_nodes
        dtypes = [np.int32, np.int64, np.float32, np.float64]
        for dtype in dtypes:
            # test face nodes with set dtype
            face_nodes = np.array([[1, 2, -1], [-1, -1, -1]], dtype=dtype)

            # output of _replace_fill_values()
            face_nodes_test = _replace_fill_values(grid_var=face_nodes,
                                                   original_fill=-1,
                                                   new_fill=INT_FILL_VALUE,
                                                   new_dtype=INT_DTYPE)

            assert np.array_equal(face_nodes_test, face_nodes_gold)

    def test_replace_fill_values_invalid(self):
        """Tests _replace_fill_values() helper function attempting to use a
        fill value that is not representable by the current dtype."""

        face_nodes = np.array([[1, 2, -1], [-1, -1, -1]], dtype=np.uint32)
        # invalid fill value with dtype should raise a valueError
        with self.assertRaises(ValueError):
            # INT_FILL_VALUE (max(uint32) not representable by int16)
            face_nodes_test = _replace_fill_values(grid_var=face_nodes,
                                                   original_fill=-1,
                                                   new_fill=INT_FILL_VALUE,
                                                   new_dtype=np.int16)


class TestSparseMatrix(TestCase):

    def test_convert_face_node_conn_to_sparse_matrix(self):
        """Tests _face_nodes_to_sparse_matrix() helper function to see if can
        generate sparse matrix from face_nodes_conn that has Fill Values."""
        face_nodes_conn = np.array([[3, 4, 5, INT_FILL_VALUE], [3, 0, 2, 5],
                                    [3, 4, 1, 0], [0, 1, 2, INT_FILL_VALUE]])

        face_indices, nodes_indices, non_zero_flag = ux.grid.connectivity._face_nodes_to_sparse_matrix(
            face_nodes_conn)
        expected_non_zero_flag = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        expected_face_indices = np.array(
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
        expected_nodes_indices = np.array(
            [3, 4, 5, 3, 0, 2, 5, 3, 4, 1, 0, 0, 1, 2])

        nt.assert_array_equal(non_zero_flag, expected_non_zero_flag)
        nt.assert_array_equal(face_indices, expected_face_indices)
        nt.assert_array_equal(nodes_indices, expected_nodes_indices)


class TestOperators(TestCase):

    def test_in_between(self):
        # Test the in_between operator
        self.assertTrue(in_between(0, 1, 2))
        self.assertTrue(in_between(-1, -1.5, -2))


class TestVectorsAngel(TestCase):

    def test_angle_of_2_vectors(self):
        # Test the angle between two vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(_angle_of_2_vectors(v1, v2), np.pi / 2.0)

        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(_angle_of_2_vectors(v1, v2), 0.0)


class TestFaceEdgeConnectivityHelper(TestCase):

    def test_get_cartesian_face_edge_nodes_pipeline(self):
        # Create the vertices for the grid, based around the North Pole
        vertices = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]]

        # Normalize the vertices
        vertices = [x / np.linalg.norm(x) for x in vertices]

        # Construct the grid from the vertices
        grid = ux.Grid.from_face_vertices(vertices, latlon=False)

        # Extract the necessary grid data
        face_node_conn = grid.face_node_connectivity.values
        n_nodes_per_face = np.array([len(face) for face in face_node_conn])
        n_face = len(face_node_conn)

        n_max_face_edges = max(n_nodes_per_face)
        node_x = grid.node_x.values
        node_y = grid.node_y.values
        node_z = grid.node_z.values

        # Call the function to test
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            face_node_conn, n_face, n_max_face_edges, node_x, node_y, node_z
        )

        # Check that the face_edges_connectivity_cartesian works as an input to _pole_point_inside_polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'North', face_edges_connectivity_cartesian[0]
        )

        # Assert that the result is True
        self.assertTrue(result)

    def test_get_cartesian_face_edge_nodes_filled_value(self):
        # Create the vertices for the grid, based around the North Pole
        vertices = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]]

        # Normalize the vertices
        vertices = [x / np.linalg.norm(x) for x in vertices]
        vertices.append([INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE])

        # Construct the grid from the vertices
        grid = ux.Grid.from_face_vertices(vertices, latlon=False)

        # Extract the necessary grid data
        face_node_conn = grid.face_node_connectivity.values
        n_nodes_per_face = np.array([len(face) for face in face_node_conn])
        n_face = len(face_node_conn)
        n_max_face_edges = max(n_nodes_per_face)
        node_x = grid.node_x.values
        node_y = grid.node_y.values
        node_z = grid.node_z.values

        # Call the function to test
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            face_node_conn, n_face, n_max_face_edges, node_x, node_y, node_z
        )

        # Check that the face_edges_connectivity_cartesian works as an input to _pole_point_inside_polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'North', face_edges_connectivity_cartesian[0]
        )

        # Assert that the result is True
        self.assertTrue(result)

    def test_get_cartesian_face_edge_nodes_filled_value2(self):
        # The face vertices order in counter-clockwise
        # face_conn = [[0,1,2],[1,3,4,2]]

        #Each vertex is a 2D vector represent the longitude and latitude in degree. Call the node_lonlat_to_xyz to convert it to 3D vector
        v0_deg = [10,10]
        v1_deg = [15,15]
        v2_deg = [5,15]
        v3_deg = [15,45]
        v4_deg = [5,45]

        # First convert them into radians
        v0_rad = np.deg2rad(v0_deg)
        v1_rad = np.deg2rad(v1_deg)
        v2_rad = np.deg2rad(v2_deg)
        v3_rad = np.deg2rad(v3_deg)
        v4_rad = np.deg2rad(v4_deg)

        # It should look like following when passing in the _get_cartesian_face_edge_nodes
        # [[v0_cart,v1_cart,v2_cart, [Fill_Value,Fill_Value,Fill_Value]],[v1_cart,v3_cart,v4_cart,v2_cart]]
        v0_cart = _lonlat_rad_to_xyz(v0_rad[0],v0_rad[1])
        v1_cart = _lonlat_rad_to_xyz(v1_rad[0],v1_rad[1])
        v2_cart = _lonlat_rad_to_xyz(v2_rad[0],v2_rad[1])
        v3_cart = _lonlat_rad_to_xyz(v3_rad[0],v3_rad[1])
        v4_cart = _lonlat_rad_to_xyz(v4_rad[0],v4_rad[1])

        face_node_conn = np.array([[0, 1, 2, INT_FILL_VALUE],[1, 3, 4, 2]])
        n_face = 2
        n_max_face_edges = 4
        n_nodes_per_face = np.array([len(face) for face in face_node_conn])
        node_x = np.array([v0_cart[0],v1_cart[0],v2_cart[0],v3_cart[0],v4_cart[0]])
        node_y = np.array([v0_cart[1],v1_cart[1],v2_cart[1],v3_cart[1],v4_cart[1]])
        node_z = np.array([v0_cart[2],v1_cart[2],v2_cart[2],v3_cart[2],v4_cart[2]])

        # call the function to test
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            face_node_conn, n_face, n_max_face_edges, node_x, node_y, node_z
        )

        # Define correct result
        correct_result = np.array([
            [
                [
                    [v0_cart[0], v0_cart[1], v0_cart[2]],
                    [v1_cart[0], v1_cart[1], v1_cart[2]]
                ],
                [
                    [v1_cart[0], v1_cart[1], v1_cart[2]],
                    [v2_cart[0], v2_cart[1], v2_cart[2]]
                ],
                [
                    [v2_cart[0], v2_cart[1], v2_cart[2]],
                    [v0_cart[0], v0_cart[1], v0_cart[2]]
                ],
                [
                    [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE],
                    [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE]
                ]
            ],
            [
                [
                    [v1_cart[0], v1_cart[1], v1_cart[2]],
                    [v3_cart[0], v3_cart[1], v3_cart[2]]
                ],
                [
                    [v3_cart[0], v3_cart[1], v3_cart[2]],
                    [v4_cart[0], v4_cart[1], v4_cart[2]]
                ],
                [
                    [v4_cart[0], v4_cart[1], v4_cart[2]],
                    [v2_cart[0], v2_cart[1], v2_cart[2]]
                ],
                [
                    [v2_cart[0], v2_cart[1], v2_cart[2]],
                    [v1_cart[0], v1_cart[1], v1_cart[2]]
                ]
            ]
        ])


        # Assert that the result is correct
        self.assertEqual(face_edges_connectivity_cartesian.shape, correct_result.shape)


    def test_get_lonlat_face_edge_nodes_pipeline(self):
        # Create the vertices for the grid, based around the North Pole
        vertices = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]]

        # Normalize the vertices
        vertices = [x / np.linalg.norm(x) for x in vertices]

        # Construct the grid from the vertices
        grid = ux.Grid.from_face_vertices(vertices, latlon=False)

        # Extract the necessary grid data
        face_node_conn = grid.face_node_connectivity.values
        n_nodes_per_face = np.array([len(face) for face in face_node_conn])
        n_face = len(face_node_conn)
        n_max_face_edges = max(n_nodes_per_face)
        node_lon = grid.node_lon.values
        node_lat = grid.node_lat.values

        # Call the function to test
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            face_node_conn, n_face, n_max_face_edges, node_lon, node_lat
        )

        # Convert the first face's edges to Cartesian coordinates
        face_edges_connectivity_lonlat = face_edges_connectivity_lonlat[0]
        face_edges_connectivity_cartesian = []
        for edge in face_edges_connectivity_lonlat:
            edge_cart = [_lonlat_rad_to_xyz(*node) for node in edge]
            face_edges_connectivity_cartesian.append(edge_cart)

        # Check that the face_edges_connectivity_cartesian works as an input to _pole_point_inside_polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'North', np.array(face_edges_connectivity_cartesian)
        )

        # Assert that the result is True
        self.assertTrue(result)

    def test_get_lonlat_face_edge_nodes_filled_value(self):
        # Create the vertices for the grid, based around the North Pole
        vertices = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]]

        # Normalize the vertices
        vertices = [x / np.linalg.norm(x) for x in vertices]
        vertices.append([INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE])

        # Construct the grid from the vertices
        grid = ux.Grid.from_face_vertices(vertices, latlon=False)

        # Extract the necessary grid data
        face_node_conn = grid.face_node_connectivity.values
        n_nodes_per_face = np.array([len(face) for face in face_node_conn])
        n_face = len(face_node_conn)
        n_max_face_edges = max(n_nodes_per_face)
        node_lon = grid.node_lon.values
        node_lat = grid.node_lat.values

        # Call the function to test
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            face_node_conn, n_face, n_max_face_edges, node_lon, node_lat
        )

        # Convert the first face's edges to Cartesian coordinates
        face_edges_connectivity_lonlat = face_edges_connectivity_lonlat[0]
        face_edges_connectivity_cartesian = []
        for edge in face_edges_connectivity_lonlat:
            edge_cart = [_lonlat_rad_to_xyz(*node) for node in edge]
            face_edges_connectivity_cartesian.append(edge_cart)

        # Check that the face_edges_connectivity_cartesian works as an input to _pole_point_inside_polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'North', np.array(face_edges_connectivity_cartesian)
        )

        # Assert that the result is True
        self.assertTrue(result)


    def test_get_lonlat_face_edge_nodes_filled_value2(self):
        # The face vertices order in counter-clockwise
        # face_conn = [[0,1,2],[1,3,4,2]]

        #Each vertex is a 2D vector represent the longitude and latitude in degree. Call the node_lonlat_to_xyz to convert it to 3D vector
        v0_deg = [10,10]
        v1_deg = [15,15]
        v2_deg = [5,15]
        v3_deg = [15,45]
        v4_deg = [5,45]

        # First convert them into radians
        v0_rad = np.deg2rad(v0_deg)
        v1_rad = np.deg2rad(v1_deg)
        v2_rad = np.deg2rad(v2_deg)
        v3_rad = np.deg2rad(v3_deg)
        v4_rad = np.deg2rad(v4_deg)

        face_node_conn = np.array([[0, 1, 2, INT_FILL_VALUE],[1, 3, 4, 2]])
        n_face = 2
        n_max_face_edges = 4
        n_nodes_per_face = np.array([len(face) for face in face_node_conn])
        node_lon = np.array([v0_rad[0],v1_rad[0],v2_rad[0],v3_rad[0],v4_rad[0]])
        node_lat = np.array([v0_rad[1],v1_rad[1],v2_rad[1],v3_rad[1],v4_rad[1]])

        # call the function to test
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            face_node_conn, n_face, n_max_face_edges, node_lon, node_lat
        )

        # Define correct result
        correct_result = np.array([
            [
                [
                    [v0_rad[0], v0_rad[1]],
                    [v1_rad[0], v1_rad[1]]
                ],
                [
                    [v1_rad[0], v1_rad[1]],
                    [v2_rad[0], v2_rad[1]]
                ],
                [
                    [v2_rad[0], v2_rad[1]],
                    [v0_rad[0], v0_rad[1]]
                ],
                [
                    [INT_FILL_VALUE, INT_FILL_VALUE],
                    [INT_FILL_VALUE, INT_FILL_VALUE]
                ]
            ],
            [
                [
                    [v1_rad[0], v1_rad[1]],
                    [v3_rad[0], v3_rad[1]]
                ],
                [
                    [v3_rad[0], v3_rad[1]],
                    [v4_rad[0], v4_rad[1]]
                ],
                [
                    [v4_rad[0], v4_rad[1]],
                    [v2_rad[0], v2_rad[1]]
                ],
                [
                    [v2_rad[0], v2_rad[1]],
                    [v1_rad[0], v1_rad[1]]
                ]
            ]
        ])

        # Assert that the result is correct
        self.assertEqual(face_edges_connectivity_lonlat.shape, correct_result.shape)
