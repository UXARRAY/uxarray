import os
import numpy as np
import numpy.testing as nt
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
import uxarray.utils.computing as ac_utils
from uxarray.grid.coordinates import _populate_node_latlon, _lonlat_rad_to_xyz, _normalize_xyz, _xyz_to_lonlat_rad
from uxarray.grid.arcs import extreme_gca_latitude
from uxarray.grid.utils import _get_cartesian_face_edge_nodes, _get_lonlat_rad_face_edge_nodes
from uxarray.grid.geometry import _populate_face_latlon_bound, _populate_bounds

from spatialpandas.geometry import MultiPolygon

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_CSne8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
datafile_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"

gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
datafile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"

grid_files = [gridfile_CSne8, gridfile_geoflow]
data_files = [datafile_CSne30, datafile_geoflow]


class TestAntimeridian(TestCase):

    def test_crossing(self):
        verts = [[[-170, 40], [180, 30], [165, 25], [-170, 20]]]

        uxgrid = ux.open_grid(verts, latlon=True)

        gdf = uxgrid.to_geodataframe()

        assert len(uxgrid.antimeridian_face_indices) == 1

        assert len(gdf['geometry']) == 1

    def test_point_on(self):
        verts = [[[-170, 40], [180, 30], [-170, 20]]]

        uxgrid = ux.open_grid(verts, latlon=True)

        assert len(uxgrid.antimeridian_face_indices) == 1


class TestLineCollection(TestCase):

    def test_linecollection_execution(self):
        uxgrid = ux.open_grid(gridfile_CSne8)
        lines = uxgrid.to_linecollection()


class TestPredicate(TestCase):

    def test_pole_point_inside_polygon_from_vertice_north(self):
        # Define a face as a list of vertices on the unit sphere
        # Here, we're defining a square like structure around the North pole
        vertices = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5]]

        # Normalize the vertices to ensure they lie on the unit sphere
        for i, vertex in enumerate(vertices):
            float_vertex = [float(coord) for coord in vertex]
            vertices[i] = _normalize_xyz(*float_vertex)

        # Create face_edge_cart from the vertices
        face_edge_cart = np.array([[vertices[0], vertices[1]],
                                   [vertices[1], vertices[2]],
                                   [vertices[2], vertices[3]],
                                   [vertices[3], vertices[0]]])

        # Check if the North pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'North', face_edge_cart)
        self.assertTrue(result, "North pole should be inside the polygon")

        # Check if the South pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'South', face_edge_cart)
        self.assertFalse(result, "South pole should not be inside the polygon")

    def test_pole_point_inside_polygon_from_vertice_south(self):
        # Define a face as a list of vertices on the unit sphere
        # Here, we're defining a square like structure around the south pole
        vertices = [[0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [0.0, 0.0, -1.0]]

        # Normalize the vertices to ensure they lie on the unit sphere
        for i, vertex in enumerate(vertices):
            float_vertex = [float(coord) for coord in vertex]
            vertices[i] = _normalize_xyz(*float_vertex)

        # Create face_edge_cart from the vertices, since we are using the south pole, and want retrive the smaller face
        # we need to reverse the order of the vertices
        # Create face_edge_cart from the vertices
        face_edge_cart = np.array([[vertices[0], vertices[1]],
                                   [vertices[1], vertices[2]],
                                   [vertices[2], vertices[0]]])

        # Check if the North pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'North', face_edge_cart)
        self.assertFalse(result, "North pole should not be inside the polygon")

        # Check if the South pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'South', face_edge_cart)
        self.assertTrue(result, "South pole should be inside the polygon")

    def test_pole_point_inside_polygon_from_vertice_pole(self):
        # Define a face as a list of vertices on the unit sphere
        # Here, we're defining a square like structure that pole is on the edge
        vertices = [[0, 0, 1], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5]]

        # Normalize the vertices to ensure they lie on the unit sphere
        for i, vertex in enumerate(vertices):
            float_vertex = [float(coord) for coord in vertex]
            vertices[i] = _normalize_xyz(*float_vertex)

        # Create face_edge_cart from the vertices
        face_edge_cart = np.array([[vertices[0], vertices[1]],
                                   [vertices[1], vertices[2]],
                                   [vertices[2], vertices[3]],
                                   [vertices[3], vertices[0]]])

        # Check if the North pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'North', face_edge_cart)
        self.assertTrue(result, "North pole should be inside the polygon")

        # Check if the South pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'South', face_edge_cart)
        self.assertFalse(result, "South pole should not be inside the polygon")

    def test_pole_point_inside_polygon_from_vertice_cross(self):
        # Define a face that crosses the equator and ecompasses the North pole
        vertices = [[0.6, -0.3, 0.5], [0.2, 0.2, -0.2], [-0.5, 0.1, -0.2],
                    [-0.1, -0.2, 0.2]]

        # Normalize the vertices to ensure they lie on the unit sphere
        for i, vertex in enumerate(vertices):
            float_vertex = [float(coord) for coord in vertex]
            vertices[i] = _normalize_xyz(*float_vertex)

        # Create face_edge_cart from the vertices
        face_edge_cart = np.array([[vertices[0], vertices[1]],
                                   [vertices[1], vertices[2]],
                                   [vertices[2], vertices[3]],
                                   [vertices[3], vertices[0]]])

        # Check if the North pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'North', face_edge_cart)
        self.assertTrue(result, "North pole should be inside the polygon")


class TestLatlonBoundUtils(TestCase):

    def _max_latitude_rad_iterative(self, gca_cart):
        """Calculate the maximum latitude of a great circle arc defined by two
        points.

        Parameters
        ----------
        gca_cart : numpy.ndarray
            An array containing two 3D vectors that define a great circle arc.

        Returns
        -------
        float
            The maximum latitude of the great circle arc in radians.

        Raises
        ------
        ValueError
            If the input vectors are not valid 2-element lists or arrays.

        Notes
        -----
        The method divides the great circle arc into subsections, iteratively refining the subsection of interest
        until the maximum latitude is found within a specified tolerance.
        """

        # Convert input vectors to radians and Cartesian coordinates

        v1_cart, v2_cart = gca_cart
        b_lonlat = _xyz_to_lonlat_rad(*v1_cart.tolist())
        c_lonlat = _xyz_to_lonlat_rad(*v2_cart.tolist())

        # Initialize variables for the iterative process
        v_temp = ac_utils.cross_fma(v1_cart, v2_cart)
        v0 = ac_utils.cross_fma(v_temp, v1_cart)
        v0 = _normalize_xyz(*v0.tolist())
        max_section = [v1_cart, v2_cart]

        # Iteratively find the maximum latitude
        while np.abs(b_lonlat[1] - c_lonlat[1]) >= ERROR_TOLERANCE or np.abs(
                b_lonlat[0] - c_lonlat[0]) >= ERROR_TOLERANCE:
            max_lat = -np.pi
            v_b, v_c = max_section
            angle_v1_v2_rad = ux.grid.arcs._angle_of_2_vectors(v_b, v_c)
            v0 = ac_utils.cross_fma(v_temp, v_b)
            v0 = _normalize_xyz(*v0.tolist())
            avg_angle_rad = angle_v1_v2_rad / 10.0

            for i in range(10):
                angle_rad_prev = avg_angle_rad * i
                angle_rad_next = angle_rad_prev + avg_angle_rad if i < 9 else angle_v1_v2_rad
                w1_new = np.cos(angle_rad_prev) * v_b + np.sin(
                    angle_rad_prev) * np.array(v0)
                w2_new = np.cos(angle_rad_next) * v_b + np.sin(
                    angle_rad_next) * np.array(v0)
                w1_lonlat = _xyz_to_lonlat_rad(
                    *w1_new.tolist())
                w2_lonlat = _xyz_to_lonlat_rad(
                    *w2_new.tolist())

                w1_lonlat = np.asarray(w1_lonlat)
                w2_lonlat = np.asarray(w2_lonlat)

                # Adjust latitude boundaries to avoid error accumulation
                if i == 0:
                    w1_lonlat[1] = b_lonlat[1]
                elif i >= 9:
                    w2_lonlat[1] = c_lonlat[1]

                # Update maximum latitude and section if needed
                max_lat = max(max_lat, w1_lonlat[1], w2_lonlat[1])
                if np.abs(w2_lonlat[1] -
                          w1_lonlat[1]) <= ERROR_TOLERANCE or w1_lonlat[
                    1] == max_lat == w2_lonlat[1]:
                    max_section = [w1_new, w2_new]
                    break
                if np.abs(max_lat - w1_lonlat[1]) <= ERROR_TOLERANCE:
                    max_section = [w1_new, w2_new] if i != 0 else [v_b, w2_new]
                elif np.abs(max_lat - w2_lonlat[1]) <= ERROR_TOLERANCE:
                    max_section = [w1_new, w2_new] if i != 9 else [w1_new, v_c]

            # Update longitude and latitude for the next iteration
            b_lonlat = _xyz_to_lonlat_rad(
                *max_section[0].tolist())
            c_lonlat = _xyz_to_lonlat_rad(
                *max_section[1].tolist())

        return np.average([b_lonlat[1], c_lonlat[1]])

    def _min_latitude_rad_iterative(self, gca_cart):
        """Calculate the minimum latitude of a great circle arc defined by two
        points.

        Parameters
        ----------
        gca_cart : numpy.ndarray
            An array containing two 3D vectors that define a great circle arc.

        Returns
        -------
        float
            The minimum latitude of the great circle arc in radians.

        Raises
        ------
        ValueError
            If the input vectors are not valid 2-element lists or arrays.

        Notes
        -----
        The method divides the great circle arc into subsections, iteratively refining the subsection of interest
        until the minimum latitude is found within a specified tolerance.
        """

        # Convert input vectors to radians and Cartesian coordinates
        v1_cart, v2_cart = gca_cart
        b_lonlat = _xyz_to_lonlat_rad(*v1_cart.tolist())
        c_lonlat = _xyz_to_lonlat_rad(*v2_cart.tolist())

        # Initialize variables for the iterative process
        v_temp = ac_utils.cross_fma(v1_cart, v2_cart)
        v0 = ac_utils.cross_fma(v_temp, v1_cart)
        v0 = np.array(_normalize_xyz(*v0.tolist()))
        min_section = [v1_cart, v2_cart]

        # Iteratively find the minimum latitude
        while np.abs(b_lonlat[1] - c_lonlat[1]) >= ERROR_TOLERANCE or np.abs(
                b_lonlat[0] - c_lonlat[0]) >= ERROR_TOLERANCE:
            min_lat = np.pi
            v_b, v_c = min_section
            angle_v1_v2_rad = ux.grid.arcs._angle_of_2_vectors(v_b, v_c)
            v0 = ac_utils.cross_fma(v_temp, v_b)
            v0 = np.array(_normalize_xyz(*v0.tolist()))
            avg_angle_rad = angle_v1_v2_rad / 10.0

            for i in range(10):
                angle_rad_prev = avg_angle_rad * i
                angle_rad_next = angle_rad_prev + avg_angle_rad if i < 9 else angle_v1_v2_rad
                w1_new = np.cos(angle_rad_prev) * v_b + np.sin(
                    angle_rad_prev) * v0
                w2_new = np.cos(angle_rad_next) * v_b + np.sin(
                    angle_rad_next) * v0
                w1_lonlat = _xyz_to_lonlat_rad(
                    *w1_new.tolist())
                w2_lonlat = _xyz_to_lonlat_rad(
                    *w2_new.tolist())

                w1_lonlat = np.asarray(w1_lonlat)
                w2_lonlat = np.asarray(w2_lonlat)

                # Adjust latitude boundaries to avoid error accumulation
                if i == 0:
                    w1_lonlat[1] = b_lonlat[1]
                elif i >= 9:
                    w2_lonlat[1] = c_lonlat[1]

                # Update minimum latitude and section if needed
                min_lat = min(min_lat, w1_lonlat[1], w2_lonlat[1])
                if np.abs(w2_lonlat[1] -
                          w1_lonlat[1]) <= ERROR_TOLERANCE or w1_lonlat[
                    1] == min_lat == w2_lonlat[1]:
                    min_section = [w1_new, w2_new]
                    break
                if np.abs(min_lat - w1_lonlat[1]) <= ERROR_TOLERANCE:
                    min_section = [w1_new, w2_new] if i != 0 else [v_b, w2_new]
                elif np.abs(min_lat - w2_lonlat[1]) <= ERROR_TOLERANCE:
                    min_section = [w1_new, w2_new] if i != 9 else [w1_new, v_c]

            # Update longitude and latitude for the next iteration
            b_lonlat = _xyz_to_lonlat_rad(
                *min_section[0].tolist())
            c_lonlat = _xyz_to_lonlat_rad(
                *min_section[1].tolist())

        return np.average([b_lonlat[1], c_lonlat[1]])

    def test_extreme_gca_latitude_max(self):
        # Define a great circle arc that is symmetrical around 0 degrees longitude
        gca_cart = np.array([
            _normalize_xyz(*[0.5, 0.5, 0.5]),
            _normalize_xyz(*[-0.5, 0.5, 0.5])
        ])

        # Calculate the maximum latitude
        max_latitude = ux.grid.arcs.extreme_gca_latitude(gca_cart, 'max')

        # Check if the maximum latitude is correct
        expected_max_latitude = self._max_latitude_rad_iterative(gca_cart)
        self.assertAlmostEqual(max_latitude,
                               expected_max_latitude,
                               delta=ERROR_TOLERANCE)

        # Define a great circle arc in 3D space
        gca_cart = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

        # Calculate the maximum latitude
        max_latitude = ux.grid.arcs.extreme_gca_latitude(gca_cart, 'max')

        # Check if the maximum latitude is correct
        expected_max_latitude = np.pi / 2  # 90 degrees in radians
        self.assertAlmostEqual(max_latitude,
                               expected_max_latitude,
                               delta=ERROR_TOLERANCE)

    def test_extreme_gca_latitude_min(self):
        # Define a great circle arc that is symmetrical around 0 degrees longitude
        gca_cart = np.array([
            _normalize_xyz(*[0.5, 0.5, -0.5]),
            _normalize_xyz(*[-0.5, 0.5, -0.5])
        ])

        # Calculate the minimum latitude
        min_latitude = ux.grid.arcs.extreme_gca_latitude(gca_cart, 'min')

        # Check if the minimum latitude is correct
        expected_min_latitude = self._min_latitude_rad_iterative(gca_cart)
        self.assertAlmostEqual(min_latitude,
                               expected_min_latitude,
                               delta=ERROR_TOLERANCE)

        # Define a great circle arc in 3D space
        gca_cart = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

        # Calculate the minimum latitude
        min_latitude = ux.grid.arcs.extreme_gca_latitude(gca_cart, 'min')

        # Check if the minimum latitude is correct
        expected_min_latitude = -np.pi / 2  # 90 degrees in radians
        self.assertAlmostEqual(min_latitude,
                               expected_min_latitude,
                               delta=ERROR_TOLERANCE)

    def test_get_latlonbox_width(self):
        # Define a great circle arc that is not wrapping around the meridian
        gca_latlon = np.array([[0.0, 0.0], [0.0, 3.0]])

        # Calculate the width of the latlonbox
        width = ux.grid.geometry._get_latlonbox_width(gca_latlon)
        self.assertEqual(width, 3.0)

        # Define a great circle arc that is not wrapping around the meridian
        gca_latlon = np.array([[0.0, 0.0], [2 * np.pi - 1.0, 1.0]])
        width = ux.grid.geometry._get_latlonbox_width(gca_latlon)
        self.assertEqual(width, 2.0)

    def test_insert_pt_in_latlonbox_non_periodic(self):
        old_box = np.array([[0.1, 0.2], [0.3, 0.4]])  # Radians
        new_pt = np.array([0.15, 0.35])
        expected = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = ux.grid.geometry._insert_pt_in_latlonbox(
            old_box, new_pt, False)
        np.testing.assert_array_equal(result, expected)

    def test_insert_pt_in_latlonbox_periodic(self):
        old_box = np.array([[0.1, 0.2], [6.0, 0.1]])  # Radians, periodic
        new_pt = np.array([0.15, 6.2])
        expected = np.array([[0.1, 0.2], [6.0, 0.1]])
        result = ux.grid.geometry._insert_pt_in_latlonbox(old_box, new_pt, True)
        np.testing.assert_array_equal(result, expected)

    def test_insert_pt_in_latlonbox_pole(self):
        old_box = np.array([[0.1, 0.2], [0.3, 0.4]])
        new_pt = np.array([np.pi / 2, INT_FILL_VALUE])  # Pole point
        expected = np.array([[0.1, np.pi / 2], [0.3, 0.4]])
        result = ux.grid.geometry._insert_pt_in_latlonbox(old_box, new_pt)
        np.testing.assert_array_equal(result, expected)

    def test_insert_pt_in_empty_state(self):
        old_box = np.array([[INT_FILL_VALUE, INT_FILL_VALUE],
                            [INT_FILL_VALUE, INT_FILL_VALUE]])  # Empty state
        new_pt = np.array([0.15, 0.35])
        expected = np.array([[0.15, 0.15], [0.35, 0.35]])
        result = ux.grid.geometry._insert_pt_in_latlonbox(old_box, new_pt)
        np.testing.assert_array_equal(result, expected)


class TestLatlonBoundsGCA(TestCase):
    def test_populate_bounds_normal(self):
        # Generate a normal face that is not crossing the antimeridian or the poles
        vertices_lonlat = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)

        vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
        # vertices_cart = [_lonlat_rad_to_xyz(vertices_rad[0], vertices_rad[1])]
        lat_max = max(np.deg2rad(60.0),
                      extreme_gca_latitude(np.array([vertices_cart[0], vertices_cart[3]]), extreme_type="max"))
        lat_min = min(np.deg2rad(10.0),
                      extreme_gca_latitude(np.array([vertices_cart[1], vertices_cart[2]]), extreme_type="min"))
        lon_min = np.deg2rad(10.0)
        lon_max = np.deg2rad(50.0)
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)

    def test_populate_bounds_antimeridian(self):
        # Generate a normal face that is crossing the antimeridian
        vertices_lonlat = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
        lat_max = max(np.deg2rad(60.0),
                      extreme_gca_latitude(np.array([vertices_cart[0], vertices_cart[3]]), extreme_type="max"))
        lat_min = min(np.deg2rad(10.0),
                      extreme_gca_latitude(np.array([vertices_cart[1], vertices_cart[2]]), extreme_type="min"))
        lon_min = np.deg2rad(350.0)
        lon_max = np.deg2rad(50.0)
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)

    def test_populate_bounds_node_on_pole(self):
        # Generate a normal face that is crossing the antimeridian
        vertices_lonlat = [[10.0, 90.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
        lat_max = np.pi / 2
        lat_min = min(np.deg2rad(10.0),
                      extreme_gca_latitude(np.array([vertices_cart[1], vertices_cart[2]]), extreme_type="min"))
        lon_min = np.deg2rad(10.0)
        lon_max = np.deg2rad(50.0)
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)

    def test_populate_bounds_edge_over_pole(self):
        # Generate a normal face that is crossing the antimeridian
        vertices_lonlat = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
        lat_max = np.pi / 2
        lat_min = min(np.deg2rad(60.0),
                      extreme_gca_latitude(np.array([vertices_cart[1], vertices_cart[2]]), extreme_type="min"))
        lon_min = np.deg2rad(210.0)
        lon_max = np.deg2rad(30.0)
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)

    def test_populate_bounds_pole_inside(self):
        # Generate a normal face that is crossing the antimeridian
        vertices_lonlat = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
        lat_max = np.pi / 2
        lat_min = min(np.deg2rad(60.0),
                      extreme_gca_latitude(np.array([vertices_cart[1], vertices_cart[2]]), extreme_type="min"))
        lon_min = 0
        lon_max = 2 * np.pi
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


class TestLatlonBoundsLatLonFace(TestCase):

    def test_populate_bounds_normal(self):
        # Generate a normal face that is not crossing the antimeridian or the poles
        vertices_lonlat = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        lat_max = np.deg2rad(60.0)
        lat_min = np.deg2rad(10.0)
        lon_min = np.deg2rad(10.0)
        lon_max = np.deg2rad(50.0)
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                             is_latlonface=True)
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)

    def test_populate_bounds_antimeridian(self):
        # Generate a normal face that is crossing the antimeridian
        vertices_lonlat = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        lat_max = np.deg2rad(60.0)
        lat_min = np.deg2rad(10.0)
        lon_min = np.deg2rad(350.0)
        lon_max = np.deg2rad(50.0)
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                             is_latlonface=True)
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)

    def test_populate_bounds_node_on_pole(self):
        # Generate a normal face that is crossing the antimeridian
        vertices_lonlat = [[10.0, 90.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        lat_max = np.pi / 2
        lat_min = np.deg2rad(10.0)
        lon_min = np.deg2rad(10.0)
        lon_max = np.deg2rad(50.0)
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                             is_latlonface=True)
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)

    def test_populate_bounds_edge_over_pole(self):
        # Generate a normal face that is crossing the antimeridian
        vertices_lonlat = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
        lat_max = np.pi / 2
        lat_min = np.deg2rad(60.0)
        lon_min = np.deg2rad(210.0)
        lon_max = np.deg2rad(30.0)
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                             is_latlonface=True)
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)

    def test_populate_bounds_pole_inside(self):
        # Generate a normal face that is crossing the antimeridian
        vertices_lonlat = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
        lat_max = np.pi / 2
        lat_min = np.deg2rad(60.0)
        lon_min = 0
        lon_max = 2 * np.pi
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                             is_latlonface=True)
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


class TestLatlonBoundsGCAList(TestCase):

    def test_populate_bounds_normal(self):
        # Generate a normal face that is not crossing the antimeridian or the poles
        vertices_lonlat = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        lat_max = np.deg2rad(60.0)
        lat_min = np.deg2rad(10.0)
        lon_min = np.deg2rad(10.0)
        lon_max = np.deg2rad(50.0)
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                             is_GCA_list=[True, False, True, False])
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)

    def test_populate_bounds_antimeridian(self):
        # Generate a normal face that is crossing the antimeridian
        vertices_lonlat = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
        lat_max = np.deg2rad(60.0)
        lat_min = np.deg2rad(10.0)
        lon_min = np.deg2rad(350.0)
        lon_max = np.deg2rad(50.0)
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                             is_GCA_list=[True, False, True, False])
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)

    def test_populate_bounds_node_on_pole(self):
        # Generate a normal face that is crossing the antimeridian
        vertices_lonlat = [[10.0, 90.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
        lat_max = np.pi / 2
        lat_min = np.deg2rad(10.0)
        lon_min = np.deg2rad(10.0)
        lon_max = np.deg2rad(50.0)
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                             is_GCA_list=[True, False, True, False])
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)

    def test_populate_bounds_edge_over_pole(self):
        # Generate a normal face that is crossing the antimeridian
        vertices_lonlat = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
        lat_max = np.pi / 2
        lat_min = np.deg2rad(60.0)
        lon_min = np.deg2rad(210.0)
        lon_max = np.deg2rad(30.0)
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                             is_GCA_list=[True, False, True, False])
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)

    def test_populate_bounds_pole_inside(self):
        # Generate a normal face that is crossing the antimeridian
        vertices_lonlat = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]
        vertices_lonlat = np.array(vertices_lonlat)

        # Convert everything into radians
        vertices_rad = np.radians(vertices_lonlat)
        vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
        lat_max = np.pi / 2
        lat_min = np.deg2rad(60.0)
        lon_min = 0
        lon_max = 2 * np.pi
        grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
        face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_x.values,
            grid.node_y.values, grid.node_z.values)
        face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes(
            grid.face_node_connectivity.values[0],
            grid.face_edge_connectivity.values[0],
            grid.edge_node_connectivity.values, grid.node_lon.values,
            grid.node_lat.values)
        expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
        bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                             is_GCA_list=[True, False, True, False])
        nt.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


class TestLatlonBoundsMix(TestCase):
    def test_populate_bounds_GCA_mix(self):
        face_1 = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
        face_2 = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
        face_3 = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
        face_4 = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]

        faces = [face_1, face_2, face_3, face_4]

        # Hand calculated bounds for the above faces in radians
        expected_bounds = [[[0.17453293, 1.07370494], [0.17453293, 0.87266463]],
                           [[0.17453293, 1.10714872], [6.10865238, 0.87266463]],
                           [[1.04719755, 1.57079633], [3.66519143, 0.52359878]],
                           [[1.04719755, 1.57079633], [0., 6.28318531]]]

        grid = ux.Grid.from_face_vertices(faces, latlon=True)
        face_bounds = grid.bounds.values
        for i in range(len(faces)):
            nt.assert_allclose(face_bounds[i], expected_bounds[i], atol=ERROR_TOLERANCE)

    def test_populate_bounds_LatlonFace_mix(self):
        face_1 = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
        face_2 = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
        face_3 = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
        face_4 = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]

        faces = [face_1, face_2, face_3, face_4]

        expected_bounds = [[[np.deg2rad(10.0), np.deg2rad(60.0)], [np.deg2rad(10.0), np.deg2rad(50.0)]],
                           [[np.deg2rad(10.0), np.deg2rad(60.0)], [np.deg2rad(350.0), np.deg2rad(50.0)]],
                           [[np.deg2rad(60.0), np.pi / 2], [np.deg2rad(210.0), np.deg2rad(30.0)]],
                           [[np.deg2rad(60.0), np.pi / 2], [0., 2 * np.pi]]]

        grid = ux.Grid.from_face_vertices(faces, latlon=True)
        bounds_xarray = _populate_bounds(grid, is_latlonface=True, return_array=True)
        face_bounds = bounds_xarray.values
        for i in range(len(faces)):
            nt.assert_allclose(face_bounds[i], expected_bounds[i], atol=ERROR_TOLERANCE)

    def test_populate_bounds_GCAList_mix(self):
        face_1 = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
        face_2 = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
        face_3 = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
        face_4 = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]

        faces = [face_1, face_2, face_3, face_4]

        expected_bounds = [[[np.deg2rad(10.0), np.deg2rad(60.0)], [np.deg2rad(10.0), np.deg2rad(50.0)]],
                           [[np.deg2rad(10.0), np.deg2rad(60.0)], [np.deg2rad(350.0), np.deg2rad(50.0)]],
                           [[np.deg2rad(60.0), np.pi / 2], [np.deg2rad(210.0), np.deg2rad(30.0)]],
                           [[np.deg2rad(60.0), np.pi / 2], [0., 2 * np.pi]]]

        grid = ux.Grid.from_face_vertices(faces, latlon=True)
        bounds_xarray = _populate_bounds(grid, is_face_GCA_list=[[True, False, True, False]] * 4, return_array=True)
        face_bounds = bounds_xarray.values
        for i in range(len(faces)):
            nt.assert_allclose(face_bounds[i], expected_bounds[i], atol=ERROR_TOLERANCE)


class TestGeoDataFrame(TestCase):

    def test_to_gdf(self):
        uxgrid = ux.open_grid(gridfile_geoflow)

        gdf_with_am = uxgrid.to_geodataframe(exclude_antimeridian=False)

        gdf_without_am = uxgrid.to_geodataframe(exclude_antimeridian=True)

    def test_cache_and_override(self):
        """Tests the cache and override functionality for GeoDataFrame
        conversion."""

        uxgrid = ux.open_grid(gridfile_geoflow)

        gdf_a = uxgrid.to_geodataframe(exclude_antimeridian=False)

        gdf_b = uxgrid.to_geodataframe(exclude_antimeridian=False)

        assert gdf_a is gdf_b

        gdf_c = uxgrid.to_geodataframe(exclude_antimeridian=True)

        assert gdf_a is not gdf_c

        gdf_d = uxgrid.to_geodataframe(exclude_antimeridian=True)

        assert gdf_d is gdf_c

        gdf_e = uxgrid.to_geodataframe(exclude_antimeridian=True,
                                       override=True,
                                       cache=False)

        assert gdf_d is not gdf_e

        gdf_f = uxgrid.to_geodataframe(exclude_antimeridian=True)

        assert gdf_f is not gdf_e
