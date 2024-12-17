import uxarray as ux
import pytest
import numpy as np
from pathlib import Path
import os

import numpy.testing as nt

# Define the current path and file paths for grid and data
current_path = Path(os.path.dirname(os.path.realpath(__file__)))
quad_hex_grid_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'grid.nc'
quad_hex_data_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'data.nc'

cube_sphere_grid = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"

from uxarray.grid.intersections import constant_lat_intersections_face_bounds



class TestQuadHex:
    """The quad hexagon grid contains four faces.

    Top Left Face: Index 1

    Top Right Face: Index 2

    Bottom Left Face: Index 0

    Bottom Right Face: Index 3

    The top two faces intersect a constant latitude of 0.1

    The bottom two faces intersect a constant latitude of -0.1

    All four faces intersect a constant latitude of 0.0
    """

    def test_constant_lat_cross_section_grid(self):

        uxgrid = ux.open_grid(quad_hex_grid_path)

        grid_top_two = uxgrid.cross_section.constant_latitude(lat=0.1, )

        assert grid_top_two.n_face == 2

        grid_bottom_two = uxgrid.cross_section.constant_latitude(lat=-0.1, )

        assert grid_bottom_two.n_face == 2

        grid_all_four = uxgrid.cross_section.constant_latitude(lat=0.0, )

        assert grid_all_four.n_face == 4

        with pytest.raises(ValueError):
            # no intersections found at this line
            uxgrid.cross_section.constant_latitude(lat=10.0, )

    def test_constant_lon_cross_section_grid(self):
        uxgrid = ux.open_grid(quad_hex_grid_path)

        grid_left_two = uxgrid.cross_section.constant_longitude(lon=-0.1, )

        assert grid_left_two.n_face == 2

        grid_right_two = uxgrid.cross_section.constant_longitude(lon=0.2, )

        assert grid_right_two.n_face == 2

        with pytest.raises(ValueError):
            # no intersections found at this line
            uxgrid.cross_section.constant_longitude(lon=10.0)

    def test_constant_lat_cross_section_uxds(self):
        uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
        uxds.uxgrid.normalize_cartesian_coordinates()

        da_top_two = uxds['t2m'].cross_section.constant_latitude(lat=0.1, )

        nt.assert_array_equal(da_top_two.data, uxds['t2m'].isel(n_face=[1, 2]).data)

        da_bottom_two = uxds['t2m'].cross_section.constant_latitude(lat=-0.1, )

        nt.assert_array_equal(da_bottom_two.data, uxds['t2m'].isel(n_face=[0, 3]).data)

        da_all_four = uxds['t2m'].cross_section.constant_latitude(lat=0.0, )

        nt.assert_array_equal(da_all_four.data , uxds['t2m'].data)

        with pytest.raises(ValueError):
            # no intersections found at this line
            uxds['t2m'].cross_section.constant_latitude(lat=10.0, )


    def test_constant_lon_cross_section_uxds(self):
        uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
        uxds.uxgrid.normalize_cartesian_coordinates()

        da_left_two = uxds['t2m'].cross_section.constant_longitude(lon=-0.1, )

        nt.assert_array_equal(da_left_two.data, uxds['t2m'].isel(n_face=[0, 2]).data)

        da_right_two = uxds['t2m'].cross_section.constant_longitude(lon=0.2, )

        nt.assert_array_equal(da_right_two.data, uxds['t2m'].isel(n_face=[1, 3]).data)

        with pytest.raises(ValueError):
            # no intersections found at this line
            uxds['t2m'].cross_section.constant_longitude(lon=10.0, )


class TestCubeSphere:

    def test_north_pole(self):
        uxgrid = ux.open_grid(cube_sphere_grid)

        lats = [89.85, 89.9, 89.95, 89.99]

        for lat in lats:
            cross_grid = uxgrid.cross_section.constant_latitude(lat=lat, )
            # Cube sphere grid should have 4 faces centered around the pole
            # assert cross_grid.n_face == 4

    def test_south_pole(self):
        uxgrid = ux.open_grid(cube_sphere_grid)

        lats = [-89.85, -89.9, -89.95, -89.99]

        for lat in lats:
            cross_grid = uxgrid.cross_section.constant_latitude(lat=lat, )
            # Cube sphere grid should have 4 faces centered around the pole
            assert cross_grid.n_face == 4



class TestCandidateFacesUsingBounds:

    def test_constant_lat(self):
        bounds = np.array([
            [[-45, 45], [0, 360]],
            [[-90, -45], [0, 360]],
            [[45, 90], [0, 360]],
        ])

        bounds_rad = np.deg2rad(bounds)

        const_lat = 0

        candidate_faces = constant_lat_intersections_face_bounds(
            lat=const_lat,
            face_bounds_lat=bounds_rad[:, 0],
        )

        # Expected output
        expected_faces = np.array([0])

        # Test the function output
        nt.assert_array_equal(candidate_faces, expected_faces)

    def test_constant_lat_out_of_bounds(self):

        bounds = np.array([
            [[-45, 45], [0, 360]],
            [[-90, -45], [0, 360]],
            [[45, 90], [0, 360]],
        ])

        bounds_rad = np.deg2rad(bounds)

        const_lat = 100

        candidate_faces = constant_lat_intersections_face_bounds(
            lat=const_lat,
            face_bounds_lat=bounds_rad[:, 0],
        )

        assert len(candidate_faces) == 0


class TestArcs:
    def test_latitude_along_arc(self):
        node_lon = np.array([-40, -40, 40, 40])
        node_lat = np.array([-20, 20, 20, -20])
        face_node_connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)

        uxgrid = ux.Grid.from_topology(node_lon, node_lat, face_node_connectivity)

        # intersection at exactly 20 degrees latitude
        out1 = uxgrid.get_faces_at_constant_latitude(lat=20)

        # intersection at 25.41 degrees latitude (max along the great circle arc)
        out2 = uxgrid.get_faces_at_constant_latitude(lat=25.41)

        nt.assert_array_equal(out1, out2)
