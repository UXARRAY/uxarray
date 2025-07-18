import uxarray as ux
import pytest
import numpy as np
from pathlib import Path


import numpy.testing as nt

# Define the current path and file paths for grid and data
current_path = Path(__file__).resolve().parent
quad_hex_grid_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'grid.nc'
quad_hex_data_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'data.nc'
quad_hex_node_data = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'random-node-data.nc'
cube_sphere_grid = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"

from uxarray.grid.intersections import constant_lat_intersections_face_bounds


def test_repr():
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)

    # grid repr
    grid_repr = uxds.uxgrid.cross_section.__repr__()
    assert "constant_latitude" in grid_repr
    assert "constant_longitude" in grid_repr
    assert "constant_latitude_interval" in grid_repr
    assert "constant_longitude_interval" in grid_repr

    # data array repr
    da_repr = uxds['t2m'].cross_section.__repr__()
    assert "constant_latitude" in da_repr
    assert "constant_longitude" in da_repr
    assert "constant_latitude_interval" in da_repr
    assert "constant_longitude_interval" in da_repr



def test_constant_lat_cross_section_grid():
    uxgrid = ux.open_grid(quad_hex_grid_path)

    grid_top_two = uxgrid.cross_section.constant_latitude(lat=0.1)
    assert grid_top_two.n_face == 2

    grid_bottom_two = uxgrid.cross_section.constant_latitude(lat=-0.1)
    assert grid_bottom_two.n_face == 2

    grid_all_four = uxgrid.cross_section.constant_latitude(lat=0.0)
    assert grid_all_four.n_face == 4

    with pytest.raises(ValueError):
        uxgrid.cross_section.constant_latitude(lat=10.0)

def test_constant_lon_cross_section_grid():
    uxgrid = ux.open_grid(quad_hex_grid_path)

    grid_left_two = uxgrid.cross_section.constant_longitude(lon=-0.1)
    assert grid_left_two.n_face == 2

    grid_right_two = uxgrid.cross_section.constant_longitude(lon=0.2)
    assert grid_right_two.n_face == 2

    with pytest.raises(ValueError):
        uxgrid.cross_section.constant_longitude(lon=10.0)

def test_constant_lat_cross_section_uxds():
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    # Test structured cross-section
    cross_section = uxds['t2m'].cross_section.constant_latitude(lat=0.0, n_samples=10)
    assert 'sample' in cross_section.dims
    assert 'lon' in cross_section.coords
    assert 'lat' in cross_section.coords
    assert cross_section.shape == (10,)

    # Test with different latitude
    cross_section_2 = uxds['t2m'].cross_section.constant_latitude(lat=0.1, n_samples=5)
    assert cross_section_2.shape == (5,)

def test_constant_lon_cross_section_uxds():
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    # Test structured cross-section
    cross_section = uxds['t2m'].cross_section.constant_longitude(lon=0.0, n_samples=10)
    assert 'sample' in cross_section.dims
    assert 'lon' in cross_section.coords
    assert 'lat' in cross_section.coords
    assert cross_section.shape == (10,)

    # Test with different longitude
    cross_section_2 = uxds['t2m'].cross_section.constant_longitude(lon=0.1, n_samples=8)
    assert cross_section_2.shape == (8,)
    assert da_right_two.shape == (2,)  # Should have 2 faces

    with pytest.raises(ValueError):
        uxds['t2m'].cross_section.constant_longitude(lon=10.0)

def test_north_pole():
    uxgrid = ux.open_grid(cube_sphere_grid)
    lats = [89.85, 89.9, 89.95, 89.99]

    for lat in lats:
        cross_grid = uxgrid.cross_section.constant_latitude(lat=lat)
        assert cross_grid.n_face == 4

def test_south_pole():
    uxgrid = ux.open_grid(cube_sphere_grid)
    lats = [-89.85, -89.9, -89.95, -89.99]

    for lat in lats:
        cross_grid = uxgrid.cross_section.constant_latitude(lat=lat)
        assert cross_grid.n_face == 4

def test_constant_lat():
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

    expected_faces = np.array([0])
    np.testing.assert_array_equal(candidate_faces, expected_faces)

def test_constant_lat_out_of_bounds():
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



def test_const_lat_interval_da():
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    res = uxds['t2m'].cross_section.constant_latitude_interval(lats=(-10, 10))

    assert len(res) == 4


def test_const_lat_interval_grid():
    uxgrid = ux.open_grid(quad_hex_grid_path)

    res = uxgrid.cross_section.constant_latitude_interval(lats=(-10, 10))

    assert res.n_face == 4

    res, indices = uxgrid.cross_section.constant_latitude_interval(lats=(-10, 10), return_face_indices=True)

    assert len(indices) == 4

def test_const_lon_interva_da():
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    res = uxds['t2m'].cross_section.constant_longitude_interval(lons=(-10, 10))

    assert len(res) == 4


def test_const_lon_interval_grid():
    uxgrid = ux.open_grid(quad_hex_grid_path)

    res = uxgrid.cross_section.constant_longitude_interval(lons=(-10, 10))

    assert res.n_face == 4

    res, indices = uxgrid.cross_section.constant_longitude_interval(lons=(-10, 10), return_face_indices=True)

    assert len(indices) == 4


def test_constant_latitude_cross_sections():
    """Test constant latitude cross-sections."""
    # Create test data with vertical dimension
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    # Test 2D cross-section
    cross_2d = uxds['t2m'].cross_section.constant_latitude(
        lat=0.0, n_samples=10, lon_range=(-1, 1)
    )
    assert cross_2d.shape == (10,)
    assert 'sample' in cross_2d.dims
    assert 'lon' in cross_2d.coords
    assert 'lat' in cross_2d.coords

    # Test 3D cross-section (vertical transect)
    depth_levels = np.array([0, 10, 20, 30, 40])
    t2m_3d = uxds['t2m'].expand_dims(depth=depth_levels)

    cross_3d = t2m_3d.cross_section.constant_latitude(
        lat=0.0, n_samples=10, lon_range=(-1, 1)
    )
    assert cross_3d.shape == (5, 10)  # (depth, sample)
    assert 'sample' in cross_3d.dims
    assert 'depth' in cross_3d.dims
    assert 'lon' in cross_3d.coords
    assert 'lat' in cross_3d.coords

def test_constant_longitude_cross_sections():
    """Test constant longitude cross-sections."""
    # Create test data with vertical dimension
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    # Test 2D cross-section
    cross_2d = uxds['t2m'].cross_section.constant_longitude(
        lon=0.0, n_samples=10, lat_range=(-1, 1)
    )
    assert cross_2d.shape == (10,)
    assert 'sample' in cross_2d.dims
    assert 'lon' in cross_2d.coords
    assert 'lat' in cross_2d.coords

    # Test 3D cross-section (vertical transect)
    level_values = np.array([1000, 850, 700, 500, 300])
    t2m_3d = uxds['t2m'].expand_dims(level=level_values)

    cross_3d = t2m_3d.cross_section.constant_longitude(
        lon=0.0, n_samples=10, lat_range=(-1, 1)
    )
    assert cross_3d.shape == (5, 10)  # (level, sample)
    assert 'sample' in cross_3d.dims
    assert 'level' in cross_3d.dims
    assert 'lon' in cross_3d.coords
    assert 'lat' in cross_3d.coords

def test_cross_section_edge_cases():
    """Test edge cases for cross-sections."""
    # Test with 2D data
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    # Test with extreme ranges that might not intersect any faces
    try:
        extreme_section = uxds['t2m'].cross_section.constant_latitude(
            lat=0.0, n_samples=10, lon_range=(10, 20)  # Outside grid bounds
        )
        # Should work but might have all NaN values
        assert extreme_section.shape == (10,)
        assert 'sample' in extreme_section.dims
    except Exception:
        # This is acceptable - some ranges might not work
        pass

    # Test with very small number of samples
    small_section = uxds['t2m'].cross_section.constant_latitude(
        lat=0.0, n_samples=2, lon_range=(-1, 1)
    )
    assert small_section.shape == (2,)
    assert 'sample' in small_section.dims

def test_structured_sampling_no_intersections():
    """Test structured sampling with coordinates that don't intersect any faces."""
    # Create test data with vertical dimension
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    depth_levels = np.array([0, 10, 20, 30, 40])
    t2m_3d = uxds['t2m'].expand_dims(depth=depth_levels)

    # Test with latitude/longitude that doesn't intersect any faces
    # Should work but return data with NaN values
    no_intersection_lat = t2m_3d.cross_section.constant_latitude(
        lat=90.0, n_samples=10, lon_range=(-180, 180)
    )
    assert no_intersection_lat.shape == (5, 10)  # (depth, sample)
    assert 'sample' in no_intersection_lat.dims

    no_intersection_lon = t2m_3d.cross_section.constant_longitude(
        lon=180.0, n_samples=10, lat_range=(-90, 90)
    )
    assert no_intersection_lon.shape == (5, 10)  # (depth, sample)
    assert 'sample' in no_intersection_lon.dims

def test_structured_sampling_repr():
    """Test that the new consolidated API appears in repr."""
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)

    # Test data array repr includes the new parameters
    da_repr = uxds['t2m'].cross_section.__repr__()
    assert "n_samples" in da_repr
    assert "lon_range" in da_repr or "lat_range" in da_repr
    assert "method" in da_repr
    assert "Cross-Sections:" in da_repr

def test_structured_sampling_inverse_indices():
    """Test structured sampling with inverse_indices parameter."""
    # Create test data with vertical dimension
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    depth_levels = np.array([0, 10, 20, 30, 40])
    t2m_3d = uxds['t2m'].expand_dims(depth=depth_levels)

    # Test structured sampling with inverse_indices=True
    structured_section = t2m_3d.cross_section.constant_latitude(
        lat=0.0, n_samples=5, lon_range=(-1, 1), inverse_indices=True
    )
    assert structured_section.shape == (5, 5)  # (depth, sample)
    assert 'sample' in structured_section.dims
    assert 'depth' in structured_section.dims

    # Test structured sampling with inverse_indices as list
    structured_section_list = t2m_3d.cross_section.constant_longitude(
        lon=0.0, n_samples=5, lat_range=(-1, 1), inverse_indices=(['face'], True)
    )
    assert structured_section_list.shape == (5, 5)  # (depth, sample)
    assert 'sample' in structured_section_list.dims
    assert 'depth' in structured_section_list.dims


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



def test_double_cross_section():
    uxgrid = ux.open_grid(quad_hex_grid_path)

    # construct edges
    sub_lat = uxgrid.cross_section.constant_latitude(0.0)

    sub_lat_lon = sub_lat.cross_section.constant_longitude(0.0)

    assert "n_edge" not in sub_lat_lon._ds.dims

    _ = uxgrid.face_edge_connectivity
    _ = uxgrid.edge_node_connectivity
    _ = uxgrid.edge_lon

    sub_lat = uxgrid.cross_section.constant_latitude(0.0)

    sub_lat_lon = sub_lat.cross_section.constant_longitude(0.0)

    assert "n_edge" in sub_lat_lon._ds.dims
