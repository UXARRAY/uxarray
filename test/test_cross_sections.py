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

    da_top_two = uxds['t2m'].cross_section.constant_latitude(lat=0.1)
    np.testing.assert_array_equal(da_top_two.data, uxds['t2m'].isel(n_face=[1, 2]).data)

    da_bottom_two = uxds['t2m'].cross_section.constant_latitude(lat=-0.1)
    np.testing.assert_array_equal(da_bottom_two.data, uxds['t2m'].isel(n_face=[0, 3]).data)

    da_all_four = uxds['t2m'].cross_section.constant_latitude(lat=0.0)
    np.testing.assert_array_equal(da_all_four.data, uxds['t2m'].data)

    with pytest.raises(ValueError):
        uxds['t2m'].cross_section.constant_latitude(lat=10.0)

def test_constant_lon_cross_section_uxds():
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    da_left_two = uxds['t2m'].cross_section.constant_longitude(lon=-0.1)
    np.testing.assert_array_equal(da_left_two.data, uxds['t2m'].isel(n_face=[0, 2]).data)

    da_right_two = uxds['t2m'].cross_section.constant_longitude(lon=0.2)
    np.testing.assert_array_equal(da_right_two.data, uxds['t2m'].isel(n_face=[1, 3]).data)

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


def test_vertical_constant_latitude():
    """Test vertical cross-section at constant latitude."""
    # Create test data with vertical dimension
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    # Add a vertical dimension to the data
    depth_levels = np.array([0, 10, 20, 30, 40])
    t2m_3d = uxds['t2m'].expand_dims(depth=depth_levels)

    # Test successful vertical cross-section with explicit vertical coordinate
    vertical_section = t2m_3d.cross_section.vertical_constant_latitude(
        lat=0.0, vertical_coord='depth'
    )

    # Should have same number of faces as horizontal cross-section
    horizontal_section = uxds['t2m'].cross_section.constant_latitude(lat=0.0)
    assert vertical_section.sizes['n_face'] == horizontal_section.sizes['n_face']
    assert 'depth' in vertical_section.dims

def test_vertical_constant_longitude():
    """Test vertical cross-section at constant longitude."""
    # Create test data with vertical dimension
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    # Add a vertical dimension to the data
    level_values = np.array([1000, 850, 700, 500, 300])
    t2m_3d = uxds['t2m'].expand_dims(level=level_values)

    # Test successful vertical cross-section with explicit vertical coordinate
    vertical_section = t2m_3d.cross_section.vertical_constant_longitude(
        lon=0.0, vertical_coord='level'
    )

    # Should have same number of faces as horizontal cross-section
    horizontal_section = uxds['t2m'].cross_section.constant_longitude(lon=0.0)
    assert vertical_section.sizes['n_face'] == horizontal_section.sizes['n_face']
    assert 'level' in vertical_section.dims

def test_vertical_cross_section_errors():
    """Test error conditions for vertical cross-sections."""
    # Test with 2D data (no vertical dimension)
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    # Should raise ValueError for missing vertical coordinate
    with pytest.raises(ValueError, match="A vertical coordinate must be explicitly specified"):
        uxds['t2m'].cross_section.vertical_constant_latitude(lat=0.0)

    with pytest.raises(ValueError, match="A vertical coordinate must be explicitly specified"):
        uxds['t2m'].cross_section.vertical_constant_longitude(lon=0.0)

    # Test with invalid vertical coordinate name
    depth_levels = np.array([0, 10, 20, 30, 40])
    t2m_3d = uxds['t2m'].expand_dims(depth=depth_levels)

    with pytest.raises(ValueError, match="Vertical coordinate 'invalid' not found"):
        t2m_3d.cross_section.vertical_constant_latitude(lat=0.0, vertical_coord='invalid')

    with pytest.raises(ValueError, match="Vertical coordinate 'invalid' not found"):
        t2m_3d.cross_section.vertical_constant_longitude(lon=0.0, vertical_coord='invalid')

def test_vertical_cross_section_no_intersections():
    """Test vertical cross-sections with no intersections."""
    # Create test data with vertical dimension
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    depth_levels = np.array([0, 10, 20, 30, 40])
    t2m_3d = uxds['t2m'].expand_dims(depth=depth_levels)

    # Test with latitude/longitude that doesn't intersect any faces
    with pytest.raises(ValueError, match="No faces found intersecting latitude"):
        t2m_3d.cross_section.vertical_constant_latitude(lat=90.0, vertical_coord='depth')

    with pytest.raises(ValueError, match="No faces found intersecting longitude"):
        t2m_3d.cross_section.vertical_constant_longitude(lon=180.0, vertical_coord='depth')

def test_check_vertical_coord_exists():
    """Test the _check_vertical_coord_exists helper method."""
    # Create test data with known vertical dimension
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    depth_levels = np.array([0, 10, 20, 30, 40])
    t2m_3d = uxds['t2m'].expand_dims(depth=depth_levels)

    # Test successful validation
    vertical_dim = t2m_3d.cross_section._check_vertical_coord_exists('depth')
    assert vertical_dim == 'depth'

    # Test with invalid coordinate
    with pytest.raises(ValueError, match="Vertical coordinate 'invalid' not found"):
        t2m_3d.cross_section._check_vertical_coord_exists('invalid')

def test_get_vertical_coord_name():
    """Test the _get_vertical_coord_name helper method."""
    # Create test data with vertical dimension
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    depth_levels = np.array([0, 10, 20, 30, 40])
    t2m_3d = uxds['t2m'].expand_dims(depth=depth_levels)

    # Test with None input (should now raise error)
    with pytest.raises(ValueError, match="A vertical coordinate must be explicitly specified"):
        t2m_3d.cross_section._get_vertical_coord_name(None)

    # Test explicit valid coordinate
    vertical_dim_explicit = t2m_3d.cross_section._get_vertical_coord_name('depth')
    assert vertical_dim_explicit == 'depth'

    # Test invalid coordinate
    with pytest.raises(ValueError, match="Vertical coordinate 'invalid' not found"):
        t2m_3d.cross_section._get_vertical_coord_name('invalid')

def test_vertical_cross_section_repr():
    """Test that vertical methods appear in repr."""
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)

    # Test data array repr includes vertical methods
    da_repr = uxds['t2m'].cross_section.__repr__()
    assert "vertical_constant_latitude" in da_repr
    assert "vertical_constant_longitude" in da_repr
    assert "Vertical Cross-Sections:" in da_repr

def test_vertical_cross_section_inverse_indices():
    """Test vertical cross-sections with inverse_indices parameter."""
    # Create test data with vertical dimension
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)
    uxds.uxgrid.normalize_cartesian_coordinates()

    depth_levels = np.array([0, 10, 20, 30, 40])
    t2m_3d = uxds['t2m'].expand_dims(depth=depth_levels)

    # Test with inverse_indices=True
    vertical_section = t2m_3d.cross_section.vertical_constant_latitude(
        lat=0.0, vertical_coord='depth', inverse_indices=True
    )
    assert vertical_section.sizes['n_face'] == 4
    assert 'depth' in vertical_section.dims

    # Test with inverse_indices as list
    vertical_section_list = t2m_3d.cross_section.vertical_constant_longitude(
        lon=0.0, vertical_coord='depth', inverse_indices=(['face'], True)
    )
    assert vertical_section_list.sizes['n_face'] == 2
    assert 'depth' in vertical_section_list.dims


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
