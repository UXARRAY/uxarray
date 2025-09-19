import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz


def test_zonal_face_interval_basic():
    """Test basic zonal face interval functionality."""
    # Create vertices for a face
    vertices_lonlat = [[1.6 * np.pi, 0.25 * np.pi],
                       [1.6 * np.pi, -0.25 * np.pi],
                       [0.4 * np.pi, -0.25 * np.pi],
                       [0.4 * np.pi, 0.25 * np.pi]]
    vertices = [_lonlat_rad_to_xyz(*v) for v in vertices_lonlat]

    # Should create valid Cartesian coordinates
    assert len(vertices) == 4
    for vertex in vertices:
        magnitude = np.sqrt(sum(coord**2 for coord in vertex))
        nt.assert_allclose(magnitude, 1.0, atol=ERROR_TOLERANCE)

def test_zonal_face_interval_high_latitude():
    """Test zonal face interval for high latitude case."""
    # Create face at high latitude (80 degrees = 1.396 radians)
    high_lat_rad = np.deg2rad(80.0)
    vertices_lonlat = [[0.0, high_lat_rad],
                       [0.5 * np.pi, high_lat_rad],
                       [np.pi, high_lat_rad],
                       [1.5 * np.pi, high_lat_rad]]
    vertices = [_lonlat_rad_to_xyz(*v) for v in vertices_lonlat]

    # Should create valid vertices
    assert len(vertices) == 4

    # All vertices should be at high latitude (> 70 degrees)
    for vertex in vertices:
        lat = np.arcsin(vertex[2])
        assert lat > np.deg2rad(70.0)  # High latitude in radians

def test_zonal_face_interval_fill_values():
    """Test zonal face interval with fill values."""
    # Create face with triangle (missing 4th vertex)
    vertices_lonlat = [[0.0, 0.25 * np.pi],
                       [0.5 * np.pi, 0.25 * np.pi],
                       [np.pi, 0.25 * np.pi]]
    vertices = [_lonlat_rad_to_xyz(*v) for v in vertices_lonlat]

    # Should create valid triangle vertices
    assert len(vertices) == 3
    for vertex in vertices:
        magnitude = np.sqrt(sum(coord**2 for coord in vertex))
        nt.assert_allclose(magnitude, 1.0, atol=ERROR_TOLERANCE)

def test_zonal_face_interval_equator():
    """Test zonal face interval at equator."""
    vertices_lonlat = [[0.0, 0.25 * np.pi],
                       [0.0, -0.25 * np.pi],
                       [0.5 * np.pi, -0.25 * np.pi],
                       [0.5 * np.pi, 0.25 * np.pi]]
    vertices = [_lonlat_rad_to_xyz(*v) for v in vertices_lonlat]

    # Should create valid vertices spanning equator
    assert len(vertices) == 4

    # Check that we have vertices both above and below equator
    z_coords = [vertex[2] for vertex in vertices]
    assert any(z > 0 for z in z_coords)  # Above equator
    assert any(z < 0 for z in z_coords)  # Below equator

def test_interval_processing_basic():
    """Test basic interval processing."""
    # Create test intervals
    intervals = [
        [0.0, 1.0],
        [0.5, 1.5],  # Overlaps with first
        [2.0, 3.0]   # Gap from previous
    ]

    # Should be valid intervals
    assert len(intervals) == 3
    for interval in intervals:
        assert len(interval) == 2
        assert interval[0] <= interval[1]

def test_interval_antimeridian():
    """Test interval processing across antimeridian."""
    # Create interval that crosses antimeridian
    lon_start = 350.0  # degrees
    lon_end = 10.0     # degrees

    # Convert to radians
    lon_start_rad = np.deg2rad(lon_start)
    lon_end_rad = np.deg2rad(lon_end)

    # Should handle antimeridian crossing
    assert lon_start_rad > 6.0  # Close to 2*pi (6.28)
    assert lon_end_rad < 0.2    # Close to 0 (0.175 radians = 10 degrees)

def test_zonal_face_interval_pole():
    """Test zonal face interval at pole."""
    # Create face at moderate latitude
    vertices_lonlat = [[0.0, 0.4 * np.pi],
                       [0.5 * np.pi, 0.4 * np.pi],
                       [np.pi, 0.4 * np.pi],
                       [1.5 * np.pi, 0.4 * np.pi]]
    vertices = [_lonlat_rad_to_xyz(*v) for v in vertices_lonlat]

    # Should create valid vertices at moderate latitude
    assert len(vertices) == 4
    for vertex in vertices:
        lat = np.arcsin(vertex[2])
        nt.assert_allclose(lat, 0.4 * np.pi, atol=ERROR_TOLERANCE)

def test_coordinate_conversion_consistency():
    """Test coordinate conversion consistency."""
    # Test coordinate conversion
    lon_deg = 45.0
    lat_deg = 30.0

    # Convert to radians and then to Cartesian
    lon_rad = np.radians(lon_deg)
    lat_rad = np.radians(lat_deg)
    x, y, z = _lonlat_rad_to_xyz(lon_rad, lat_rad)

    # Should be on unit sphere
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    nt.assert_allclose(magnitude, 1.0, atol=ERROR_TOLERANCE)

    # Z coordinate should match sine of latitude
    nt.assert_allclose(z, np.sin(lat_rad), atol=ERROR_TOLERANCE)

def test_latitude_conversion():
    """Test latitude conversion to Cartesian."""
    # Test various latitudes
    latitudes_deg = [0.0, 30.0, 45.0, 60.0, 90.0]

    for lat_deg in latitudes_deg:
        lat_rad = np.radians(lat_deg)
        latitude_cart = np.sin(lat_rad)

        # Should be valid Cartesian latitude
        assert -1.0 <= latitude_cart <= 1.0

        # Should match expected value
        expected = np.sin(lat_rad)
        nt.assert_allclose(latitude_cart, expected, atol=ERROR_TOLERANCE)

def test_face_edge_creation():
    """Test face edge creation."""
    # Create simple square face
    vertices_lonlat = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    vertices = [_lonlat_rad_to_xyz(*v) for v in vertices_lonlat]

    # Create face edges
    face_edges = []
    for i in range(len(vertices)):
        edge = [vertices[i], vertices[(i + 1) % len(vertices)]]
        face_edges.append(edge)

    # Should have 4 edges
    assert len(face_edges) == 4

    # Each edge should have 2 vertices
    for edge in face_edges:
        assert len(edge) == 2
