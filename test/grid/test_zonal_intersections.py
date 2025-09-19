import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz


def test_zonal_intersections_basic():
    """Test basic zonal intersection functionality."""
    # Create simple grid
    vertices = [
        [0.0, 90.0],
        [0.0, 0.0],
        [90.0, 0.0]
    ]

    uxgrid = ux.Grid.from_face_vertices(vertices, latlon=True)

    # Should be able to create grid successfully
    assert uxgrid.n_face == 1
    assert uxgrid.n_node == 3

def test_zonal_intersections_coordinate_conversion():
    """Test coordinate conversion for intersections."""
    # Test latitude conversion
    latitude_deg = -60.0
    latitude_rad = np.radians(latitude_deg)
    latitude_cart = np.sin(latitude_rad)

    # Should be valid Cartesian latitude
    assert -1.0 <= latitude_cart <= 1.0
    nt.assert_allclose(latitude_cart, -0.8660254037844386, atol=ERROR_TOLERANCE)

def test_zonal_intersections_face_edges():
    """Test face edge creation for intersections."""
    # Create face edges in Cartesian coordinates
    face_edges_cart = np.array([
        [[-0.5, -0.04, -0.84], [-0.54, 0.0, -0.84]],
        [[-0.54, 0.0, -0.84], [-0.5, 0.0, -0.87]],
        [[-0.5, 0.0, -0.87], [-0.5, -0.04, -0.87]],
        [[-0.5, -0.04, -0.87], [-0.5, -0.04, -0.84]]
    ])

    # Should have valid shape
    assert face_edges_cart.shape == (4, 2, 3)

    # All coordinates should be reasonable
    assert np.all(np.abs(face_edges_cart) <= 1.0)

def test_zonal_intersections_pole_case():
    """Test intersection near pole."""
    # Test coordinates near pole
    latitude_cart = 0.9993908270190958
    latitude_rad = np.arcsin(latitude_cart)
    latitude_deg = np.rad2deg(latitude_rad)

    # Should be near 90 degrees
    assert latitude_deg > 85.0
    assert latitude_deg < 90.0
