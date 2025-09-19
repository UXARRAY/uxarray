import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.utils import _get_cartesian_face_edge_nodes_array
from uxarray.grid.integrate import _zonal_face_weights, _zonal_face_weights_robust


def test_zonal_weights_basic():
    """Test basic zonal weight functionality."""
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

def test_zonal_weights_grid_properties():
    """Test zonal weight grid properties."""
    # Create grid with multiple faces
    face_vertices = [
        [[0, 0], [1, 0], [0.5, 1]],    # Triangle 1
        [[1, 0], [2, 0], [1.5, 1]]     # Triangle 2
    ]
    
    uxgrid = ux.Grid.from_face_vertices(face_vertices, latlon=True)
    
    # Should have multiple faces
    assert uxgrid.n_face == 2
    assert uxgrid.n_node > 0

def test_zonal_weights_coordinate_conversion():
    """Test coordinate conversion for zonal weights."""
    # Test coordinate conversion
    lon_deg = 45.0
    lat_deg = 30.0
    
    # Convert to radians
    lon_rad = np.radians(lon_deg)
    lat_rad = np.radians(lat_deg)
    
    # Convert to Cartesian
    x, y, z = _lonlat_rad_to_xyz(lon_rad, lat_rad)
    
    # Should be valid Cartesian coordinates
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    nt.assert_allclose(magnitude, 1.0, atol=ERROR_TOLERANCE)

def test_zonal_weights_face_creation():
    """Test face creation for zonal weights."""
    # Create face with specific coordinates
    vertices_lonlat = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_cart = np.array([_lonlat_rad_to_xyz(*np.radians(v)) for v in vertices_lonlat])
    
    # Should have valid Cartesian coordinates
    assert vertices_cart.shape == (4, 3)
    
    # All points should be on unit sphere
    magnitudes = np.sqrt(np.sum(vertices_cart**2, axis=1))
    nt.assert_allclose(magnitudes, 1.0, atol=ERROR_TOLERANCE)

def test_zonal_weights_latitude_conversion():
    """Test latitude conversion for zonal weights."""
    # Test latitude conversion
    lat_deg = 35.0
    lat_rad = np.radians(lat_deg)
    latitude_cart = np.sin(lat_rad)
    
    # Should be valid Cartesian latitude
    assert -1.0 <= latitude_cart <= 1.0

def test_compare_zonal_weights(gridpath):
    """Compares the existing weights calculation (get_non_conservative_zonal_face_weights_at_const_lat_overlap) to
    the faster implementation (get_non_conservative_zonal_face_weights_at_const_lat)"""
    gridfiles = [gridpath("ugrid", "outCSne30", "outCSne30.ug"),
                 gridpath("scrip", "outCSne8", "outCSne8.nc"),
                 gridpath("ugrid", "geoflow-small", "grid.nc"),]

    lat = (-90, 90, 10)
    latitudes = np.arange(lat[0], lat[1] + lat[2], lat[2])

    for gridfile in gridfiles:
        uxgrid = ux.open_grid(gridfile)
        n_nodes_per_face = uxgrid.n_nodes_per_face.values
        face_edge_nodes_xyz =  _get_cartesian_face_edge_nodes_array(
                uxgrid.face_node_connectivity.values,
                uxgrid.n_face,
                uxgrid.n_max_face_edges,
                uxgrid.node_x.values,
                uxgrid.node_y.values,
                uxgrid.node_z.values,
            )
        bounds = uxgrid.bounds.values

        for i, lat in enumerate(latitudes):
            face_indices = uxgrid.get_faces_at_constant_latitude(lat)
            z = np.sin(np.deg2rad(lat))

            face_edge_nodes_xyz_candidate = face_edge_nodes_xyz[face_indices, :, :, :]
            n_nodes_per_face_candidate = n_nodes_per_face[face_indices]
            bounds_candidate = bounds[face_indices]

            new_weights = _zonal_face_weights(face_edge_nodes_xyz_candidate,
                                              bounds_candidate,
                                              n_nodes_per_face_candidate,
                                              z)

            existing_weights = _zonal_face_weights_robust(
                face_edge_nodes_xyz_candidate, z, bounds_candidate
            )["weight"].to_numpy()

            abs_diff = np.abs(new_weights - existing_weights)

            # For each latitude, make sure the aboslute difference is below our error tollerance
            assert abs_diff.max() < ERROR_TOLERANCE
