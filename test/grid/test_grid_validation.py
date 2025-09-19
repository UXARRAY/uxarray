import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.validation import _find_duplicate_nodes, _check_normalization


# Removed test_grid_validation_basic - redundant with test_grid_core.py::test_grid_validate
# Removed test_grid_validation_mpas - covered by IO tests

def test_find_duplicate_nodes_no_duplicates():
    """Test duplicate node detection with no duplicates."""
    # Create a simple grid with no duplicate nodes
    face_node_connectivity = [[0, 1, 2]]
    node_lon = [0.0, 1.0, 0.5]
    node_lat = [0.0, 0.0, 1.0]

    grid = ux.Grid.from_topology(
        face_node_connectivity=face_node_connectivity,
        node_lon=node_lon,
        node_lat=node_lat
    )

    duplicates = _find_duplicate_nodes(grid)

    # Should find no duplicates
    assert len(duplicates) == 0


def test_find_duplicate_nodes_with_duplicates():
    """Test duplicate node detection with duplicates."""
    # Create a grid with duplicate nodes (same coordinates)
    face_node_connectivity = [[0, 1, 2]]
    node_lon = [0.0, 1.0, 0.0]  # First and third are same
    node_lat = [0.0, 0.0, 0.0]  # First and third are same

    grid = ux.Grid.from_topology(
        face_node_connectivity=face_node_connectivity,
        node_lon=node_lon,
        node_lat=node_lat
    )

    duplicates = _find_duplicate_nodes(grid)

    # Should find duplicates
    assert len(duplicates) > 0

def test_check_normalization_normalized(gridpath):
    """Test normalization check for normalized coordinates."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # Should be normalized
    assert _check_normalization(uxgrid)

def test_check_normalization_not_normalized(gridpath):
    """Test normalization check for non-normalized coordinates."""
    # Use a real grid file that has Cartesian coordinates
    grid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    # Manually set non-normalized coordinates
    grid.node_x.data = grid.node_x.data * 2.0  # Make non-normalized
    grid.node_y.data = grid.node_y.data * 2.0
    grid.node_z.data = grid.node_z.data * 2.0

    # Should not be normalized
    assert not _check_normalization(grid)

def test_grid_validation_comprehensive():
    """Test comprehensive grid validation scenarios."""
    # Test single face grid
    face_node_connectivity = [[0, 1, 2]]
    node_lon = [0.0, 1.0, 0.5]
    node_lat = [0.0, 0.0, 1.0]

    uxgrid = ux.Grid.from_topology(
        face_node_connectivity=face_node_connectivity,
        node_lon=node_lon,
        node_lat=node_lat
    )

    # Should validate
    assert uxgrid.validate()
    assert uxgrid.n_face == 1
    assert uxgrid.n_node == 3

def test_grid_validation_connectivity():
    """Test grid validation of connectivity."""
    # Create simple valid grid
    face_node_connectivity = [[0, 1, 2]]
    node_lon = [0.0, 1.0, 0.5]
    node_lat = [0.0, 0.0, 1.0]

    uxgrid = ux.Grid.from_topology(
        face_node_connectivity=face_node_connectivity,
        node_lon=node_lon,
        node_lat=node_lat
    )

    # Should validate
    assert uxgrid.validate()

def test_grid_validation_edge_cases(gridpath):
    """Test grid validation edge cases and error conditions."""
    # Test with a valid grid file
    grid = ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc"))

    # Basic validation that grid loaded correctly
    assert grid.n_node > 0
    assert grid.n_face > 0

    # Test that validation functions work
    assert _check_normalization(grid)  # Should be normalized by default

def test_grid_validation_mixed_face_types():
    """Test validation with mixed face types and fill values."""
    # Create grid with triangle and quad
    face_node_connectivity = [
        [0, 1, 2, INT_FILL_VALUE],  # Triangle
        [3, 4, 5, 6]                # Quad
    ]
    node_lon = [0.0, 1.0, 0.5, 2.0, 3.0, 2.5, 2.0]
    node_lat = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]

    uxgrid = ux.Grid.from_topology(
        face_node_connectivity=face_node_connectivity,
        node_lon=node_lon,
        node_lat=node_lat
    )

    # Should validate mixed face types
    assert uxgrid.validate()
    assert uxgrid.n_face == 2
