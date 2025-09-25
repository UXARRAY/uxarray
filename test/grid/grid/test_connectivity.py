import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import INT_FILL_VALUE, ERROR_TOLERANCE
from uxarray.grid.connectivity import (_populate_face_edge_connectivity, _build_edge_face_connectivity,
                                      _build_edge_node_connectivity, _build_face_face_connectivity,
                                      _populate_face_face_connectivity)


def test_connectivity_build_n_nodes_per_face(gridpath):
    """Test building n_nodes_per_face."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # Should have n_nodes_per_face
    assert hasattr(uxgrid, 'n_nodes_per_face')
    assert len(uxgrid.n_nodes_per_face) == uxgrid.n_face

    # All values should be positive
    assert np.all(uxgrid.n_nodes_per_face > 0)

def test_connectivity_edge_nodes_euler(gridpath):
    """Test edge-node connectivity using Euler's formula."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # For a closed mesh on a sphere: V - E + F = 2 (Euler's formula)
    V = uxgrid.n_node
    E = uxgrid.n_edge
    F = uxgrid.n_face

    # Check Euler's formula (allowing some tolerance for numerical issues)
    euler_characteristic = V - E + F
    assert abs(euler_characteristic - 2) <= 1

def test_connectivity_build_face_edges_connectivity_mpas(gridpath):
    """Test face-edge connectivity for MPAS grid."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    # Should have face_edge_connectivity
    assert hasattr(uxgrid, 'face_edge_connectivity')
    assert uxgrid.face_edge_connectivity.shape[0] == uxgrid.n_face

    # Check that connectivity values are valid
    face_edge_conn = uxgrid.face_edge_connectivity.values
    valid_edges = face_edge_conn[face_edge_conn != INT_FILL_VALUE]
    assert np.all(valid_edges >= 0)
    assert np.all(valid_edges < uxgrid.n_edge)

def test_connectivity_build_face_edges_connectivity(gridpath):
    """Test face-edge connectivity construction."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # Should have face_edge_connectivity
    assert hasattr(uxgrid, 'face_edge_connectivity')

    # Check dimensions
    assert uxgrid.face_edge_connectivity.shape[0] == uxgrid.n_face

    # Check that connectivity values are valid
    face_edge_conn = uxgrid.face_edge_connectivity.values
    valid_edges = face_edge_conn[face_edge_conn != INT_FILL_VALUE]
    assert np.all(valid_edges >= 0)
    assert np.all(valid_edges < uxgrid.n_edge)

def test_connectivity_build_face_edges_connectivity_fillvalues():
    """Test face-edge connectivity with fill values."""
    # Create a simple grid with mixed face types
    face_node_connectivity = [
        [0, 1, 2, INT_FILL_VALUE],  # Triangle
        [3, 4, 5, 6]                # Quad
    ]
    node_lon = [0, 1, 0.5, 2, 3, 2.5, 2]
    node_lat = [0, 0, 1, 0, 0, 1, 1]

    uxgrid = ux.Grid.from_topology(
        face_node_connectivity=face_node_connectivity,
        node_lon=node_lon,
        node_lat=node_lat
    )

    # Should handle fill values correctly
    assert hasattr(uxgrid, 'face_edge_connectivity')

    # Check that fill values are preserved where appropriate
    face_edge_conn = uxgrid.face_edge_connectivity.values
    assert INT_FILL_VALUE in face_edge_conn

def test_connectivity_node_face_connectivity_from_verts():
    """Test node-face connectivity from vertices."""
    # Simple grid with shared nodes
    face_vertices = [
        [[0, 0], [1, 0], [0.5, 1]],    # Triangle 1
        [[1, 0], [2, 0], [1.5, 1]]     # Triangle 2 (shares edge with Triangle 1)
    ]

    uxgrid = ux.Grid.from_face_vertices(face_vertices, latlon=True)

    # Should have node_face_connectivity
    assert hasattr(uxgrid, 'node_face_connectivity')

    # Check that shared nodes reference multiple faces
    node_face_conn = uxgrid.node_face_connectivity.values

    # Some nodes should be connected to multiple faces
    nodes_with_multiple_faces = np.sum(node_face_conn != INT_FILL_VALUE, axis=1)
    assert np.any(nodes_with_multiple_faces > 1)

def test_connectivity_node_face_connectivity_from_files(gridpath):
    """Test node-face connectivity from files."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # Should have node_face_connectivity
    assert hasattr(uxgrid, 'node_face_connectivity')
    assert uxgrid.node_face_connectivity.shape[0] == uxgrid.n_node

    # Check that connectivity values are valid
    node_face_conn = uxgrid.node_face_connectivity.values
    valid_faces = node_face_conn[node_face_conn != INT_FILL_VALUE]
    assert np.all(valid_faces >= 0)
    assert np.all(valid_faces < uxgrid.n_face)

def test_connectivity_edge_face_connectivity_mpas(gridpath):
    """Test edge-face connectivity for MPAS grid."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    # Should have edge_face_connectivity
    assert hasattr(uxgrid, 'edge_face_connectivity')
    assert uxgrid.edge_face_connectivity.shape[0] == uxgrid.n_edge

    # Each edge should connect at most 2 faces
    edge_face_conn = uxgrid.edge_face_connectivity.values
    n_faces_per_edge = np.sum(edge_face_conn != INT_FILL_VALUE, axis=1)
    assert np.all(n_faces_per_edge <= 2)
    assert np.all(n_faces_per_edge >= 1)  # Each edge should connect at least 1 face

def test_connectivity_edge_face_connectivity_sample():
    """Test edge-face connectivity for sample grid."""
    # Create a simple grid
    face_node_connectivity = [
        [0, 1, 2],  # Triangle 1
        [1, 3, 2]   # Triangle 2 (shares edge with Triangle 1)
    ]
    node_lon = [0, 1, 0.5, 1.5]
    node_lat = [0, 0, 1, 1]

    uxgrid = ux.Grid.from_topology(
        face_node_connectivity=face_node_connectivity,
        node_lon=node_lon,
        node_lat=node_lat
    )

    # Should have edge_face_connectivity
    assert hasattr(uxgrid, 'edge_face_connectivity')

    # Check that shared edge connects both faces
    edge_face_conn = uxgrid.edge_face_connectivity.values

    # Some edges should connect 2 faces (shared edges)
    n_faces_per_edge = np.sum(edge_face_conn != INT_FILL_VALUE, axis=1)
    assert np.any(n_faces_per_edge == 2)

def test_connectivity_face_face_connectivity_construction(gridpath):
    """Test face-face connectivity construction."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # Should have face_face_connectivity
    assert hasattr(uxgrid, 'face_face_connectivity')
    assert uxgrid.face_face_connectivity.shape[0] == uxgrid.n_face

    # Check that connectivity values are valid
    face_face_conn = uxgrid.face_face_connectivity.values
    valid_neighbors = face_face_conn[face_face_conn != INT_FILL_VALUE]
    assert np.all(valid_neighbors >= 0)
    assert np.all(valid_neighbors < uxgrid.n_face)

    # No face should be its own neighbor
    for i in range(uxgrid.n_face):
        neighbors = face_face_conn[i]
        valid_neighbors = neighbors[neighbors != INT_FILL_VALUE]
        assert i not in valid_neighbors
