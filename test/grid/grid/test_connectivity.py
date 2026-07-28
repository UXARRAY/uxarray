import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE, ERROR_TOLERANCE
from uxarray.grid.connectivity import (_populate_face_edge_connectivity, _build_edge_face_connectivity,
                                      _build_edge_node_connectivity, _build_face_face_connectivity,
                                      _populate_face_face_connectivity, MAX_INSERTION_SORT_SIZE)


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

@pytest.mark.parametrize("grid_parts", [("ugrid", "outCSne30", "outCSne30.ug"),
                                        ("ugrid", "quad-hexagon", "grid.nc"),
                                        ("ugrid", "geoflow-small", "grid.nc")])
def test_connectivity_edge_node_canonical_order(gridpath, grid_parts):
    """Test that constructed edges are numbered in lexicographic node order."""
    uxgrid = ux.open_grid(gridpath(*grid_parts))
    edge_nodes = uxgrid.edge_node_connectivity.values

    # Each edge is stored as an ascending node pair
    assert np.all(edge_nodes[:, 0] < edge_nodes[:, 1])

    # Edges are numbered lexicographically by that pair, with no duplicates
    lexicographic_order = np.lexsort((edge_nodes[:, 1], edge_nodes[:, 0]))
    nt.assert_array_equal(lexicographic_order, np.arange(uxgrid.n_edge))
    assert len(np.unique(edge_nodes, axis=0)) == uxgrid.n_edge

@pytest.mark.parametrize("n_spoke", [MAX_INSERTION_SORT_SIZE - 1, MAX_INSERTION_SORT_SIZE + 1, 500])
def test_connectivity_edge_node_high_degree_node(n_spoke):
    """Test edge construction for a node shared by more faces than the bucket sort will
    insertion sort, which takes the heap sort path."""
    # A fan of triangles around node 0, with the spokes numbered in descending order so
    # that they reach the sort already reversed
    spokes = np.arange(n_spoke, 0, -1, dtype=INT_DTYPE)
    face_node_connectivity = np.stack(
        [np.zeros(n_spoke, dtype=INT_DTYPE), spokes, np.roll(spokes, -1)], axis=1
    )

    edge_nodes, face_edges = _build_edge_node_connectivity(
        face_node_connectivity, np.full(n_spoke, 3, dtype=INT_DTYPE), n_spoke + 1
    )

    # Same invariants as any other mesh: ascending pairs, lexicographic numbering
    assert np.all(edge_nodes[:, 0] < edge_nodes[:, 1])
    nt.assert_array_equal(
        np.lexsort((edge_nodes[:, 1], edge_nodes[:, 0])), np.arange(len(edge_nodes))
    )
    assert len(np.unique(edge_nodes, axis=0)) == len(edge_nodes)

    # The hub is shared by every face, so it has one edge per spoke
    assert np.count_nonzero(edge_nodes == 0) == n_spoke

    # And face_edge_connectivity still points at the right node pairs
    for face_idx in range(n_spoke):
        for cur in range(3):
            expected = sorted((face_node_connectivity[face_idx, cur],
                               face_node_connectivity[face_idx, (cur + 1) % 3]))
            assert sorted(edge_nodes[face_edges[face_idx, cur]]) == expected

def test_connectivity_face_edge_positional_alignment(gridpath):
    """Test that face_edge_connectivity[i, j] is the edge between face nodes j and j+1."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    face_nodes = uxgrid.face_node_connectivity.values
    face_edges = uxgrid.face_edge_connectivity.values
    edge_nodes = uxgrid.edge_node_connectivity.values

    for face_idx, n_edges in enumerate(uxgrid.n_nodes_per_face.values):
        for cur in range(n_edges):
            start_node = face_nodes[face_idx, cur]
            end_node = face_nodes[face_idx, (cur + 1) % n_edges]

            expected = sorted((start_node, end_node))
            actual = sorted(edge_nodes[face_edges[face_idx, cur]])
            assert actual == expected

        # Remaining slots stay padded
        assert np.all(face_edges[face_idx, n_edges:] == INT_FILL_VALUE)

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
