import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE, ERROR_TOLERANCE
from uxarray.grid.connectivity import (_populate_face_edge_connectivity, _build_edge_face_connectivity,
                                      _build_edge_node_connectivity, _build_face_face_connectivity,
                                      _populate_face_face_connectivity)
from uxarray.grid.utils import (_adaptive_sort_bucket, _insertion_sort_bucket,
                                MIN_ADAPTIVE_SORT_SIZE)


def test_connectivity_build_n_nodes_per_face(gridpath):
    """Test building n_nodes_per_face."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # Should have n_nodes_per_face
    assert hasattr(uxgrid, 'n_nodes_per_face')
    assert len(uxgrid.n_nodes_per_face) == uxgrid.n_face

    # All values should be positive
    assert np.all(uxgrid.n_nodes_per_face > 0)

def test_connectivity_n_nodes_per_face_ragged(gridpath):
    """n_nodes_per_face counts non-fill-value nodes on a grid with mixed face sizes."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    face_nodes = uxgrid.face_node_connectivity.values
    expected = (face_nodes != INT_FILL_VALUE).sum(axis=1).astype(INT_DTYPE)

    nt.assert_array_equal(uxgrid.n_nodes_per_face.values, expected)
    assert uxgrid.n_nodes_per_face.dtype == INT_DTYPE
    # a ragged grid is the point of the test; a uniform one would pass trivially
    assert len(np.unique(expected)) > 1

def test_connectivity_n_nodes_per_face_chunked(gridpath):
    """n_nodes_per_face is counted blockwise and stays chunked over ``n_face``."""
    path = gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")
    expected = ux.open_grid(path).n_nodes_per_face.values

    uxgrid = ux.open_grid(path, chunks={"n_face": 20})
    n_nodes_per_face = uxgrid.n_nodes_per_face

    # never materialized, and partitioned the same way as its input
    assert hasattr(n_nodes_per_face.data, "dask")
    assert n_nodes_per_face.chunks == uxgrid.face_node_connectivity.chunks[:1]
    nt.assert_array_equal(n_nodes_per_face.values, expected)

def test_connectivity_n_nodes_per_face_chunked_core_dim(gridpath):
    """Chunking ``n_max_face_nodes`` is refused rather than silently rechunked."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"),
                          chunks={"n_face": 20, "n_max_face_nodes": 2})

    with pytest.raises(ValueError, match="n_max_face_nodes"):
        uxgrid.n_nodes_per_face.compute()

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

@pytest.mark.parametrize("sort", [_insertion_sort_bucket, _adaptive_sort_bucket],
                         ids=["insertion", "adaptive"])
def test_connectivity_bucket_sort(sort):
    """Test that each bucket sort orders its own slice and nothing else.

    The bucket sizes straddle ``MIN_ADAPTIVE_SORT_SIZE``: the small ones cannot accumulate
    enough shifts to exhaust the budget, so the metered sort stays on its insertion path,
    while the 500 element bucket is shuffled far past the budget and falls back to the heap
    sort. Keys repeat, since an interior edge reaches its bucket once per adjacent face.
    """
    rng = np.random.default_rng(0)

    sizes = [5, MIN_ADAPTIVE_SORT_SIZE, MIN_ADAPTIVE_SORT_SIZE + 1, 500]
    bounds = np.cumsum([0] + sizes)
    n_half_edge = int(bounds[-1])
    buckets = list(zip(bounds[:-1], bounds[1:]))

    keys = rng.integers(0, 40, n_half_edge).astype(INT_DTYPE)
    order = rng.permutation(n_half_edge).astype(INT_DTYPE)

    # the key each half edge must still be paired with once the permutation has moved it
    key_for = np.empty(n_half_edge, dtype=INT_DTYPE)
    key_for[order] = keys

    expected_keys = np.concatenate([np.sort(keys[start:end]) for start, end in buckets])

    got_keys, got_order = keys.copy(), order.copy()
    for start, end in buckets:
        shuffle = rng.permutation(end - start)
        got_keys[start:end] = got_keys[start:end][shuffle]
        got_order[start:end] = got_order[start:end][shuffle]

        sort(got_keys, got_order, start, end - start)

    nt.assert_array_equal(got_keys, expected_keys)

    # sorted keys alone would pass even if the permutation had been scrambled independently
    nt.assert_array_equal(key_for[got_order], got_keys)
    nt.assert_array_equal(np.sort(got_order), np.arange(n_half_edge))


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
