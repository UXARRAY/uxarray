import pytest
from pathlib import Path
import uxarray as ux
import numpy as np

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE as fv


class TestQuadHexagon:
    """Test suite for the quad-hexagon mesh connectivity construction"""

    @pytest.fixture()
    def uxgrid(self):
        grid_path = (
            Path(__file__).parent / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"
        )
        return ux.open_grid(grid_path)

    # ─── Node Connectivity ────────────────────────────────────────────────────────────────────────────────────────────

    def test_node_edge_connectivity(self, uxgrid):
        node_edges = uxgrid.node_edge_connectivity.values
        expected_node_edges = np.array(
            [
                [0, 1, 2],
                [0, 3, 4],
                [3, 5, fv],
                [5, 6, fv],
                [6, 7, 8],
                [1, 7, 9],
                [10, 11, fv],
                [10, 12, fv],
                [13, 14, fv],
                [13, 15, fv],
                [16, 17, fv],
                [16, 18, fv],
                [2, 11, 14],
                [4, 15, fv],
                [8, 18, fv],
                [9, 12, 17],
            ],
            dtype=INT_DTYPE,
        )
        assert "face_node_connectivity" in uxgrid.connectivity
        assert "edge_node_connectivity" in uxgrid.connectivity
        assert "node_edge_connectivity" in uxgrid.connectivity
        np.testing.assert_array_equal(node_edges, expected_node_edges)

    def test_node_face_connectivity(self, uxgrid):
        node_faces = uxgrid.node_face_connectivity.values
        expected_node_faces = np.array(
            [
                [0, 1, 2],
                [0, 2, fv],
                [0, fv, fv],
                [0, fv, fv],
                [0, 3, fv],
                [0, 1, 3],
                [1, fv, fv],
                [1, fv, fv],
                [2, fv, fv],
                [2, fv, fv],
                [3, fv, fv],
                [3, fv, fv],
                [1, 2, fv],
                [2, fv, fv],
                [3, fv, fv],
                [1, 3, fv],
            ],
            dtype=INT_DTYPE,
        )
        assert "face_node_connectivity" in uxgrid.connectivity
        assert "node_face_connectivity" in uxgrid.connectivity
        np.testing.assert_array_equal(node_faces, expected_node_faces)

    def test_node_node_connectivity(self, uxgrid):
        with pytest.raises(NotImplementedError):
            _ = uxgrid.node_node_connectivity.values

    # ─── Edge Connectivity ───────────────────────────────────────────────────────────────────────────────────────────

    def test_edge_node_connectivity(self, uxgrid):
        edge_nodes = uxgrid.edge_node_connectivity.values
        expected_edge_nodes = np.array(
            [
                [0, 1],
                [0, 5],
                [0, 12],
                [1, 2],
                [1, 13],
                [2, 3],
                [3, 4],
                [4, 5],
                [4, 14],
                [5, 15],
                [6, 7],
                [6, 12],
                [7, 15],
                [8, 9],
                [8, 12],
                [9, 13],
                [10, 11],
                [10, 15],
                [11, 14],
            ],
            dtype=INT_DTYPE,
        )
        assert "face_node_connectivity" in uxgrid.connectivity
        assert "edge_node_connectivity" in uxgrid.connectivity
        np.testing.assert_array_equal(edge_nodes, expected_edge_nodes)

    def test_edge_edge_connectivity(self, uxgrid):
        with pytest.raises(NotImplementedError):
            _ = uxgrid.edge_edge_connectivity.values

    def test_edge_face_connectivity(self, uxgrid):
        edge_faces = uxgrid.edge_face_connectivity.values
        expected_edge_faces = np.array(
            [
                [0, 2],
                [0, 1],
                [1, 2],
                [0, fv],
                [2, fv],
                [0, fv],
                [0, fv],
                [0, 3],
                [3, fv],
                [1, 3],
                [1, fv],
                [1, fv],
                [1, fv],
                [2, fv],
                [2, fv],
                [2, fv],
                [3, fv],
                [3, fv],
                [3, fv],
            ],
            dtype=INT_DTYPE,
        )
        assert "face_node_connectivity" in uxgrid.connectivity
        assert "edge_node_connectivity" in uxgrid.connectivity
        assert "edge_face_connectivity" in uxgrid.connectivity
        assert "face_edge_connectivity" in uxgrid.connectivity
        np.testing.assert_array_equal(edge_faces, expected_edge_faces)

    # ─── Face Connectivity ───────────────────────────────────────────────────────────────────────────────────────────

    def test_face_node_connectivity(self, uxgrid):
        expected_face_nodes = np.array(
            [
                [0, 1, 2, 3, 4, 5],
                [15, 7, 6, 12, 0, 5],
                [0, 12, 8, 9, 13, 1],
                [4, 14, 11, 10, 15, 5],
            ],
            dtype=INT_DTYPE,
        )
        face_nodes = uxgrid.face_node_connectivity.values
        assert "face_node_connectivity" in uxgrid.connectivity
        assert len(uxgrid.connectivity) == 1
        np.testing.assert_array_equal(face_nodes, expected_face_nodes)

    def test_face_edge_connectivity(self, uxgrid):
        expected_face_edges = np.array(
            [
                [0, 3, 5, 6, 7, 1],
                [12, 10, 11, 2, 1, 9],
                [2, 14, 13, 15, 4, 0],
                [8, 18, 16, 17, 9, 7],
            ],
            dtype=INT_DTYPE,
        )
        face_edges = uxgrid.face_edge_connectivity.values
        assert "face_node_connectivity" in uxgrid.connectivity
        assert "face_edge_connectivity" in uxgrid.connectivity
        assert "edge_node_connectivity" in uxgrid.connectivity
        np.testing.assert_array_equal(face_edges, expected_face_edges)

    def test_face_face_connectivity(self, uxgrid):
        face_faces = uxgrid.face_face_connectivity.values
        expected_face_faces = np.array(
            [
                [2, 1, 3, fv, fv, fv],
                [0, 2, 3, fv, fv, fv],
                [0, 1, fv, fv, fv, fv],
                [0, 1, fv, fv, fv, fv],
            ],
            dtype=INT_DTYPE,
        )
        assert "face_node_connectivity" in uxgrid.connectivity
        assert "edge_node_connectivity" in uxgrid.connectivity
        assert "edge_face_connectivity" in uxgrid.connectivity
        assert "face_edge_connectivity" in uxgrid.connectivity
        assert "face_face_connectivity" in uxgrid.connectivity
        np.testing.assert_array_equal(face_faces, expected_face_faces)
