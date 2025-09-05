import os
from pathlib import Path

import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import INT_FILL_VALUE

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *

@pytest.fixture(params=["healpix", "quadhex"])
def grid(request):
    if request.param == "healpix":
        g = ux.Grid.from_healpix(zoom=0)
    else:
        g = ux.open_grid(QUAD_HEXAGON_GRID)
    g.normalize_cartesian_coordinates()
    return g

def test_face_centers(grid):
    """
    For each face of the grid, verify that
      - querying its Cartesian center returns exactly that face
      - querying its lon/lat center returns exactly that face
    """
    centers_xyz = np.vstack([
        grid.face_x.values,
        grid.face_y.values,
        grid.face_z.values,
    ]).T

    for fid, center in enumerate(centers_xyz):
        hits = grid.get_faces_containing_point(
            points=center,
            return_counts=False
        )
        assert isinstance(hits, list)
        assert len(hits) == 1
        assert hits[0] == [fid]

    centers_lonlat = np.vstack([
        grid.face_lon.values,
        grid.face_lat.values,
    ]).T

    for fid, (lon, lat) in enumerate(centers_lonlat):
        hits = grid.get_faces_containing_point(
            points=(lon, lat),
            return_counts=False
        )
        assert hits[0] == [fid]

def test_node_corners(grid):
    """
    For each corner node, verify that querying its coordinate in both
    Cartesian and spherical (lon/lat) returns exactly the faces sharing it.
    """

    node_coords = np.vstack([
        grid.node_x.values,
        grid.node_y.values,
        grid.node_z.values,
    ]).T

    conn = grid.node_face_connectivity.values
    counts = np.sum(conn != INT_FILL_VALUE, axis=1)

    # Cartesian tests
    for nid, (x, y, z) in enumerate(node_coords):
        expected = conn[nid, :counts[nid]].tolist()

        hits_xyz = grid.get_faces_containing_point(
            points=(x, y, z),
            return_counts=False
        )[0]
        assert set(hits_xyz) == set(expected)
        assert len(hits_xyz) == len(expected)

    node_lonlat = np.vstack([
        grid.node_lon.values,
        grid.node_lat.values,
    ]).T

    # Spherical tests
    for nid, (lon, lat) in enumerate(node_lonlat):
        expected = conn[nid, :counts[nid]].tolist()

        hits_ll = grid.get_faces_containing_point(
            points=(lon, lat),
            return_counts=False
        )[0]
        assert set(hits_ll) == set(expected)
        assert len(hits_ll) == len(expected)

def test_number_of_faces_found():
    """Test function for `self.get_face_containing_point`,
    to ensure the correct number of faces is found, depending on where the point is."""
    grid = ux.open_grid(MPAS_QU_MESH)
    partial_grid = ux.open_grid(QUAD_HEXAGON_GRID)

    # For a face center only one face should be found
    point_xyz = np.array([grid.face_x[100].values, grid.face_y[100].values, grid.face_z[100].values], dtype=np.float64)

    assert len(grid.get_faces_containing_point(point_xyz, return_counts=False)[0]) == 1

    # For an edge two faces should be found
    point_xyz = np.array([grid.edge_x[100].values, grid.edge_y[100].values, grid.edge_z[100].values], dtype=np.float64)

    assert len(grid.get_faces_containing_point(point_xyz, return_counts=False)[0]) == 2

    # For a node three faces should be found
    point_xyz = np.array([grid.node_x[100].values, grid.node_y[100].values, grid.node_z[100].values], dtype=np.float64)

    assert len(grid.get_faces_containing_point(point_xyz, return_counts=False)[0]) == 3

    partial_grid.normalize_cartesian_coordinates()

    # Test for a node on the edge where only 2 faces should be found
    point_xyz = np.array([partial_grid.node_x[1].values, partial_grid.node_y[1].values, partial_grid.node_z[1].values], dtype=np.float64)

    assert len(partial_grid.get_faces_containing_point(point_xyz, return_counts=False)[0]) == 2

def test_point_along_arc():
    node_lon = np.array([-40, -40, 40, 40])
    node_lat = np.array([-20, 20, 20, -20])
    face_node_connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)

    uxgrid = ux.Grid.from_topology(node_lon, node_lat, face_node_connectivity)

    # point at exactly 20 degrees latitude
    out1 = uxgrid.get_faces_containing_point(np.array([0, 20], dtype=np.float64), return_counts=False)

    # point at 25.41 degrees latitude (max along the great circle arc)
    out2 = uxgrid.get_faces_containing_point(np.array([0, 25.41], dtype=np.float64), return_counts=False)

    nt.assert_array_equal(out1[0], out2[0])

def test_coordinates(grid):

    lonlat = np.vstack([grid.node_lon.values, grid.node_lat.values]).T
    xyz = np.vstack([grid.node_x.values, grid.node_y.values, grid.node_z.values]).T

    faces_from_lonlat, _ = grid.get_faces_containing_point(points=lonlat)
    faces_from_xyz, _ = grid.get_faces_containing_point(points=xyz)

    nt.assert_array_equal(faces_from_lonlat, faces_from_xyz)

    with pytest.raises(ValueError):
        dummy_points = np.ones((10, 4))
        faces_query_both, _ = grid.get_faces_containing_point(points=dummy_points)
