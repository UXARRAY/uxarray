import numpy as np
import pytest
import xarray as xr
import uxarray as ux
from uxarray.grid.neighbors import _construct_edge_face_distances





def test_construction_from_nodes(gridpath):
    """Tests the construction of the ball tree on nodes and performs a sample query."""
    corner_grid_files = [gridpath("ugrid", "outCSne30", "outCSne30.ug"), gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    for grid_file in corner_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_ball_tree(coordinates="nodes").query([3.0, 3.0], k=3)
        assert len(d) == len(ind)
        assert len(d) > 0 and len(ind) > 0

def test_construction_from_face_centers(gridpath):
    """Tests the construction of the ball tree on center nodes and performs a sample query."""
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_ball_tree(coordinates="face centers").query([3.0, 3.0], k=3)
        assert len(d) == len(ind)
        assert len(d) > 0 and len(ind) > 0

def test_construction_from_edge_centers(gridpath):
    """Tests the construction of the ball tree on edge_centers and performs a sample query."""
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_ball_tree(coordinates="edge centers").query([3.0, 3.0], k=3)
        assert len(d) == len(ind)
        assert len(d) > 0 and len(ind) > 0

def test_construction_from_both_sequentially(gridpath):
    """Tests the construction of the ball tree on center nodes and performs a sample query."""
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_ball_tree(coordinates="nodes").query([3.0, 3.0], k=3)
        d_centers, ind_centers = uxgrid.get_ball_tree(coordinates="face centers").query([3.0, 3.0], k=3)

def test_antimeridian_distance_nodes(gridpath):
    """Verifies nearest neighbor search across Antimeridian."""
    verts = [(0.0, 90.0), (-180, 0.0), (0.0, -90)]
    uxgrid = ux.open_grid(verts, latlon=True)
    d, ind = uxgrid.get_ball_tree(coordinates="nodes").query([180.0, 0.0], k=1)
    assert np.isclose(d, 0.0)
    assert ind == 0
    d, ind = uxgrid.get_ball_tree(coordinates="nodes").query_radius([-180, 0.0], r=90.01, return_distance=True)
    expected_d = np.array([0.0, 90.0, 90.0])
    assert np.allclose(a=d, b=expected_d, atol=1e-03)

def test_antimeridian_distance_face_centers(gridpath):
    """TODO: Write addition tests once construction and representation of face centers is implemented."""
    pass

def test_construction_using_cartesian_coords(gridpath):
    """Test the BallTree creation and query function using cartesian coordinates."""
    corner_grid_files = [gridpath("ugrid", "outCSne30", "outCSne30.ug"), gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    for grid_file in corner_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_ball_tree(coordinates="nodes", coordinate_system="cartesian", distance_metric="minkowski").query([1.0, 0.0, 0.0], k=2)
        assert len(d) == len(ind)
        assert len(d) > 0 and len(ind) > 0

def test_query_radius(gridpath):
    """Test the BallTree creation and query_radius function using the grids face centers."""
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        ind = uxgrid.get_ball_tree(coordinates="face centers", coordinate_system="spherical", distance_metric="haversine", reconstruct=True).query_radius([3.0, 3.0], r=15, return_distance=False)
        assert len(ind) > 0
        d, ind = uxgrid.get_ball_tree(coordinates="face centers", coordinate_system="cartesian", distance_metric="minkowski", reconstruct=True).query_radius([0.0, 0.0, 1.0], r=15, return_distance=True)
        assert len(d) > 0 and len(ind) > 0

def test_query(gridpath):
    """Test the creation and querying function of the BallTree structure."""
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_ball_tree(coordinates="face centers", coordinate_system="spherical").query([0.0, 0.0], return_distance=True, k=3)
        assert len(d) > 0 and len(ind) > 0
        ind = uxgrid.get_ball_tree(coordinates="face centers", coordinate_system="spherical").query([0.0, 0.0], return_distance=False, k=3)
        assert len(ind) > 0

def test_multi_point_query(gridpath):
    """Tests a query on multiple points."""
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        c = [[-100, 40], [-101, 38], [-102, 38]]
        multi_ind = uxgrid.get_ball_tree(coordinates="nodes").query_radius(c, 45)
        for i, cur_c in enumerate(c):
            single_ind = uxgrid.get_ball_tree(coordinates="nodes").query_radius(cur_c, 45)
            assert np.array_equal(single_ind, multi_ind[i])

# KDTree tests
def test_kdtree_construction_from_nodes(gridpath):
    """Test the KDTree creation and query function using the grids nodes."""
    corner_grid_files = [gridpath("ugrid", "outCSne30", "outCSne30.ug"), gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    for grid_file in corner_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_kd_tree(coordinates="nodes").query([0.0, 0.0, 1.0], k=5)
        assert len(d) == len(ind)
        assert len(d) > 0 and len(ind) > 0

def test_kdtree_construction_using_spherical_coords(gridpath):
    """Test the KDTree creation and query function using spherical coordinates."""
    corner_grid_files = [gridpath("ugrid", "outCSne30", "outCSne30.ug"), gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    for grid_file in corner_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_kd_tree(coordinates="nodes", coordinate_system="spherical").query([3.0, 3.0], k=5)
        assert len(d) == len(ind)
        assert len(d) > 0 and len(ind) > 0

def test_kdtree_construction_from_face_centers(gridpath):
    """Test the KDTree creation and query function using the grids face centers."""
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    uxgrid = ux.open_grid(center_grid_files[0])
    d, ind = uxgrid.get_kd_tree(coordinates="face centers").query([1.0, 0.0, 0.0], k=5, return_distance=True)
    assert len(d) == len(ind)
    assert len(d) > 0 and len(ind) > 0

def test_kdtree_construction_from_edge_centers(gridpath):
    """Tests the construction of the KDTree with cartesian coordinates on edge_centers and performs a sample query."""
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    uxgrid = ux.open_grid(center_grid_files[0])
    d, ind = uxgrid.get_kd_tree(coordinates="edge centers").query([1.0, 0.0, 1.0], k=2, return_distance=True)
    assert len(d) == len(ind)
    assert len(d) > 0 and len(ind) > 0

def test_kdtree_query_radius(gridpath):
    """Test the KDTree creation and query_radius function using the grids face centers."""
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    uxgrid = ux.open_grid(center_grid_files[0])
    d, ind = uxgrid.get_kd_tree(coordinates="face centers", coordinate_system="spherical", reconstruct=True).query_radius([3.0, 3.0], r=5, return_distance=True)
    assert len(d) > 0 and len(ind) > 0
    ind = uxgrid.get_kd_tree(coordinates="face centers", coordinate_system="spherical", reconstruct=True).query_radius([3.0, 3.0], r=5, return_distance=False)
    assert len(ind) > 0

def test_kdtree_query(gridpath):
    """Test the creation and querying function of the KDTree structure."""
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    uxgrid = ux.open_grid(center_grid_files[0])
    d, ind = uxgrid.get_kd_tree(coordinates="face centers", coordinate_system="spherical").query([0.0, 0.0], return_distance=True)
    assert d, ind
    ind = uxgrid.get_kd_tree(coordinates="face centers", coordinate_system="cartesian", reconstruct=True).query([0.0, 0.0, 1.0], return_distance=False)
    assert ind

def test_kdtree_multi_point_query(gridpath):
    """Tests a query on multiple points."""
    center_grid_files = [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")]
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        c = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
        multi_ind = uxgrid.get_kd_tree(coordinates="nodes").query_radius(c, 45)
        for i, cur_c in enumerate(c):
            single_ind = uxgrid.get_kd_tree(coordinates="nodes").query_radius(cur_c, 45)
            assert np.array_equal(single_ind, multi_ind[i])

def test_construct_edge_face_distances(gridpath):
    """
    Test _construct_edge_face_distances by verifying known great-circle distances
    between face centers on a unit sphere.
    """
    face_lon = np.array([0, 0, 90, 90, -45])
    face_lat = np.array([0, 90, 0, 90, 0])
    edge_faces = np.array([
            [0, 1],  # from (0,0) to (0,90)
            [0, 2],  # from (0,0) to (90,0)
            [1, 3],  # from (0,90) to (90,90) — both poles, same point
            [2, 4],  # from (90,0) to (-45,0)
        ])

    # Expected great-circle distances in radians
    expected = np.array([
        np.pi / 2,  # 0 → 1
        np.pi / 2,  # 0 → 2
        0.0,  # 1 → 3
        3 * np.pi / 4  # 2 → 4
    ])

    # Run the function under test
    calculated = _construct_edge_face_distances(face_lon, face_lat, edge_faces)
    np.testing.assert_array_almost_equal(calculated, expected, decimal=5)
