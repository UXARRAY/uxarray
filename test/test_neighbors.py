import os
import numpy as np
import pytest
import xarray as xr
from pathlib import Path
import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

# Sample grid file paths
gridfile_CSne8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
gridfile_RLL1deg = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
gridfile_RLL10deg_CSne4 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"
gridfile_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
gridfile_fesom = current_path / "meshfiles" / "ugrid" / "fesom" / "fesom.mesh.diag.nc"
gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
gridfile_mpas = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'

corner_grid_files = [gridfile_CSne30, gridfile_mpas]
center_grid_files = [gridfile_mpas]

def test_construction_from_nodes():
    """Tests the construction of the ball tree on nodes and performs a sample query."""
    for grid_file in corner_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_ball_tree(coordinates="nodes").query([3.0, 3.0], k=3)
        assert len(d) == len(ind)
        assert len(d) > 0 and len(ind) > 0

def test_construction_from_face_centers():
    """Tests the construction of the ball tree on center nodes and performs a sample query."""
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_ball_tree(coordinates="face centers").query([3.0, 3.0], k=3)
        assert len(d) == len(ind)
        assert len(d) > 0 and len(ind) > 0

def test_construction_from_edge_centers():
    """Tests the construction of the ball tree on edge_centers and performs a sample query."""
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_ball_tree(coordinates="edge centers").query([3.0, 3.0], k=3)
        assert len(d) == len(ind)
        assert len(d) > 0 and len(ind) > 0

def test_construction_from_both_sequentially():
    """Tests the construction of the ball tree on center nodes and performs a sample query."""
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_ball_tree(coordinates="nodes").query([3.0, 3.0], k=3)
        d_centers, ind_centers = uxgrid.get_ball_tree(coordinates="face centers").query([3.0, 3.0], k=3)

def test_antimeridian_distance_nodes():
    """Verifies nearest neighbor search across Antimeridian."""
    verts = [(0.0, 90.0), (-180, 0.0), (0.0, -90)]
    uxgrid = ux.open_grid(verts, latlon=True)
    d, ind = uxgrid.get_ball_tree(coordinates="nodes").query([180.0, 0.0], k=1)
    assert np.isclose(d, 0.0)
    assert ind == 0
    d, ind = uxgrid.get_ball_tree(coordinates="nodes").query_radius([-180, 0.0], r=90.01, return_distance=True)
    expected_d = np.array([0.0, 90.0, 90.0])
    assert np.allclose(a=d, b=expected_d, atol=1e-03)

def test_antimeridian_distance_face_centers():
    """TODO: Write addition tests once construction and representation of face centers is implemented."""
    pass

def test_construction_using_cartesian_coords():
    """Test the BallTree creation and query function using cartesian coordinates."""
    for grid_file in corner_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_ball_tree(coordinates="nodes", coordinate_system="cartesian", distance_metric="minkowski").query([1.0, 0.0, 0.0], k=2)
        assert len(d) == len(ind)
        assert len(d) > 0 and len(ind) > 0

def test_query_radius():
    """Test the BallTree creation and query_radius function using the grids face centers."""
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        ind = uxgrid.get_ball_tree(coordinates="face centers", coordinate_system="spherical", distance_metric="haversine", reconstruct=True).query_radius([3.0, 3.0], r=15, return_distance=False)
        assert len(ind) > 0
        d, ind = uxgrid.get_ball_tree(coordinates="face centers", coordinate_system="cartesian", distance_metric="minkowski", reconstruct=True).query_radius([0.0, 0.0, 1.0], r=15, return_distance=True)
        assert len(d) > 0 and len(ind) > 0

def test_query():
    """Test the creation and querying function of the BallTree structure."""
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_ball_tree(coordinates="face centers", coordinate_system="spherical").query([0.0, 0.0], return_distance=True, k=3)
        assert len(d) > 0 and len(ind) > 0
        ind = uxgrid.get_ball_tree(coordinates="face centers", coordinate_system="spherical").query([0.0, 0.0], return_distance=False, k=3)
        assert len(ind) > 0

def test_multi_point_query():
    """Tests a query on multiple points."""
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        c = [[-100, 40], [-101, 38], [-102, 38]]
        multi_ind = uxgrid.get_ball_tree(coordinates="nodes").query_radius(c, 45)
        for i, cur_c in enumerate(c):
            single_ind = uxgrid.get_ball_tree(coordinates="nodes").query_radius(cur_c, 45)
            assert np.array_equal(single_ind, multi_ind[i])

# KDTree tests
def test_kdtree_construction_from_nodes():
    """Test the KDTree creation and query function using the grids nodes."""
    for grid_file in corner_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_kd_tree(coordinates="nodes").query([0.0, 0.0, 1.0], k=5)
        assert len(d) == len(ind)
        assert len(d) > 0 and len(ind) > 0

def test_kdtree_construction_using_spherical_coords():
    """Test the KDTree creation and query function using spherical coordinates."""
    for grid_file in corner_grid_files:
        uxgrid = ux.open_grid(grid_file)
        d, ind = uxgrid.get_kd_tree(coordinates="nodes", coordinate_system="spherical").query([3.0, 3.0], k=5)
        assert len(d) == len(ind)
        assert len(d) > 0 and len(ind) > 0

def test_kdtree_construction_from_face_centers():
    """Test the KDTree creation and query function using the grids face centers."""
    uxgrid = ux.open_grid(center_grid_files[0])
    d, ind = uxgrid.get_kd_tree(coordinates="face centers").query([1.0, 0.0, 0.0], k=5, return_distance=True)
    assert len(d) == len(ind)
    assert len(d) > 0 and len(ind) > 0

def test_kdtree_construction_from_edge_centers():
    """Tests the construction of the KDTree with cartesian coordinates on edge_centers and performs a sample query."""
    uxgrid = ux.open_grid(center_grid_files[0])
    d, ind = uxgrid.get_kd_tree(coordinates="edge centers").query([1.0, 0.0, 1.0], k=2, return_distance=True)
    assert len(d) == len(ind)
    assert len(d) > 0 and len(ind) > 0

def test_kdtree_query_radius():
    """Test the KDTree creation and query_radius function using the grids face centers."""
    uxgrid = ux.open_grid(center_grid_files[0])
    d, ind = uxgrid.get_kd_tree(coordinates="face centers", coordinate_system="spherical", reconstruct=True).query_radius([3.0, 3.0], r=5, return_distance=True)
    assert len(d) > 0 and len(ind) > 0
    ind = uxgrid.get_kd_tree(coordinates="face centers", coordinate_system="spherical", reconstruct=True).query_radius([3.0, 3.0], r=5, return_distance=False)
    assert len(ind) > 0

def test_kdtree_query():
    """Test the creation and querying function of the KDTree structure."""
    uxgrid = ux.open_grid(center_grid_files[0])
    d, ind = uxgrid.get_kd_tree(coordinates="face centers", coordinate_system="spherical").query([0.0, 0.0], return_distance=True)
    assert d, ind
    ind = uxgrid.get_kd_tree(coordinates="face centers", coordinate_system="cartesian", reconstruct=True).query([0.0, 0.0, 1.0], return_distance=False)
    assert ind

def test_kdtree_multi_point_query():
    """Tests a query on multiple points."""
    for grid_file in center_grid_files:
        uxgrid = ux.open_grid(grid_file)
        c = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
        multi_ind = uxgrid.get_kd_tree(coordinates="nodes").query_radius(c, 45)
        for i, cur_c in enumerate(c):
            single_ind = uxgrid.get_kd_tree(coordinates="nodes").query_radius(cur_c, 45)
            assert np.array_equal(single_ind, multi_ind[i])
