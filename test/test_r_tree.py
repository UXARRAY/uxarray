import os
from pathlib import Path

import numpy as np
import uxarray as ux

from uxarray.grid.r_tree import (
    _face_aabb_xyz_kernel,
    face_aabb_xyz,
    construct_face_rtree_from_bounds,
    faces_aabb_overlap_from_bounds,
    find_intersecting_face_pairs,
    RTREE_AVAILABLE,
)


def test_face_aabb_single_face_no_wrap():
    lat0 = np.deg2rad(10.0)
    lat1 = np.deg2rad(20.0)
    lon0 = np.deg2rad(30.0)
    lon1 = np.deg2rad(40.0)
    xmin, ymin, zmin, xmax, ymax, zmax = _face_aabb_xyz_kernel(lat0, lat1, lon0, lon1)
    assert zmin < zmax
    assert np.isfinite([xmin, ymin, zmin, xmax, ymax, zmax]).all()


def test_face_aabb_batch_with_wrap():
    lat_bounds = np.array([
        [np.deg2rad(0.0), np.deg2rad(10.0)],
        [np.deg2rad(-5.0), np.deg2rad(5.0)],
    ])
    lon_bounds = np.array([
        [np.deg2rad(10.0), np.deg2rad(20.0)],
        [np.deg2rad(170.0), np.deg2rad(-170.0)],
    ])
    boxes = face_aabb_xyz(lat_bounds, lon_bounds)
    assert boxes.shape == (2, 6)
    assert np.isfinite(boxes).all()


def test_construct_rtree_from_grid_bounds_geoflow():
    if not RTREE_AVAILABLE:
        import pytest
        pytest.skip("rtree not available")

    here = Path(os.path.dirname(os.path.realpath(__file__)))
    grid_path = here / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
    grid = ux.open_grid(grid_path)
    bounds = grid.bounds
    rtree, boxes, dim = construct_face_rtree_from_bounds(bounds)
    assert boxes.shape[0] == grid.n_face
    assert dim in (2, 3)
    # Query using the rtree directly
    hits_any = list(rtree.intersection(boxes[0]))
    assert isinstance(hits_any, list)


def test_faces_aabb_overlap_and_pairs_synthetic():
    A = np.array([[29.5, 11.0], [29.5, 10.0], [30.5, 10.0], [30.5, 11.0]])
    B = np.array([[30.0, 10.5], [30.0,  9.8], [31.0,  9.8], [31.0, 10.5]])
    grid = ux.Grid.from_face_vertices([A, B], latlon=True)
    bounds = grid.bounds
    assert faces_aabb_overlap_from_bounds(bounds, 0, 1)
    pairs = find_intersecting_face_pairs(bounds)
    assert pairs.shape[1] == 2
    assert (pairs == np.array([[0, 1]])).all()


def test_rtree_face_center_point_hits_quad_hexagon():
    if not RTREE_AVAILABLE:
        import pytest
        pytest.skip("rtree not available")

    here = Path(os.path.dirname(os.path.realpath(__file__)))
    quad_hex_path = here / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"
    uxgrid = ux.open_grid(quad_hex_path)
    rtree, boxes, dim = construct_face_rtree_from_bounds(uxgrid.bounds)
    for i in range(uxgrid.n_face):
        x = uxgrid.face_x[i].item(); y = uxgrid.face_y[i].item(); z = uxgrid.face_z[i].item()
        # Build a tight 3D point-like box; for 2D trees, reduce to XY
        if dim == 3:
            query = (x, y, z, x, y, z)
        else:
            query = (x, y, x, y)
        hits = list(rtree.intersection(query))
        assert len(hits) >= 1
        # Ensure exact face is among hits; filter by AABB overlap exactness
        # Convert to Python ints for consistency
        assert int(i) in set(int(h) for h in hits)
