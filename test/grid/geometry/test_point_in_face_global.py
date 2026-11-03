"""Global containment regressions independent of the predicate implementation."""

import numpy as np
import numpy.testing as nt
import pytest

from uxarray.grid.point_in_face import (
    _LOC_INSIDE,
    _LOC_OUTSIDE,
    _point_in_face,
    _point_in_polygon_sphere,
)


@pytest.mark.parametrize("reverse", [False, True])
def test_small_triangle_matches_convex_halfspaces_globally(reverse):
    """Far-away rays and the antipodal triangle must not produce false hits."""
    polygon = np.array([[1.0, -0.1, -0.1], [1.0, 0.1, -0.1], [1.0, 0.0, 0.1]])
    polygon /= np.linalg.norm(polygon, axis=1)[:, None]
    normals = np.cross(polygon, np.roll(polygon, -1, axis=0))
    normals *= np.sign(normals @ polygon.mean(axis=0))[:, None]
    rng = np.random.default_rng(42)
    points = rng.normal(size=(10000, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    expected = np.all(points @ normals.T >= 0, axis=1)
    if reverse:
        polygon = polygon[::-1].copy()
    actual = np.array(
        [_point_in_polygon_sphere(q, polygon) == _LOC_INSIDE for q in points]
    )
    nt.assert_array_equal(actual, expected)
    assert _point_in_polygon_sphere(np.array([-1.0, 0.0, 0.0]), polygon) == _LOC_OUTSIDE


def test_rotated_quads_match_convex_halfspaces():
    """Rotation must not change containment, including around poles and seams."""
    rng = np.random.default_rng(123)
    for _ in range(10):
        polygon = np.array(
            [[1.0, -0.2, -0.2], [1.0, 0.2, -0.2],
             [1.0, 0.2, 0.2], [1.0, -0.2, 0.2]]
        )
        polygon /= np.linalg.norm(polygon, axis=1)[:, None]
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        polygon = polygon @ rotation
        normals = np.cross(polygon, np.roll(polygon, -1, axis=0))
        normals *= np.sign(normals @ polygon.mean(axis=0))[:, None]
        points = rng.normal(size=(1000, 3))
        points /= np.linalg.norm(points, axis=1)[:, None]
        expected = np.all(points @ normals.T >= 0, axis=1)
        actual = np.array(
            [_point_in_polygon_sphere(q, polygon) == _LOC_INSIDE for q in points]
        )
        nt.assert_array_equal(actual, expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_concave_ring_matches_gnomonic_planar_crossings(reverse):
    """Great-circle edges project to straight edges in a gnomonic chart."""
    vertices = np.array(
        [[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [0.0, 0.0], [-0.3, 0.3]]
    )
    polygon = np.column_stack((np.ones(len(vertices)), vertices))
    polygon /= np.linalg.norm(polygon, axis=1)[:, None]
    rng = np.random.default_rng(19)
    projected = rng.uniform(-0.5, 0.5, size=(2000, 2))
    expected = np.zeros(len(projected), dtype=bool)
    x, y = projected.T
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        if a[1] != b[1]:
            crosses = (a[1] > y) != (b[1] > y)
            edge_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            expected ^= crosses & (x < edge_x)
    points = np.column_stack((np.ones(len(projected)), projected))
    points /= np.linalg.norm(points, axis=1)[:, None]
    if reverse:
        polygon = polygon[::-1].copy()
    actual = np.array(
        [_point_in_polygon_sphere(q, polygon) == _LOC_INSIDE for q in points]
    )
    nt.assert_array_equal(actual, expected)


def test_node_index_entry_point_accepts_tuple():
    """The batch API's tuple query retains the same containment semantics."""
    polygon = np.array([[1.0, -0.1, -0.1], [1.0, 0.1, -0.1], [1.0, 0.0, 0.1]])
    polygon /= np.linalg.norm(polygon, axis=1)[:, None]
    nodes = np.arange(3)
    assert _point_in_face((1.0, 0.0, 0.0), nodes, *polygon.T)
    assert not _point_in_face((-1.0, 0.0, 0.0), nodes, *polygon.T)
