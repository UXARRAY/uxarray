import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _normalize_xyz, _xyz_to_lonlat_rad
from uxarray.grid.geometry import _pole_point_inside_polygon_cartesian


def test_antimeridian_crossing():
    verts = [[[-170, 40], [180, 30], [165, 25], [-170, 20]]]

    uxgrid = ux.open_grid(verts, latlon=True)

    gdf = uxgrid.to_geodataframe(periodic_elements='ignore')

    assert len(uxgrid.antimeridian_face_indices) == 1
    assert len(gdf['geometry']) == 1


def test_antimeridian_point_on():
    verts = [[[-170, 40], [180, 30], [-170, 20]]]

    uxgrid = ux.open_grid(verts, latlon=True)

    assert len(uxgrid.antimeridian_face_indices) == 1


def test_linecollection_execution(gridpath):
    uxgrid = ux.open_grid(gridpath("scrip", "outCSne8", "outCSne8.nc"))
    lines = uxgrid.to_linecollection()


def test_pole_point_inside_polygon_from_vertice_north():
    vertices = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5]]

    for i, vertex in enumerate(vertices):
        float_vertex = [float(coord) for coord in vertex]
        vertices[i] = _normalize_xyz(*float_vertex)

    face_edge_cart = np.array([[vertices[0], vertices[1]],
                               [vertices[1], vertices[2]],
                               [vertices[2], vertices[3]],
                               [vertices[3], vertices[0]]])

    result = _pole_point_inside_polygon_cartesian('North', face_edge_cart)
    assert result, "North pole should be inside the polygon"

    result = _pole_point_inside_polygon_cartesian('South', face_edge_cart)
    assert not result, "South pole should not be inside the polygon"


def test_pole_point_inside_polygon_from_vertice_south():
    vertices = [[0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [0.0, 0.0, -1.0]]

    for i, vertex in enumerate(vertices):
        float_vertex = [float(coord) for coord in vertex]
        vertices[i] = _normalize_xyz(*float_vertex)

    face_edge_cart = np.array([[vertices[0], vertices[1]],
                               [vertices[1], vertices[2]],
                               [vertices[2], vertices[0]]])

    result = _pole_point_inside_polygon_cartesian('North', face_edge_cart)
    assert not result, "North pole should not be inside the polygon"

    result = _pole_point_inside_polygon_cartesian('South', face_edge_cart)
    assert result, "South pole should be inside the polygon"


def test_to_gdf_geodataframe(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    gdf_with_am = uxgrid.to_geodataframe(exclude_antimeridian=False)

    gdf_without_am = uxgrid.to_geodataframe(exclude_antimeridian=True)

    assert len(gdf_with_am) >= len(gdf_without_am)


def test_cache_and_override_geodataframe(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    gdf_a = uxgrid.to_geodataframe(exclude_antimeridian=False)

    gdf_b = uxgrid.to_geodataframe(exclude_antimeridian=False)

    # Should be the same object (cached)
    gdf_c = uxgrid.to_geodataframe(exclude_antimeridian=True)

    # Should be different from gdf_a and gdf_b
    gdf_d = uxgrid.to_geodataframe(exclude_antimeridian=True)

    gdf_e = uxgrid.to_geodataframe(exclude_antimeridian=True, override=True, cache=False)

    gdf_f = uxgrid.to_geodataframe(exclude_antimeridian=True)

    # gdf_a and gdf_b should be the same (cached)
    assert gdf_a is gdf_b

    # gdf_c and gdf_d should be the same (cached)
    assert gdf_c is gdf_d

    # gdf_e should be different (no cache)
    assert gdf_e is not gdf_c

    # gdf_f should be the same as gdf_c (cached)
    assert gdf_f is gdf_c
