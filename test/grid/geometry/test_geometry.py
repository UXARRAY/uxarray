import numpy as np
import pytest
import xarray as xr

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _normalize_xyz
from uxarray.grid.geometry import haversine_distance, _pole_point_inside_polygon_cartesian
from uxarray.grid.point_in_face import _face_contains_point
from uxarray.grid.utils import _get_cartesian_face_edge_nodes_array


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



def test_face_at_antimeridian():
    """Test the function `point_in_face`, where the face crosses the antimeridian"""

    # Generate a face crossing the antimeridian
    vertices_lonlat = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)
    point = np.array(_lonlat_rad_to_xyz(np.deg2rad(25), np.deg2rad(30)))

    # Create the grid and face edges
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    faces_edges_cartesian = _get_cartesian_face_edge_nodes_array(
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_edges,
        grid.node_x.values,
        grid.node_y.values,
        grid.node_z.values,
    )

    assert _face_contains_point(faces_edges_cartesian[0], point)


def test_face_at_pole():
    """Test the function `point_in_face`, when the face is at the North Pole"""

    # Generate a face that is at a pole
    vertices_lonlat = [[10.0, 90.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    point = np.array(_lonlat_rad_to_xyz(np.deg2rad(25), np.deg2rad(30)))

    # Create the grid and face edges
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    faces_edges_cartesian = _get_cartesian_face_edge_nodes_array(
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_edges,
        grid.node_x.values,
        grid.node_y.values,
        grid.node_z.values,
    )

    assert _face_contains_point(faces_edges_cartesian[0], point)


def test_face_normal_face():
    """Test the function `point_in_face`, where the face is a normal face, not crossing the antimeridian or the
    poles"""

    # Generate a normal face that is not crossing the antimeridian or the poles
    vertices_lonlat = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)
    point = np.array(_lonlat_rad_to_xyz(np.deg2rad(25), np.deg2rad(30)))

    # Create the grid and face edges
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    faces_edges_cartesian = _get_cartesian_face_edge_nodes_array(
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_edges,
        grid.node_x.values,
        grid.node_y.values,
        grid.node_z.values,
    )

    assert _face_contains_point(faces_edges_cartesian[0], point)


def test_haversine_distance_creation():
    """Test the haversine distance function"""
    # Test points
    lon1, lat1 = 0.0, 0.0  # Point 1: (0°, 0°)
    lon2, lat2 = 90.0, 0.0  # Point 2: (90°, 0°)

    # Convert to radians
    lon1_rad, lat1_rad = np.deg2rad(lon1), np.deg2rad(lat1)
    lon2_rad, lat2_rad = np.deg2rad(lon2), np.deg2rad(lat2)

    # Calculate haversine distance
    distance = haversine_distance(lon1_rad, lat1_rad, lon2_rad, lat2_rad)

    # Expected distance is 1/4 of Earth circumference (π/2 radians)
    expected_distance = np.pi / 2

    np.testing.assert_allclose(distance, expected_distance, atol=ERROR_TOLERANCE)


def test_engine_geodataframe(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc"))
    for engine in ["geopandas", "spatialpandas"]:
        gdf = uxgrid.to_geodataframe(engine=engine)


def test_periodic_elements_geodataframe(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc"))
    for periodic_elements in ["ignore", "exclude", "split"]:
        gdf = uxgrid.to_geodataframe(periodic_elements=periodic_elements)


def test_geodataframe_crashes_when_nnode_equals_nface():
    """Ensure UxDataArray.to_geodataframe() crashes for non-face_centered data, even if n_node==n_face.
    regression test for issue #1616.
    """
    # Below is a visualization of the example here, with ni node i, fj face j:
    # (n1)----------------(n2)
    #  | %%    f1        =%/|
    #  |  %%           =% / |
    #  |   %%       =%   /  |
    #  |    %%   =%  f2 /   |
    #  |     (n3)----(n4) f3|
    #  | f0  %%  =% f4  \   |
    #  |   %%       =%    \ |
    #  | %%   f5       =%  \|
    # (n0)----------------(n5)
    XA, XB, XC, XF = 0, 10, 20, 30
    YA, YB, YF = 0, 10, 25
    node_lonlat = [
        [XA, YA],  # n0
        [XA, YF],  # n1
        [XF, YF],  # n2
        [XB, YB],  # n3
        [XC, YB],  # n4
        [XF, YA],  # n5
    ]
    node_lon = [n[0] for n in node_lonlat]
    node_lat = [n[1] for n in node_lonlat]
    face_node_connectivity = [
        [0, 1, 3],  # f0
        [1, 3, 2],  # f1
        [3, 4, 2],  # f2
        [4, 5, 2],  # f3
        [3, 5, 4],  # f4
        [0, 5, 3],  # f5
    ]
    # convert to xarray
    ds_vars = {
        'node_lon': xr.DataArray(node_lon, dims=['n_node']),
        'node_lat': xr.DataArray(node_lat, dims=['n_node']),
        'face_node_connectivity': xr.DataArray(face_node_connectivity, dims=['n_face', 'n_max_face_nodes']),
    }
    ds = xr.Dataset(ds_vars)
    # convert to uxarray
    uxgrid = ux.Grid(ds, 'UGRID')  # putting 'UGRID' avoids raising warning but does not affect results.
    assert uxgrid.n_node == uxgrid.n_face
    # make array of values, convert to uxarray
    vals = xr.DataArray([100,200,300,400,500,600], dims=['n_node'])
    uxarr = ux.UxDataArray(vals, uxgrid=uxgrid)
    # ensure to_geodataframe crashes for non-face_centered data, even if n_node==n_face
    with pytest.raises(ValueError):
        uxarr.to_geodataframe()
