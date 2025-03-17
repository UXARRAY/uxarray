import numpy as np
import pytest
import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE
from uxarray.grid.arcs import extreme_gca_z
from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _xyz_to_lonlat_rad,_xyz_to_lonlat_rad_scalar
from uxarray.grid.intersections import gca_gca_intersection, gca_const_lat_intersection, _gca_gca_intersection_cartesian, get_number_of_intersections

def test_get_GCA_GCA_intersections_antimeridian():
    GCA1 = _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(89.99))
    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(89.99)),
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(10.0))
    ])
    GCR2_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(70.0), 0.0),
        _lonlat_rad_to_xyz(np.deg2rad(179.0), 0.0)
    ])
    res_cart = _gca_gca_intersection_cartesian(GCR1_cart, GCR2_cart)

    assert len(res_cart) == 0

    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(89.0)),
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(-10.0))
    ])


    GCR2_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(70.0), 0.0),
        _lonlat_rad_to_xyz(np.deg2rad(175.0), 0.0)
    ])

    res_cart = _gca_gca_intersection_cartesian(GCR1_cart, GCR2_cart)
    res_cart = res_cart[0]

    assert np.allclose(np.linalg.norm(res_cart, axis=0), 1.0, atol=ERROR_TOLERANCE)
    res_lonlat_rad = _xyz_to_lonlat_rad(res_cart[0], res_cart[1], res_cart[2])

    assert np.array_equal(res_lonlat_rad, np.array([np.deg2rad(170.0), np.deg2rad(0.0)]))

def test_get_GCA_GCA_intersections_parallel():
    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(0.3 * np.pi, 0.0),
        _lonlat_rad_to_xyz(0.5 * np.pi, 0.0)
    ])
    GCR2_cart = np.array([
        _lonlat_rad_to_xyz(0.5 * np.pi, 0.0),
        _lonlat_rad_to_xyz(-0.5 * np.pi - 0.01, 0.0)
    ])
    res_cart = _gca_gca_intersection_cartesian(GCR1_cart, GCR2_cart)
    res_cart = res_cart[0]
    expected_res = np.array(_lonlat_rad_to_xyz(0.5 * np.pi, 0.0))

    assert np.isclose(np.linalg.norm(res_cart - expected_res), 0.0, atol=ERROR_TOLERANCE)

def test_get_GCA_GCA_intersections_perpendicular():
    # Test the case where the two GCAs are perpendicular to each other
    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                np.deg2rad(0.0)),
        _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                np.deg2rad(10.0))
    ])
    GCR2_cart = np.array([
        _lonlat_rad_to_xyz(*[0.5 * np.pi - 0.01, 0.0]),
        _lonlat_rad_to_xyz(*[-0.5 * np.pi + 0.01, 0.0])
    ])
    res_cart = _gca_gca_intersection_cartesian(GCR1_cart, GCR2_cart)

    # rest_cart should be empty since these two GCAs are not intersecting
    assert(len(res_cart) == 0)


    # def test_GCA_GCA_single_edge_to_pole(self):
    #     # GCA_a - Face Center connected to South Pole
    #     # Point A - South Pole
    #     ref_point_lonlat = np.deg2rad(np.array([0.0, -90.0]))
    #     ref_point_xyz = np.array(_lonlat_rad_to_xyz(*ref_point_lonlat))
    #     # Point B - Face Center
    #     face_lonlat = np.deg2rad(np.array([-175, 26.5]))
    #     face_xyz = np.array(_lonlat_rad_to_xyz(*face_lonlat))
    #     gca_a_xyz = np.array([face_xyz, ref_point_xyz])
    #
    #     # GCA_b - Single Face Edge
    #     # Point A - First Edge Point
    #     edge_a_lonlat = np.deg2rad(np.array((-175, -24.5)))
    #     edge_b_lonlat = np.deg2rad(np.array((-173, 28.7)))
    #
    #     # Point B - Second Edge Point
    #     edge_a_xyz = np.array(_lonlat_rad_to_xyz(*edge_a_lonlat))
    #     edge_b_xyz = np.array(_lonlat_rad_to_xyz(*edge_b_lonlat))
    #     gca_b_xyz = np.array([edge_a_xyz, edge_b_xyz])
    #
    #     # The edge should intersect
    #     self.assertTrue(len(gca_gca_intersection(gca_a_xyz, gca_b_xyz)))

def test_GCA_GCA_south_pole():

    # GCA_a - Face Center connected to South Pole
    # Point A - South Pole
    ref_point_lonlat = np.deg2rad(np.array([0.0, -90.0]))
    ref_point_xyz = np.array(_lonlat_rad_to_xyz(*ref_point_lonlat))
    # Point B - Face Center
    face_lonlat = np.deg2rad(np.array([0.0, 0.0]))
    face_xyz = np.array(_lonlat_rad_to_xyz(*face_lonlat))
    gca_a_xyz = np.array([face_xyz, ref_point_xyz])

    # GCA_b - Single Face Edge
    # Point A - First Edge Point
    edge_a_lonlat = np.deg2rad(np.array((-45, -1.0)))
    edge_b_lonlat = np.deg2rad(np.array((45, -1.0)))

    # Point B - Second Edge Point
    edge_a_xyz = np.array(_lonlat_rad_to_xyz(*edge_a_lonlat))
    edge_b_xyz = np.array(_lonlat_rad_to_xyz(*edge_b_lonlat))
    gca_b_xyz = np.array([edge_a_xyz, edge_b_xyz])

    # The edge should intersect
    assert(len(gca_gca_intersection(gca_a_xyz, gca_b_xyz)))

def test_GCA_GCA_north_pole():
    # GCA_a - Face Center connected to South Pole
    ref_point_lonlat = np.deg2rad(np.array([0.0, 90.0]))
    ref_point_xyz = np.array(_lonlat_rad_to_xyz(*ref_point_lonlat))
    face_lonlat = np.deg2rad(np.array([0.0, 0.0]))
    face_xyz = np.array(_lonlat_rad_to_xyz(*face_lonlat))
    gca_a_xyz = np.array([face_xyz, ref_point_xyz])

    # GCA_b - Single Face Edge
    edge_a_lonlat = np.deg2rad(np.array((-45, 1.0)))
    edge_b_lonlat = np.deg2rad(np.array((45, 1.0)))

    edge_a_xyz = np.array(_lonlat_rad_to_xyz(*edge_a_lonlat))
    edge_b_xyz = np.array(_lonlat_rad_to_xyz(*edge_b_lonlat))
    gca_b_xyz = np.array([edge_a_xyz, edge_b_xyz])

    # The edge should intersect
    assert(len(gca_gca_intersection(gca_a_xyz, gca_b_xyz)))

def test_GCA_GCA_north_pole_angled():
    # GCA_a
    ref_point_lonlat = np.deg2rad(np.array([0.0, 90.0]))
    ref_point_xyz = np.array(_lonlat_rad_to_xyz(*ref_point_lonlat))
    face_lonlat = np.deg2rad(np.array([-45.0, 45.0]))
    face_xyz = np.array(_lonlat_rad_to_xyz(*face_lonlat))
    gca_a_xyz = np.array([face_xyz, ref_point_xyz])

    # GCA_b
    edge_a_lonlat = np.deg2rad(np.array((-45.0, 50.0)))
    edge_b_lonlat = np.deg2rad(np.array((-40.0, 45.0)))

    # Point B - Second Edge Point
    edge_a_xyz = np.array(_lonlat_rad_to_xyz(*edge_a_lonlat))
    edge_b_xyz = np.array(_lonlat_rad_to_xyz(*edge_b_lonlat))
    gca_b_xyz = np.array([edge_a_xyz, edge_b_xyz])

    # The edge should intersect
    assert(len(gca_gca_intersection(gca_a_xyz, gca_b_xyz)))


def test_GCA_edge_intersection_count():

    from uxarray.grid.utils import _get_cartesian_face_edge_nodes

    # Generate a normal face that is not crossing the antimeridian or the poles
    vertices_lonlat = [[29.5, 11.0], [29.5, 10.0], [30.5, 10.0], [30.5, 11.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edge_nodes_cartesian = _get_cartesian_face_edge_nodes(
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_edges,
        grid.node_x.values,
        grid.node_y.values,
        grid.node_z.values)

    face_center_xyz = np.array([grid.face_x.values[0], grid.face_y.values[0], grid.face_z.values[0]], dtype=np.float64)
    north_pole_xyz = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    south_pole_xyz = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    gca_face_center_north_pole = np.array([face_center_xyz, north_pole_xyz], dtype=np.float64)
    gca_face_center_south_pole = np.array([face_center_xyz, south_pole_xyz], dtype=np.float64)

    intersect_north_pole_count = 0
    intersect_south_pole_count = 0

    for edge in face_edge_nodes_cartesian[0]:
        res1 = gca_gca_intersection(edge, gca_face_center_north_pole)
        res2 = gca_gca_intersection(edge, gca_face_center_south_pole)

        if len(res1):
            intersect_north_pole_count += 1
        if len(res2):
            intersect_south_pole_count += 1

    print(intersect_north_pole_count, intersect_south_pole_count)
    assert(intersect_north_pole_count == 1)
    assert(intersect_south_pole_count == 1)

def test_GCA_GCA_single_edge_to_pole():
    # GCA_a - Face Center connected to South Pole
    # Point A - South Pole
    ref_point_lonlat_exact = np.deg2rad(np.array([0.0, -90.0]))
    ref_point_lonlat_close = np.deg2rad(np.array([0.0, -89.99999]))
    ref_point_xyz_exact = np.array(_lonlat_rad_to_xyz(*ref_point_lonlat_exact))
    ref_point_xyz_close = np.array(_lonlat_rad_to_xyz(*ref_point_lonlat_close))

    # Point B - Face Center
    face_lonlat = np.deg2rad(np.array([-175.0, 26.5]))
    face_xyz = np.array(_lonlat_rad_to_xyz(*face_lonlat))
    gca_a_xyz_close = np.array([face_xyz, ref_point_xyz_close])
    gca_a_xyz_exact = np.array([face_xyz, ref_point_xyz_exact])

    # GCA_b - Single Face Edge
    # Point A - First Edge Point
    edge_a_lonlat = np.deg2rad(np.array((-175.0, -24.5)))
    edge_b_lonlat = np.deg2rad(np.array((-173.0, 28.7)))

    # Point B - Second Edge Point
    edge_a_xyz = np.array(_lonlat_rad_to_xyz(*edge_a_lonlat))
    edge_b_xyz = np.array(_lonlat_rad_to_xyz(*edge_b_lonlat))
    gca_b_xyz = np.array([edge_a_xyz, edge_b_xyz])

    # The edge should intersect
    assert(len(gca_gca_intersection(gca_a_xyz_close, gca_b_xyz)))
    assert(len(gca_gca_intersection(gca_a_xyz_exact, gca_b_xyz)))

def test_GCA_constLat_intersections_antimeridian():
    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(89.99)),
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(10.0))
    ])

    res = gca_const_lat_intersection(GCR1_cart, np.sin(np.deg2rad(60.0)))
    res_lonlat_rad = _xyz_to_lonlat_rad(*(res[0].tolist()))
    assert np.allclose(res_lonlat_rad, np.array([np.deg2rad(170.0), np.deg2rad(60.0)]))

def test_GCA_constLat_intersections_empty():
    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(89.99)),
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(10.0))
    ])

    res = gca_const_lat_intersection(GCR1_cart, np.sin(np.deg2rad(-10.0)))
    assert get_number_of_intersections(res) == 0

def test_GCA_constLat_intersections_two_pts():
    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(10.0), np.deg2rad(10)),
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(10.0))
    ])
    max_lat = extreme_gca_z(GCR1_cart, 'max')

    query_lat = (np.deg2rad(10.0) + max_lat) / 2.0

    res = gca_const_lat_intersection(GCR1_cart, np.sin(query_lat))
    assert res.shape[0] == 2
