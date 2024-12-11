import numpy as np
import pytest
import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE
from uxarray.grid.arcs import _extreme_gca_latitude_cartesian
from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _xyz_to_lonlat_rad
from uxarray.grid.intersections import gca_const_lat_intersection, _gca_gca_intersection_cartesian

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
    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(0.0)),
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(10.0))
    ])
    GCR2_cart = np.array([
        _lonlat_rad_to_xyz(*[0.5 * np.pi, 0.0]),
        _lonlat_rad_to_xyz(*[-0.5 * np.pi - 0.01, 0.0])
    ])
    res_cart = _gca_gca_intersection_cartesian(GCR1_cart, GCR2_cart)
    res_cart = res_cart[0]

    assert np.allclose(np.linalg.norm(res_cart, axis=0), 1.0, atol=ERROR_TOLERANCE)
    res_lonlat_rad = _xyz_to_lonlat_rad(*res_cart)
    assert np.allclose(res_lonlat_rad, np.array([np.deg2rad(170.0), np.deg2rad(0.0)]))

def test_GCA_constLat_intersections_antimeridian():
    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(89.99)),
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(10.0))
    ])

    res = gca_const_lat_intersection(GCR1_cart, np.sin(np.deg2rad(60.0)), verbose=True)
    res_lonlat_rad = _xyz_to_lonlat_rad(*(res[0].tolist()))
    assert np.allclose(res_lonlat_rad, np.array([np.deg2rad(170.0), np.deg2rad(60.0)]))

def test_GCA_constLat_intersections_empty():
    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(89.99)),
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(10.0))
    ])

    res = gca_const_lat_intersection(GCR1_cart, np.sin(np.deg2rad(-10.0)), verbose=False)
    assert res.size == 0

def test_GCA_constLat_intersections_two_pts():
    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(10.0), np.deg2rad(10)),
        _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(10.0))
    ])
    max_lat = _extreme_gca_latitude_cartesian(GCR1_cart, 'max')

    query_lat = (np.deg2rad(10.0) + max_lat) / 2.0

    res = gca_const_lat_intersection(GCR1_cart, np.sin(query_lat), verbose=False)
    assert res.shape[0] == 2

def test_GCA_constLat_intersections_no_converge():
    GCR1_cart = np.array([[-0.59647278, 0.59647278, -0.53706651],
                          [-0.61362973, 0.61362973, -0.49690755]])

    constZ = -0.5150380749100542

    with pytest.warns(UserWarning):
        res = gca_const_lat_intersection(GCR1_cart, constZ, verbose=False)
        assert res.shape[0] == 1
