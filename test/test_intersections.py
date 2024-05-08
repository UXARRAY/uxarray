import numpy as np
from unittest import TestCase
import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE

# from uxarray.grid.coordinates import node_lonlat_rad_to_xyz, node_xyz_to_lonlat_rad

from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _xyz_to_lonlat_rad
from uxarray.grid.intersections import gca_gca_intersection, gca_constLat_intersection


class TestGCAGCAIntersection(TestCase):

    def test_get_GCA_GCA_intersections_antimeridian(self):
        # Test the case where the two GCAs are on the antimeridian




        GCA1 = _lonlat_rad_to_xyz(np.deg2rad(170.0), np.deg2rad(89.99))
        GCR1_cart = np.array([
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                    np.deg2rad(89.99)),
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                    np.deg2rad(10.0))
        ])
        GCR2_cart = np.array([
            _lonlat_rad_to_xyz(np.deg2rad(70.0), 0.0),
            _lonlat_rad_to_xyz(np.deg2rad(179.0), 0.0)
        ])
        res_cart = gca_gca_intersection(GCR1_cart, GCR2_cart)

        # res_cart should be empty since these two GCRs are not intersecting
        self.assertTrue(np.array_equal(res_cart, np.array([])))

        GCR1_cart = np.array([
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                    np.deg2rad(89.0)),
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                    np.deg2rad(-10.0))
        ])
        GCR2_cart = np.array([
            _lonlat_rad_to_xyz(np.deg2rad(70.0), 0.0),
            _lonlat_rad_to_xyz(np.deg2rad(175.0), 0.0)
        ])

        res_cart = gca_gca_intersection(GCR1_cart, GCR2_cart)

        # Test if the result is normalized
        self.assertTrue(
            np.allclose(np.linalg.norm(res_cart, axis=0),
                        1.0,
                        atol=ERROR_TOLERANCE))
        res_lonlat_rad = _xyz_to_lonlat_rad(res_cart[0], res_cart[1], res_cart[2])

        # res_cart should be [170, 0]
        self.assertTrue(
            np.array_equal(res_lonlat_rad,
                           np.array([np.deg2rad(170.0),
                                     np.deg2rad(0.0)])))

    def test_get_GCA_GCA_intersections_parallel(self):
        # Test the case where the two GCAs are parallel
        GCR1_cart = np.array([
            _lonlat_rad_to_xyz(0.3 * np.pi, 0.0),
            _lonlat_rad_to_xyz(0.5 * np.pi, 0.0)
        ])
        GCR2_cart = np.array([
            _lonlat_rad_to_xyz(0.5 * np.pi, 0.0),
            _lonlat_rad_to_xyz(-0.5 * np.pi - 0.01, 0.0)
        ])
        res_cart = gca_gca_intersection(GCR1_cart, GCR2_cart)
        self.assertTrue(np.array_equal(res_cart, np.array([])))

    def test_get_GCA_GCA_intersections_perpendicular(self):
        # Test the case where the two GCAs are perpendicular to each other
        GCR1_cart = np.array([
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                    np.deg2rad(0.0)),
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                    np.deg2rad(10.0))
        ])
        GCR2_cart = np.array([
            _lonlat_rad_to_xyz(*[0.5 * np.pi, 0.0]),
            _lonlat_rad_to_xyz(*[-0.5 * np.pi - 0.01, 0.0])
        ])
        res_cart = gca_gca_intersection(GCR1_cart, GCR2_cart)

        # Test if the result is normalized
        self.assertTrue(
            np.allclose(np.linalg.norm(res_cart, axis=0),
                        1.0,
                        atol=ERROR_TOLERANCE))
        res_lonlat_rad = _xyz_to_lonlat_rad(*res_cart)
        self.assertTrue(
            np.allclose(res_lonlat_rad,
                        np.array([np.deg2rad(170.0),
                                  np.deg2rad(0.0)])))


class TestGCAconstLatIntersection(TestCase):

    def test_GCA_constLat_intersections_antimeridian(self):
        GCR1_cart = np.array([
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                              np.deg2rad(89.99)),
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                              np.deg2rad(10.0))
        ])

        res = gca_constLat_intersection(GCR1_cart, np.sin(np.deg2rad(60.0)), verbose=True)
        res_lonlat_rad = _xyz_to_lonlat_rad(*(res[0].tolist()))
        self.assertTrue(
            np.allclose(res_lonlat_rad,
                        np.array([np.deg2rad(170.0),
                                  np.deg2rad(60.0)])))

    def test_GCA_constLat_intersections_empty(self):
        GCR1_cart = np.array([
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                    np.deg2rad(89.99)),
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                    np.deg2rad(10.0))
        ])

        res = gca_constLat_intersection(GCR1_cart, np.sin(np.deg2rad(-10.0)), verbose=False)
        self.assertTrue(res.size == 0)

    def test_GCA_constLat_intersections_two_pts(self):
        GCR1_cart = np.array([
            _lonlat_rad_to_xyz(np.deg2rad(10.0),
                                    np.deg2rad(10)),
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                    np.deg2rad(10.0))
        ])
        max_lat = ux.grid.arcs.extreme_gca_latitude(GCR1_cart, 'max')

        query_lat = (np.deg2rad(10.0) + max_lat) / 2.0

        res = gca_constLat_intersection(GCR1_cart, np.sin(query_lat), verbose=False)
        self.assertTrue(res.shape[0] == 2)
