import numpy as np
from unittest import TestCase
import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE

# from uxarray.grid.coordinates import node_lonlat_rad_to_xyz, node_xyz_to_lonlat_rad

from uxarray.grid.arcs import _extreme_gca_latitude_cartesian
from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _xyz_to_lonlat_rad,_xyz_to_lonlat_rad_scalar
from uxarray.grid.intersections import gca_gca_intersection, gca_const_lat_intersection, _gca_gca_intersection_cartesian


class TestGCAGCAIntersection(TestCase):

    def test_get_GCA_GCA_intersections_antimeridian(self):
        # Test the case where the two GCAs are on the antimeridian
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
        res_cart = _gca_gca_intersection_cartesian(GCR1_cart, GCR2_cart)

        # res_cart should be empty since these two GCRs are not intersecting
        self.assertTrue(len(res_cart) == 0)

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

        res_cart = _gca_gca_intersection_cartesian(GCR1_cart, GCR2_cart)
        res_cart = res_cart[0]

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
        res_cart = _gca_gca_intersection_cartesian(GCR1_cart, GCR2_cart)
        res_cart = res_cart[0]
        expected_res = np.array(_lonlat_rad_to_xyz(0.5 * np.pi, 0.0))
        # Test if two results are equal within the error tolerance
        self.assertAlmostEqual(np.linalg.norm(res_cart - expected_res), 0.0, delta=ERROR_TOLERANCE)

    def test_get_GCA_GCA_intersections_perpendicular(self):
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
        self.assertTrue(len(res_cart) == 0)


    def test_GCA_GCA_pole(self):
        face_lonlat = np.deg2rad(np.array([-175, 26.5]))

        # this fails when the pole is set to exactly -90.0
        ref_point_lonlat = np.deg2rad(np.array([0.0, -89.9]))
        face_xyz = np.array(_lonlat_rad_to_xyz(*face_lonlat))
        ref_point_xyz = np.array(_lonlat_rad_to_xyz(*ref_point_lonlat))

        edge_a_lonlat = np.deg2rad(np.array((-175, -24.5)))
        edge_b_lonlat = np.deg2rad(np.array((-173, 25.7)))

        edge_a_xyz = np.array(_lonlat_rad_to_xyz(*edge_a_lonlat))
        edge_b_xyz = np.array(_lonlat_rad_to_xyz(*edge_b_lonlat))

        gca_a_xyz = np.array([face_xyz, ref_point_xyz])

        gca_b_xyz = np.array([edge_a_xyz, edge_b_xyz])

        # The edge should intersect
        self.assertTrue(len(gca_gca_intersection(gca_a_xyz, gca_b_xyz)))




class TestGCAconstLatIntersection(TestCase):

    def test_GCA_constLat_intersections_antimeridian(self):
        GCR1_cart = np.array([
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                              np.deg2rad(89.99)),
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                              np.deg2rad(10.0))
        ])

        res = gca_const_lat_intersection(GCR1_cart, np.sin(np.deg2rad(60.0)), verbose=True)
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

        res = gca_const_lat_intersection(GCR1_cart, np.sin(np.deg2rad(-10.0)), verbose=False)
        self.assertTrue(res.size == 0)

    def test_GCA_constLat_intersections_two_pts(self):
        GCR1_cart = np.array([
            _lonlat_rad_to_xyz(np.deg2rad(10.0),
                                    np.deg2rad(10)),
            _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                    np.deg2rad(10.0))
        ])
        max_lat = _extreme_gca_latitude_cartesian(GCR1_cart, 'max')

        query_lat = (np.deg2rad(10.0) + max_lat) / 2.0

        res = gca_const_lat_intersection(GCR1_cart, np.sin(query_lat), verbose=False)
        self.assertTrue(res.shape[0] == 2)


    def test_GCA_constLat_intersections_no_convege(self):
        # It should return an one single point and a warning about unable to be converged should be raised
        GCR1_cart = np.array([[-0.59647278, 0.59647278, -0.53706651],
                              [-0.61362973, 0.61362973, -0.49690755]])

        constZ = -0.5150380749100542

        with self.assertWarns(UserWarning):
            res = gca_const_lat_intersection(GCR1_cart, constZ, verbose=False)
            self.assertTrue(res.shape[0] == 1)
