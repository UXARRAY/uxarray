import numpy as np
from unittest import TestCase
from uxarray.grid.coordinates import node_lonlat_rad_to_xyz, node_xyz_to_lonlat_rad
from uxarray.grid.intersections import get_GCA_GCA_intersection


class TestGCAGCAIntersection(TestCase):

    def test_get_GCA_GCA_intersections_antimeridian(self):
        GCA1 = node_lonlat_rad_to_xyz([np.deg2rad(170.0), np.deg2rad(89.99)])
        GCR1_cart = np.array([
            node_lonlat_rad_to_xyz([np.deg2rad(170.0),
                                    np.deg2rad(89.99)]),
            node_lonlat_rad_to_xyz([np.deg2rad(170.0),
                                    np.deg2rad(10.0)])
        ])
        GCR2_cart = np.array([
            node_lonlat_rad_to_xyz([np.deg2rad(70.0), 0.0]),
            node_lonlat_rad_to_xyz([np.deg2rad(179.0), 0.0])
        ])
        res_cart = get_GCA_GCA_intersection(GCR1_cart, GCR2_cart)

        # res_cart should be [-1, -1, -1] since these two GCRs are not intersecting
        self.assertTrue(np.array_equal(res_cart, np.array([-1, -1, -1])))

        GCR1_cart = np.array([
            node_lonlat_rad_to_xyz([np.deg2rad(170.0),
                                    np.deg2rad(89.0)]),
            node_lonlat_rad_to_xyz([np.deg2rad(170.0),
                                    np.deg2rad(-10.0)])
        ])
        GCR2_cart = np.array([
            node_lonlat_rad_to_xyz([np.deg2rad(70.0), 0.0]),
            node_lonlat_rad_to_xyz([np.deg2rad(175.0), 0.0])
        ])

        res_cart = get_GCA_GCA_intersection(GCR1_cart, GCR2_cart)
        res_lonlat_rad = node_xyz_to_lonlat_rad(res_cart.tolist())

        # res_cart should be [170, 0]
        self.assertTrue(
            np.array_equal(res_lonlat_rad,
                           np.array([np.deg2rad(170.0),
                                     np.deg2rad(0.0)])))

    def test_get_GCA_GCA_intersections_parallel(self):
        GCR1_cart = np.array([
            node_lonlat_rad_to_xyz([0.3 * np.pi, 0.0]),
            node_lonlat_rad_to_xyz([0.5 * np.pi, 0.0])
        ])
        GCR2_cart = np.array([
            node_lonlat_rad_to_xyz([0.5 * np.pi, 0.0]),
            node_lonlat_rad_to_xyz([-0.5 * np.pi - 0.01, 0.0])
        ])
        res_cart = get_GCA_GCA_intersection(GCR1_cart, GCR2_cart)
        self.assertTrue(np.allclose(res_cart, 0.0))

    def test_get_GCA_GCA_intersections_perpendicular(self):
        GCR1_cart = np.array([
            node_lonlat_rad_to_xyz([np.deg2rad(170.0),
                                    np.deg2rad(0.0)]),
            node_lonlat_rad_to_xyz([np.deg2rad(170.0),
                                    np.deg2rad(10.0)])
        ])
        GCR2_cart = np.array([
            node_lonlat_rad_to_xyz([0.5 * np.pi, 0.0]),
            node_lonlat_rad_to_xyz([-0.5 * np.pi - 0.01, 0.0])
        ])
        res_cart = get_GCA_GCA_intersection(GCR1_cart, GCR2_cart)
        res_lonlat_rad = node_xyz_to_lonlat_rad(res_cart.tolist())
        self.assertTrue(
            np.allclose(res_lonlat_rad,
                        np.array([np.deg2rad(170.0),
                                  np.deg2rad(0.0)])))
