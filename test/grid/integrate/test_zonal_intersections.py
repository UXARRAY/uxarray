import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.integrate import _get_faces_constLat_intersection_info


def test_get_faces_constLat_intersection_info_one_intersection():
    face_edges_cart = np.array([
        [[-5.4411371445381629e-01, -4.3910468172333759e-02, -8.3786164521844386e-01],
         [-5.4463903501502697e-01, -6.6699045092185599e-17, -8.3867056794542405e-01]],

        [[-5.4463903501502697e-01, -6.6699045092185599e-17, -8.3867056794542405e-01],
         [-4.9999999999999994e-01, -6.1232339957367648e-17, -8.6602540378443871e-01]],

        [[-4.9999999999999994e-01, -6.1232339957367648e-17, -8.6602540378443871e-01],
         [-4.9948581138450826e-01, -4.5339793804534498e-02, -8.6513480297773349e-01]],

        [[-4.9948581138450826e-01, -4.5339793804534498e-02, -8.6513480297773349e-01],
         [-5.4411371445381629e-01, -4.3910468172333759e-02, -8.3786164521844386e-01]]
    ])

    latitude_cart = -0.8660254037844386
    is_latlonface = False
    is_GCA_list = None
    unique_intersections, pt_lon_min, pt_lon_max = _get_faces_constLat_intersection_info(face_edges_cart, latitude_cart,
                                                                                         is_GCA_list, is_latlonface)
    assert len(unique_intersections) == 1


def test_get_faces_constLat_intersection_info_encompass_pole():
    face_edges_cart = np.array([
        [[0.03982285692494229, 0.00351700770436231, 0.9992005658140627],
         [0.00896106681877875, 0.03896060263227105, 0.9992005658144913]],

        [[0.00896106681877875, 0.03896060263227105, 0.9992005658144913],
         [-0.03428461218295055, 0.02056197086916728, 0.9992005658132106]],

        [[-0.03428461218295055, 0.02056197086916728, 0.9992005658132106],
         [-0.03015012448894485, -0.02625260499902213, 0.9992005658145248]],

        [[-0.03015012448894485, -0.02625260499902213, 0.9992005658145248],
         [0.01565081128889155, -0.03678697293262131, 0.9992005658167203]],

        [[0.01565081128889155, -0.03678697293262131, 0.9992005658167203],
         [0.03982285692494229, 0.00351700770436231, 0.9992005658140627]]
    ])

    latitude_cart = 0.9993908270190958
    latitude_rad = np.arcsin(latitude_cart)
    latitude_deg = np.rad2deg(latitude_rad)
    print(latitude_deg)

    is_latlonface = False
    is_GCA_list = None
    unique_intersections, pt_lon_min, pt_lon_max = _get_faces_constLat_intersection_info(face_edges_cart, latitude_cart,
                                                                                         is_GCA_list, is_latlonface)
    assert len(unique_intersections) <= 2 * len(face_edges_cart)


def test_get_faces_constLat_intersection_info_on_pole():
    face_edges_cart = np.array([
        [[-5.2264427688714095e-02, -5.2264427688714102e-02, -9.9726468863423734e-01],
         [-5.2335956242942412e-02, -6.4093061293235361e-18, -9.9862953475457394e-01]],

        [[-5.2335956242942412e-02, -6.4093061293235361e-18, -9.9862953475457394e-01],
         [6.1232339957367660e-17, 0.0000000000000000e+00, -1.0000000000000000e+00]],

        [[6.1232339957367660e-17, 0.0000000000000000e+00, -1.0000000000000000e+00],
         [3.2046530646617680e-18, -5.2335956242942412e-02, -9.9862953475457394e-01]],

        [[3.2046530646617680e-18, -5.2335956242942412e-02, -9.9862953475457394e-01],
         [-5.2264427688714095e-02, -5.2264427688714102e-02, -9.9726468863423734e-01]]
    ])
    latitude_cart = -0.9998476951563913
    is_latlonface = False
    is_GCA_list = None
    unique_intersections, pt_lon_min, pt_lon_max = _get_faces_constLat_intersection_info(face_edges_cart, latitude_cart,
                                                                                         is_GCA_list, is_latlonface)
    assert len(unique_intersections) == 2


def test_get_faces_constLat_intersection_info_near_pole():
    face_edges_cart = np.array([
        [[-5.1693346290592648e-02, 1.5622531297347531e-01, -9.8636780641686628e-01],
         [-5.1195320928843470e-02, 2.0763904784932552e-01, -9.7686491641537532e-01]],
        [[-5.1195320928843470e-02, 2.0763904784932552e-01, -9.7686491641537532e-01],
         [1.2730919333264125e-17, 2.0791169081775882e-01, -9.7814760073380580e-01]],
        [[1.2730919333264125e-17, 2.0791169081775882e-01, -9.7814760073380580e-01],
         [9.5788483443923397e-18, 1.5643446504023048e-01, -9.8768834059513777e-01]],
        [[9.5788483443923397e-18, 1.5643446504023048e-01, -9.8768834059513777e-01],
         [-5.1693346290592648e-02, 1.5622531297347531e-01, -9.8636780641686628e-01]]
    ])

    latitude_cart = -0.9876883405951378
    latitude_rad = np.arcsin(latitude_cart)
    latitude_deg = np.rad2deg(latitude_rad)
    is_latlonface = False
    is_GCA_list = None
    unique_intersections, pt_lon_min, pt_lon_max = _get_faces_constLat_intersection_info(face_edges_cart, latitude_cart,
                                                                                         is_GCA_list, is_latlonface)
    assert len(unique_intersections) == 1
