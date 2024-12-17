import os
import numpy as np
import numpy.testing as nt
from pathlib import Path
import uxarray as ux
import pytest
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.arcs import point_within_gca

try:
    import constants
except ImportError:
    from . import constants

# Data files
current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_exo_CSne8 = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
gridfile_scrip_CSne8 = current_path / 'meshfiles' / "scrip" / "outCSne8" / 'outCSne8.nc'
gridfile_geoflowsmall_grid = current_path / 'meshfiles' / "ugrid" / "geoflow-small" / 'grid.nc'
gridfile_geoflowsmall_var = current_path / 'meshfiles' / "ugrid" / "geoflow-small" / 'v1.nc'

def test_pt_within_gcr():
    # The GCR that's exactly 180 degrees will raise a ValueError
    gcr_180degree_cart = np.asarray([
        _lonlat_rad_to_xyz(0.0, np.pi / 2.0),
        _lonlat_rad_to_xyz(0.0, -np.pi / 2.0)
    ])
    pt_same_lon_in = np.asarray(_lonlat_rad_to_xyz(0.0, 0.0))

    with pytest.raises(ValueError):
        point_within_gca(pt_same_lon_in, gcr_180degree_cart[0],gcr_180degree_cart[1] )
    #
    # Test when the point and the GCA all have the same longitude
    gcr_same_lon_cart = np.asarray([
        _lonlat_rad_to_xyz(0.0, 1.5),
        _lonlat_rad_to_xyz(0.0, -1.5)
    ])
    pt_same_lon_in = np.asarray(_lonlat_rad_to_xyz(0.0, 0.0))
    assert(point_within_gca(pt_same_lon_in, gcr_same_lon_cart[0], gcr_same_lon_cart[1]))

    pt_same_lon_out = np.asarray(_lonlat_rad_to_xyz(0.0, 1.5000001))
    res = point_within_gca(pt_same_lon_out, gcr_same_lon_cart[0], gcr_same_lon_cart[1])
    assert not res

    pt_same_lon_out_2 = np.asarray(_lonlat_rad_to_xyz(0.1, 1.0))
    res = point_within_gca(pt_same_lon_out_2, gcr_same_lon_cart[0], gcr_same_lon_cart[1])
    assert not res

def test_pt_within_gcr_antimeridian():
    # GCR vertex0 in radian : [5.163808182822441, 0.6351384888657234],
    # GCR vertex1 in radian : [0.8280410325693055, 0.42237025187091526]
    # Point in radian : [0.12574759138415173, 0.770098701904903]
    gcr_cart = np.array([[0.351, -0.724, 0.593], [0.617, 0.672, 0.410]])
    pt_cart = np.array(
        [0.9438777657502077, 0.1193199333436068, 0.922714737029319])
    assert(
        point_within_gca(pt_cart, gcr_cart[0], gcr_cart[1]))

    gcr_cart_flip = np.array([[0.617, 0.672, 0.410], [0.351, -0.724,
                                                        0.593]])
    # If we flip the gcr in the undirected mode, it should still work
    assert(
        point_within_gca(pt_cart, gcr_cart_flip[0], gcr_cart_flip[1]))

    # 2nd anti-meridian case
    # GCR vertex0 in radian : [4.104711496596806, 0.5352983676533828],
    # GCR vertex1 in radian : [2.4269979227622533, -0.007003212877856825]
    # Point in radian : [0.43400375562899113, -0.49554509841586936]
    gcr_cart_1 = np.array([[-0.491, -0.706, 0.510], [-0.755, 0.655,
                                                        -0.007]])
    pt_cart_within = np.array(
        [0.6136726305712109, 0.28442243941920053, -0.365605190899831])
    assert( not
        point_within_gca(pt_cart_within, gcr_cart_1[0], gcr_cart_1[1]))

def test_pt_within_gcr_cross_pole():
    gcr_cart = np.array([[0.351, 0.0, 0.3], [-0.351, 0.0, 0.3]])
    pt_cart = np.array([0.10, 0.0, 0.8])

    # Normalize the point abd the GCA
    pt_cart = pt_cart / np.linalg.norm(pt_cart)
    gcr_cart = np.array([x / np.linalg.norm(x) for x in gcr_cart])
    assert(point_within_gca(pt_cart, gcr_cart[0], gcr_cart[1]))
