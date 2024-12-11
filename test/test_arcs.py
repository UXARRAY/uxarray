import os
import numpy as np
import numpy.testing as nt
from pathlib import Path
import uxarray as ux
import pytest
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.arcs import _point_within_gca_cartesian

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
    gcr_180degree_cart = [
        _lonlat_rad_to_xyz(0.0, np.pi / 2.0),
        _lonlat_rad_to_xyz(0.0, -np.pi / 2.0)
    ]

    pt_same_lon_in = _lonlat_rad_to_xyz(0.0, 0.0)
    with pytest.raises(ValueError):
        _point_within_gca_cartesian(pt_same_lon_in, gcr_180degree_cart)

    # Test when the point and the GCA all have the same longitude
    gcr_same_lon_cart = [
        _lonlat_rad_to_xyz(0.0, 1.5),
        _lonlat_rad_to_xyz(0.0, -1.5)
    ]
    assert _point_within_gca_cartesian(pt_same_lon_in, gcr_same_lon_cart)

    pt_same_lon_out = _lonlat_rad_to_xyz(0.0, 1.500000000000001)
    res = _point_within_gca_cartesian(pt_same_lon_out, gcr_same_lon_cart)
    assert not res

    pt_same_lon_out_2 = _lonlat_rad_to_xyz(0.1, 1.0)
    res = _point_within_gca_cartesian(pt_same_lon_out_2, gcr_same_lon_cart)
    assert not res

    # And if we increase the decimal place by one, it should be true again
    pt_same_lon_out_add_one_place = _lonlat_rad_to_xyz(0.0, 1.5000000000000001)
    res = _point_within_gca_cartesian(pt_same_lon_out_add_one_place, gcr_same_lon_cart)
    assert res

    # Normal case
    gcr_cart_2 = np.array([[0.267, 0.963, -0.007], [-0.073, -0.036, -0.997]])
    pt_cart_within = np.array([0.25616109352676675, 0.9246590335292105, -0.010021496695000144])
    assert _point_within_gca_cartesian(pt_cart_within, gcr_cart_2, True)

def test_pt_within_gcr_antimeridian():
    gcr_cart = np.array([[0.351, -0.724, 0.593], [0.617, 0.672, 0.410]])
    pt_cart = np.array([0.9438777657502077, 0.1193199333436068, 0.922714737029319])
    assert _point_within_gca_cartesian(pt_cart, gcr_cart, is_directed=True)

    gcr_cart_flip = np.array([[0.617, 0.672, 0.410], [0.351, -0.724, 0.593]])
    with pytest.raises(ValueError):
        _point_within_gca_cartesian(pt_cart, gcr_cart_flip, is_directed=True)

    assert _point_within_gca_cartesian(pt_cart, gcr_cart_flip, is_directed=False)

    # 2nd anti-meridian case
    gcr_cart_1 = np.array([[-0.491, -0.706, 0.510], [-0.755, 0.655, -0.007]])
    pt_cart_within = np.array([0.6136726305712109, 0.28442243941920053, -0.365605190899831])
    assert not _point_within_gca_cartesian(pt_cart_within, gcr_cart_1, is_directed=True)
    assert not _point_within_gca_cartesian(pt_cart_within, gcr_cart_1, is_directed=False)

    v1_rad = [0.1, 0.0]
    v2_rad = [2 * np.pi - 0.1, 0.0]
    v1_cart = _lonlat_rad_to_xyz(v1_rad[0], v1_rad[1])
    v2_cart = _lonlat_rad_to_xyz(v2_rad[0], v1_rad[1])
    gcr_cart = np.array([v1_cart, v2_cart])
    pt_cart = _lonlat_rad_to_xyz(0.01, 0.0)
    with pytest.raises(ValueError):
        _point_within_gca_cartesian(pt_cart, gcr_cart, is_directed=True)
    gcr_cart_flipped = np.array([v2_cart, v1_cart])
    assert _point_within_gca_cartesian(pt_cart, gcr_cart_flipped, is_directed=True)

def test_pt_within_gcr_cross_pole():
    gcr_cart = np.array([[0.351, 0.0, 0.3], [-0.351, 0.0, 0.3]])
    pt_cart = np.array([0.10, 0.0, 0.8])

    # Normalize the point and the GCA
    pt_cart = pt_cart / np.linalg.norm(pt_cart)
    gcr_cart = np.array([x / np.linalg.norm(x) for x in gcr_cart])
    assert _point_within_gca_cartesian(pt_cart, gcr_cart, is_directed=False)

    gcr_cart = np.array([[0.351, 0.0, 0.3], [-0.351, 0.0, -0.6]])
    pt_cart = np.array([0.10, 0.0, 0.8])

    # When the point is not within the GCA
    pt_cart = pt_cart / np.linalg.norm(pt_cart)
    gcr_cart = np.array([x / np.linalg.norm(x) for x in gcr_cart])
    assert not _point_within_gca_cartesian(pt_cart, gcr_cart, is_directed=False)
    with pytest.raises(ValueError):
        _point_within_gca_cartesian(pt_cart, gcr_cart, is_directed=True)
