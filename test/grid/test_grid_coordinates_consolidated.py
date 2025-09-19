import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.coordinates import _populate_node_latlon


def test_populate_coordinates_populate_cartesian_xyz_coord():
    # The following testcases are generated through the matlab cart2sph/sph2cart functions
    lon_deg = [
        45.0001052295749, 45.0001052295749, 360 - 45.0001052295749,
                                            360 - 45.0001052295749
    ]
    lat_deg = [
        35.2655522903022, -35.2655522903022, 35.2655522903022,
        -35.2655522903022
    ]
    cart_x = [
        0.577340924821405, 0.577340924821405, 0.577340924821405,
        0.577340924821405
    ]
    cart_y = [
        0.577343045516932, 0.577343045516932, -0.577343045516932,
        -0.577343045516932
    ]
    cart_z = [
        -0.577366836872017, 0.577366836872017, -0.577366836872017,
        0.577366836872017
    ]

    verts_degree = np.stack((lon_deg, lat_deg), axis=1)
    vgrid = ux.open_grid(verts_degree, latlon=True)

    for i in range(0, vgrid.n_node):
        nt.assert_almost_equal(vgrid.node_x.values[i], cart_x[i], decimal=12)
        nt.assert_almost_equal(vgrid.node_y.values[i], cart_y[i], decimal=12)
        nt.assert_almost_equal(vgrid.node_z.values[i], cart_z[i], decimal=12)


def test_populate_coordinates_populate_lonlat_coord():
    lon_deg = [
        45.0001052295749, 45.0001052295749, 360 - 45.0001052295749,
                                            360 - 45.0001052295749
    ]
    lat_deg = [
        35.2655522903022, -35.2655522903022, 35.2655522903022,
        -35.2655522903022
    ]
    cart_x = [
        0.577340924821405, 0.577340924821405, 0.577340924821405,
        0.577340924821405
    ]
    cart_y = [
        0.577343045516932, 0.577343045516932, -0.577343045516932,
        -0.577343045516932
    ]
    cart_z = [
        0.577366836872017, -0.577366836872017, 0.577366836872017,
        -0.577366836872017
    ]

    verts_cart = np.stack((cart_x, cart_y, cart_z), axis=1)
    vgrid = ux.open_grid(verts_cart, latlon=False)
    _populate_node_latlon(vgrid)
    lon_deg, lat_deg = zip(*reversed(list(zip(lon_deg, lat_deg))))
    for i in range(0, vgrid.n_node):
        nt.assert_almost_equal(vgrid._ds["node_lon"].values[i], lon_deg[i], decimal=12)
        nt.assert_almost_equal(vgrid._ds["node_lat"].values[i], lat_deg[i], decimal=12)
