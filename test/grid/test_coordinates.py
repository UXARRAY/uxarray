import os
from pathlib import Path

import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parent


def test_populate_coordinates_populate_cartesian_xyz_coord():
    """Test populating Cartesian coordinates from lat/lon."""
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
    """Test populating lat/lon coordinates from Cartesian."""
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

    for i in range(0, vgrid.n_node):
        nt.assert_almost_equal(vgrid.node_lon.values[i], np.deg2rad(lon_deg[i]), decimal=12)
        nt.assert_almost_equal(vgrid.node_lat.values[i], np.deg2rad(lat_deg[i]), decimal=12)


def test_normalize_existing_coordinates_non_norm_initial():
    """Test normalizing coordinates that are not initially normalized."""
    verts = [
        [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [6.0, 6.0, 6.0]]
    ]

    grid_verts = ux.open_grid(verts, latlon=False)

    # Check that coordinates are normalized
    for i in range(grid_verts.n_node):
        node_magnitude = np.sqrt(
            grid_verts.node_x.values[i]**2 +
            grid_verts.node_y.values[i]**2 +
            grid_verts.node_z.values[i]**2
        )
        nt.assert_almost_equal(node_magnitude, 1.0, decimal=10)


def test_normalize_existing_coordinates_norm_initial():
    """Test normalizing coordinates that are already normalized."""
    verts = [
        [[0.57735027, 0.57735027, 0.57735027],
         [0.70710678, 0.70710678, 0.0],
         [0.0, 0.0, 1.0]]
    ]

    grid_verts = ux.open_grid(verts, latlon=False)

    # Check that coordinates remain normalized
    for i in range(grid_verts.n_node):
        node_magnitude = np.sqrt(
            grid_verts.node_x.values[i]**2 +
            grid_verts.node_y.values[i]**2 +
            grid_verts.node_z.values[i]**2
        )
        nt.assert_almost_equal(node_magnitude, 1.0, decimal=10)
