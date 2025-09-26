import numpy as np
import pytest

import uxarray as ux
from uxarray.grid.geometry import stereographic_projection, inverse_stereographic_projection


def test_stereographic_projection_stereographic_projection():
    lon = np.array(0)
    lat = np.array(0)

    central_lon = np.array(0)
    central_lat = np.array(0)

    x, y = stereographic_projection(lon, lat, central_lon, central_lat)

    new_lon, new_lat = inverse_stereographic_projection(x, y, central_lon, central_lat)

    assert np.array_equal(lon, new_lon)
    assert np.array_equal(lat, new_lat)
    assert np.array_equal(x, y) and x == 0
