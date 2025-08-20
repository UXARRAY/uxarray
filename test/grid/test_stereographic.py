import numpy as np
import pytest
from sklearn.metrics.pairwise import haversine_distances

from uxarray.grid.geometry import (
    stereographic_projection,
    inverse_stereographic_projection,
    haversine_distance
)


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


def test_haversine_distance_creation():
    """Tests the use of `haversine_distance`"""

    # Create two points
    point_a = [np.deg2rad(-34.8), np.deg2rad(-58.5)]
    point_b = [np.deg2rad(49.0), np.deg2rad(2.6)]

    result = haversine_distances([point_a, point_b])

    distance = haversine_distance(point_a[1], point_a[0], point_b[1], point_b[0])

    assert np.isclose(result[0][1], distance, atol=1e-6)
