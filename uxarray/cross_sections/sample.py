import numpy as np
from numba import njit, prange
from pyproj import Geod


@njit(parallel=True)
def _fill_numba(flat_orig, face_idx, n_face, n_steps):
    M = flat_orig.shape[0]
    out = np.full((M, n_steps), np.nan, np.float64)
    for i in prange(n_steps):
        f = face_idx[i]
        if 0 <= f < n_face:
            out[:, i] = flat_orig[:, f]
    return out


def sample_geodesic(
    start: tuple[float, float], end: tuple[float, float], steps: int
) -> tuple[np.ndarray, np.ndarray]:
    lon0, lat0 = start
    lon1, lat1 = end

    # validate
    for name, val, lo, hi in [
        ("start lon", lon0, -180, 180),
        ("start lat", lat0, -90, 90),
        ("end   lon", lon1, -180, 180),
        ("end   lat", lat1, -90, 90),
    ]:
        if not (lo <= val <= hi):
            raise ValueError(f"{name}={val} out of bounds [{lo}, {hi}]")

    geod = Geod(ellps="WGS84")
    # compute intermediate (lon, lat) points on the ellipsoid
    middle = geod.npts(lon0, lat0, lon1, lat1, steps - 2)

    # preallocate arrays
    lons = np.empty(steps, dtype=float)
    lats = np.empty(steps, dtype=float)

    # endpoints
    lons[0], lats[0] = lon0, lat0
    lons[-1], lats[-1] = lon1, lat1

    # fill middle points
    for i, (lon, lat) in enumerate(middle, start=1):
        lons[i] = ((lon + 180) % 360) - 180  # normalize to (–180, +180)
        lats[i] = lat  # geod.npts yields lat in (–90, +90)

    # convert to radians
    rad_lat = np.deg2rad(lats)
    rad_lon = np.deg2rad(lons)

    # Cartesian coords
    x = np.cos(rad_lat) * np.cos(rad_lon)
    y = np.cos(rad_lat) * np.sin(rad_lon)
    z = np.sin(rad_lat)
    points_xyz = np.column_stack([x, y, z])

    # preserve input order (lon, lat)
    points_lonlat = np.column_stack([lons, lats])

    return points_xyz, points_lonlat


def sample_constant_latitude(lat: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude {lat} out of bounds [-90, 90]")
    if steps < 2:
        raise ValueError(f"steps must be ≥ 2, got {steps}")

    # sample longitudes evenly from –180 to +180
    lons = np.linspace(-180.0, 180.0, steps)
    lats = np.full(steps, lat, dtype=float)

    # convert to radians
    rad_lon = np.deg2rad(lons)
    rad_lat = np.deg2rad(lats)

    # spherical to Cartesian
    x = np.cos(rad_lat) * np.cos(rad_lon)
    y = np.cos(rad_lat) * np.sin(rad_lon)
    z = np.sin(rad_lat)
    points_xyz = np.column_stack([x, y, z])

    # normalize longitudes back into (–180, +180]
    norm_lons = ((lons + 180) % 360) - 180
    points_lonlat = np.column_stack([norm_lons, lats])

    return points_xyz, points_lonlat


def sample_constant_longitude(lon: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude {lon} out of bounds [-180, 180]")
    if steps < 2:
        raise ValueError(f"steps must be ≥ 2, got {steps}")

    # sample latitudes evenly from –90 to +90
    lats = np.linspace(-90.0, 90.0, steps)
    lons = np.full(steps, lon, dtype=float)

    # convert to radians
    rad_lon = np.deg2rad(lons)
    rad_lat = np.deg2rad(lats)

    # spherical to Cartesian
    x = np.cos(rad_lat) * np.cos(rad_lon)
    y = np.cos(rad_lat) * np.sin(rad_lon)
    z = np.sin(rad_lat)
    points_xyz = np.column_stack([x, y, z])

    # lon stays constant, so no need to renormalize
    points_lonlat = np.column_stack([lons, lats])

    return points_xyz, points_lonlat
