import numpy as np
from numba import njit, prange
from spatialpandas.spatialindex import HilbertRtree


def _construct_rtree(bounds, p=10, page_size=512):
    lat_bounds = bounds.sel(lon_lat=0, min_max=[0, 1]).values
    lon_bounds = bounds.sel(lon_lat=1, min_max=[0, 1]).values

    boxes = face_aabb_xyz(lat_bounds, lon_bounds)

    return HilbertRtree(boxes, p, page_size)


@njit(cache=True)
def face_aabb_xyz_kernel(lat0, lat1, lon0, lon1, eps=1e-12):
    two_pi = 2 * np.pi

    # if it crosses the antimeridian, unwrap lon1
    if lon1 < lon0:
        lon1 += two_pi

    # build list of theta samples: ends and any cardinal meridians inside
    samples = [lon0, lon1]
    for theta_c in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        t = theta_c
        if t < lon0:
            t += two_pi
        if lon0 <= t <= lon1:
            samples.append(t)

    # build list of phi samples: bounds and equator if spanned
    phis = [lat0, lat1]
    if lat0 <= 0.0 <= lat1:
        phis.append(0.0)

    # initialize extremes TODO
    xmin = ymin = zmin = 1e20
    xmax = ymax = zmax = -1e20

    # sample all (phi, theta)
    for phi in phis:
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        for theta in samples:
            x = cos_phi * np.cos(theta)
            y = cos_phi * np.sin(theta)
            z = sin_phi

            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
            if z < zmin:
                zmin = z
            if z > zmax:
                zmax = z

    return (xmin - eps, ymin - eps, zmin - eps, xmax + eps, ymax + eps, zmax + eps)


@njit(cache=True, parallel=True)
def face_aabb_xyz(lat_bounds, lon_bounds, eps=1e-12):
    n = lat_bounds.shape[0]
    boxes = np.empty((n, 6), dtype=np.float64)
    for i in prange(n):
        boxes[i, :] = face_aabb_xyz_kernel(
            lat_bounds[i, 0], lat_bounds[i, 1], lon_bounds[i, 0], lon_bounds[i, 1], eps
        )
    return boxes
