import numpy as np
from numba import njit, prange
from spatialpandas.spatialindex import HilbertRtree


@njit(cache=True)
def _face_aabb_xyz_kernel(
    lat0: float, lat1: float, lon0: float, lon1: float, eps: float = 1e-12
):
    two_pi = 2.0 * np.pi
    if lon1 < lon0:
        lon1 += two_pi

    theta_samples = np.empty(6, dtype=np.float64)
    ts_n = 0
    theta_samples[ts_n] = lon0
    ts_n += 1
    theta_samples[ts_n] = lon1
    ts_n += 1
    for t0 in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
        t = t0
        if t < lon0:
            t += two_pi
        if (t >= lon0) and (t <= lon1):
            theta_samples[ts_n] = t
            ts_n += 1

    phi_samples = np.empty(3, dtype=np.float64)
    ps_n = 0
    phi_samples[ps_n] = lat0
    ps_n += 1
    phi_samples[ps_n] = lat1
    ps_n += 1
    if (lat0 <= 0.0) and (0.0 <= lat1):
        phi_samples[ps_n] = 0.0
        ps_n += 1

    xmin = 1e20
    ymin = 1e20
    zmin = 1e20
    xmax = -1e20
    ymax = -1e20
    zmax = -1e20

    for ip in range(ps_n):
        phi = phi_samples[ip]
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        for it in range(ts_n):
            theta = theta_samples[it]
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
def face_aabb_xyz(
    lat_bounds: np.ndarray, lon_bounds: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    n = lat_bounds.shape[0]
    boxes = np.empty((n, 6), dtype=np.float64)
    for i in prange(n):
        lat0 = lat_bounds[i, 0]
        lat1 = lat_bounds[i, 1]
        lon0 = lon_bounds[i, 0]
        lon1 = lon_bounds[i, 1]
        boxes[i, :] = _face_aabb_xyz_kernel(lat0, lat1, lon0, lon1, eps)
    return boxes


def construct_face_rtree_from_bounds(bounds_da, p: int = 10, page_size: int = 512):
    arr = bounds_da.values
    lat_bounds = arr[:, 0, :]
    lon_bounds = arr[:, 1, :]
    boxes = face_aabb_xyz(lat_bounds, lon_bounds)

    if HilbertRtree is None:
        return None, boxes, 3

    try:
        rtree = HilbertRtree(boxes, p, page_size)
        dim = 3
    except Exception:
        xy_boxes = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4]])
        rtree = HilbertRtree(xy_boxes, p, page_size)
        dim = 2
    return rtree, boxes, dim


def aabb_overlap3(b1: np.ndarray, b2: np.ndarray) -> bool:
    return not (
        (b1[3] < b2[0])
        or (b2[3] < b1[0])
        or (b1[4] < b2[1])
        or (b2[4] < b1[1])
        or (b1[5] < b2[2])
        or (b2[5] < b1[2])
    )


def faces_aabb_overlap_from_bounds(bounds_da, i: int, j: int) -> bool:
    arr = bounds_da.values
    lat_bounds = arr[[i, j], 0, :]
    lon_bounds = arr[[i, j], 1, :]
    boxes = face_aabb_xyz(lat_bounds, lon_bounds)
    return aabb_overlap3(boxes[0], boxes[1])


def find_intersecting_face_pairs(bounds_da):
    arr = bounds_da.values
    lat_bounds = arr[:, 0, :]
    lon_bounds = arr[:, 1, :]
    boxes = face_aabb_xyz(lat_bounds, lon_bounds)

    n = boxes.shape[0]
    pairs = []

    if HilbertRtree is not None:
        try:
            rtree = HilbertRtree(boxes)
            for i in range(n):
                hits = rtree.query(boxes[i, :])
                for j in hits:
                    j = int(j)
                    if j > i and aabb_overlap3(boxes[i], boxes[j]):
                        pairs.append((i, j))
            return np.array(pairs, dtype=np.int64)
        except Exception:
            pass

    for i in range(n):
        bi = boxes[i]
        for j in range(i + 1, n):
            if aabb_overlap3(bi, boxes[j]):
                pairs.append((i, j))
    return np.array(pairs, dtype=np.int64)
