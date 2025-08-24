import numpy as np
from numba import njit, prange

try:
    from rtree import index

    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False
    index = None


@njit(cache=True)
def _face_aabb_xyz_kernel(
    lat0: float, lat1: float, lon0: float, lon1: float, eps: float = 1e-12
):
    """Compute 3D Cartesian AABB for a single spherical face.

    Parameters
    ----------
    lat0, lat1 : float
        Latitude bounds in radians.
    lon0, lon1 : float
        Longitude bounds in radians.
    eps : float, optional
        Small padding to add to bounding box for numerical stability.

    Returns
    -------
    tuple
        (xmin, ymin, zmin, xmax, ymax, zmax) bounding box coordinates.
    """
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
    """Compute 3D Cartesian AABBs for multiple spherical faces.

    Parameters
    ----------
    lat_bounds : np.ndarray
        Array of shape (n, 2) with latitude bounds in radians.
    lon_bounds : np.ndarray
        Array of shape (n, 2) with longitude bounds in radians.
    eps : float, optional
        Small padding to add to bounding boxes for numerical stability.

    Returns
    -------
    np.ndarray
        Array of shape (n, 6) with bounding boxes as (xmin, ymin, zmin, xmax, ymax, zmax).
    """
    n = lat_bounds.shape[0]
    boxes = np.empty((n, 6), dtype=np.float64)
    for i in prange(n):
        lat0 = lat_bounds[i, 0]
        lat1 = lat_bounds[i, 1]
        lon0 = lon_bounds[i, 0]
        lon1 = lon_bounds[i, 1]
        boxes[i, :] = _face_aabb_xyz_kernel(lat0, lat1, lon0, lon1, eps)
    return boxes


def construct_face_rtree_from_bounds(
    bounds_da, leaf_capacity: int = 10, near_minimum_overlap_factor: int = 7
):
    """Construct an R-tree spatial index from face bounds.

    Parameters
    ----------
    bounds_da : xarray.DataArray
        Face bounds data array with shape (n_face, 2, 2).
    leaf_capacity : int, optional
        Maximum number of entries in a leaf node (default: 10).
    near_minimum_overlap_factor : int, optional
        Used during bulk loading to improve performance (default: 7, must be < leaf_capacity).

    Returns
    -------
    tuple
        (rtree, boxes, dim) where rtree is the rtree.index.Index instance,
        boxes are the computed bounding boxes, and dim is 2 or 3.
    """
    arr = bounds_da.values
    lat_bounds = arr[:, 0, :]
    lon_bounds = arr[:, 1, :]
    boxes = face_aabb_xyz(lat_bounds, lon_bounds)

    if not RTREE_AVAILABLE:
        return None, boxes, 3

    try:
        # Try 3D R-tree first
        properties = index.Property(
            dimension=3,
            leaf_capacity=leaf_capacity,
            near_minimum_overlap_factor=near_minimum_overlap_factor,
        )
        rtree = index.Index(properties=properties)

        # Bulk insert all bounding boxes
        for i, box in enumerate(boxes):
            rtree.insert(i, box)

        dim = 3
    except Exception:
        # Fallback to 2D if 3D fails
        xy_boxes = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4]])
        properties = index.Property(
            dimension=2,
            leaf_capacity=leaf_capacity,
            near_minimum_overlap_factor=near_minimum_overlap_factor,
        )
        rtree = index.Index(properties=properties)

        # Bulk insert 2D bounding boxes
        for i, box in enumerate(xy_boxes):
            rtree.insert(i, box)

        dim = 2

    return rtree, boxes, dim


def aabb_overlap3(b1: np.ndarray, b2: np.ndarray) -> bool:
    """Check if two 3D axis-aligned bounding boxes overlap.

    Parameters
    ----------
    b1, b2 : np.ndarray
        Bounding boxes as (xmin, ymin, zmin, xmax, ymax, zmax).

    Returns
    -------
    bool
        True if boxes overlap, False otherwise.
    """
    return not (
        (b1[3] < b2[0])
        or (b2[3] < b1[0])
        or (b1[4] < b2[1])
        or (b2[4] < b1[1])
        or (b1[5] < b2[2])
        or (b2[5] < b1[2])
    )


def faces_aabb_overlap_from_bounds(bounds_da, i: int, j: int) -> bool:
    """Check if two faces' bounding boxes overlap.

    Parameters
    ----------
    bounds_da : xarray.DataArray
        Face bounds data array.
    i, j : int
        Face indices to check.

    Returns
    -------
    bool
        True if face bounding boxes overlap.
    """
    arr = bounds_da.values
    lat_bounds = arr[[i, j], 0, :]
    lon_bounds = arr[[i, j], 1, :]
    boxes = face_aabb_xyz(lat_bounds, lon_bounds)
    return aabb_overlap3(boxes[0], boxes[1])


def find_intersecting_face_pairs(bounds_da):
    """Find all pairs of faces with overlapping bounding boxes.

    Parameters
    ----------
    bounds_da : xarray.DataArray
        Face bounds data array.

    Returns
    -------
    np.ndarray
        Array of shape (n_pairs, 2) with face index pairs.
    """
    arr = bounds_da.values
    lat_bounds = arr[:, 0, :]
    lon_bounds = arr[:, 1, :]
    boxes = face_aabb_xyz(lat_bounds, lon_bounds)

    n = boxes.shape[0]
    pairs = []

    if RTREE_AVAILABLE:
        try:
            # Create 3D R-tree
            properties = index.Property(dimension=3)
            rtree = index.Index(properties=properties)

            # Insert all boxes
            for i, box in enumerate(boxes):
                rtree.insert(i, box)

            # Find intersecting pairs
            for i in range(n):
                hits = list(rtree.intersection(boxes[i, :]))
                for j in hits:
                    j = int(j)
                    if j > i and aabb_overlap3(boxes[i], boxes[j]):
                        pairs.append((i, j))
            return np.array(pairs, dtype=np.int64)
        except Exception:
            pass

    # Fallback to brute force
    for i in range(n):
        bi = boxes[i]
        for j in range(i + 1, n):
            if aabb_overlap3(bi, boxes[j]):
                pairs.append((i, j))
    return np.array(pairs, dtype=np.int64)
