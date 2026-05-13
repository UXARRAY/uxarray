import numpy as np
from numba import njit

from uxarray.grid.area import calculate_face_area
from uxarray.grid.geometry import _unique_points
from uxarray.grid.integrate import _zonal_face_weights, _zonal_face_weights_robust
from uxarray.grid.intersections import (
    gca_const_lat_intersection,
    get_number_of_intersections,
)
from uxarray.grid.utils import (
    _get_cartesian_face_edge_nodes_array,
    _small_angle_of_2_vectors,
)


def _compute_non_conservative_zonal_mean(uxda, latitudes, use_robust_weights=False):
    """Computes the non-conservative zonal mean across one or more latitudes."""
    import dask.array as da

    uxgrid = uxda.uxgrid
    n_nodes_per_face = uxgrid.n_nodes_per_face.values

    face_axis = uxda.get_axis_num("n_face")

    shape = list(uxda.shape)
    shape[face_axis] = len(latitudes)

    if np.issubdtype(uxda.dtype, np.integer) or np.issubdtype(uxda.dtype, np.bool_):
        # Promote integers/bools so we can represent NaNs
        result_dtype = np.float64
    else:
        # Preserve existing float/complex dtype
        result_dtype = uxda.dtype

    if isinstance(uxda.data, da.Array):
        # Pre-fill with NaNs so empty slices stay missing without extra work
        result = da.full(shape, np.nan, dtype=result_dtype)
    else:
        # Create a NumPy array for storing results
        result = np.full(shape, np.nan, dtype=result_dtype)

    faces_edge_nodes_xyz = _get_cartesian_face_edge_nodes_array(
        uxgrid.face_node_connectivity.values,
        uxgrid.n_face,
        uxgrid.n_max_face_nodes,
        uxgrid.node_x.values,
        uxgrid.node_y.values,
        uxgrid.node_z.values,
    )

    bounds = uxgrid.bounds.values

    for i, lat in enumerate(latitudes):
        face_indices = uxda.uxgrid.get_faces_at_constant_latitude(lat)

        idx = [slice(None)] * result.ndim
        idx[face_axis] = i

        if face_indices.size == 0:
            # No intersecting faces for this latitude
            continue

        z = np.sin(np.deg2rad(lat))

        fe = faces_edge_nodes_xyz[face_indices]
        nn = n_nodes_per_face[face_indices]
        b = bounds[face_indices]

        if use_robust_weights:
            w = _zonal_face_weights_robust(fe, z, b)["weight"].to_numpy()
        else:
            w = _zonal_face_weights(fe, b, nn, z)

        total = w.sum()

        if total == 0.0 or not np.isfinite(total):
            # If weights collapse to zero, keep the pre-filled NaNs
            continue

        data_slice = uxda.isel(n_face=face_indices, ignore_grid=True).data
        w_shape = [1] * data_slice.ndim
        w_shape[face_axis] = w.size
        w_reshaped = w.reshape(w_shape)
        weighted = (data_slice * w_reshaped).sum(axis=face_axis) / total

        result[tuple(idx)] = weighted

    return result


@njit(cache=True)
def _sort_points_by_angle(points):
    """Sort points by longitude angle for proper polygon formation.

    Parameters
    ----------
    points : ndarray
        Array of 3D points on unit sphere, shape (n_points, 3)

    Returns
    -------
    ndarray
        Points sorted by longitude angle
    """
    n_points = points.shape[0]
    if n_points <= 1:
        return points.copy()

    # Calculate angles (longitude)
    angles = np.empty(n_points, dtype=np.float64)
    x_axis = np.array([1.0, 0.0, 0.0])
    for i in range(n_points):
        # Project point to xy plane for longitude calculation
        point_xy = np.array([points[i, 0], points[i, 1], 0.0])
        point_xy_norm = np.linalg.norm(point_xy)

        if point_xy_norm < 1e-15:
            angles[i] = 0.0  # Point at pole
        else:
            point_xy_unit = point_xy / point_xy_norm
            angle = _small_angle_of_2_vectors(x_axis, point_xy_unit)
            # Determine sign based on y coordinate
            if points[i, 1] < 0:
                angle = -angle
            angles[i] = angle

    # Simple insertion sort (numba-friendly for small arrays)
    sorted_points = points.copy()
    for i in range(1, n_points):
        key_angle = angles[i]
        key_point = sorted_points[i].copy()
        j = i - 1

        while j >= 0 and angles[j] > key_angle:
            angles[j + 1] = angles[j]
            sorted_points[j + 1] = sorted_points[j]
            j -= 1

        angles[j + 1] = key_angle
        sorted_points[j + 1] = key_point

    return sorted_points


@njit(cache=True)
def _compute_band_overlap_area(
    face_edges_xyz, z_min, z_max, quadrature_rule="gaussian", order=4
):
    """Compute overlap area between a face and latitude band using area.py functions.

    This function finds the intersection polygon between a face and latitude band,
    then uses calculate_face_area from area.py with latitude_adjusted_area=True
    for accurate area computation when edges lie on constant latitude lines.

    Parameters
    ----------
    face_edges_xyz : ndarray
        Cartesian coordinates of face edge nodes, shape (n_edges, 2, 3)
    z_min, z_max : float
        Z-coordinate bounds of the latitude band (z = sin(latitude))
    quadrature_rule : str, optional
        Quadrature rule to use ("gaussian" or "triangular", default: "gaussian")
    order : int, optional
        Quadrature order (default: 4, same as area.py)

    Returns
    -------
    float
        Overlap area between face and latitude band
    """
    # Pre-allocate for maximum possible intersection points
    # Worst case: 2 intersections per edge * 2 boundaries + all vertices
    max_points = face_edges_xyz.shape[0] * 4 + face_edges_xyz.shape[0]
    polygon_points = np.empty((max_points, 3), dtype=np.float64)
    point_count = 0

    # Find intersections with z_min and z_max boundaries
    z_boundaries = np.array([z_min, z_max])
    for z_boundary in z_boundaries:
        for e in range(face_edges_xyz.shape[0]):
            edge = face_edges_xyz[e]
            inter = gca_const_lat_intersection(edge, z_boundary)
            nint = get_number_of_intersections(inter)

            for i in range(nint):
                if point_count < max_points:
                    polygon_points[point_count] = inter[i]
                    point_count += 1

    # Add face vertices that lie within the band
    for e in range(face_edges_xyz.shape[0]):
        vertex = face_edges_xyz[e, 0]  # First point of edge
        if z_min <= vertex[2] <= z_max:
            if point_count < max_points:
                polygon_points[point_count] = vertex
                point_count += 1

    if point_count < 3:
        return 0.0

    # Remove duplicate points
    unique_points = _unique_points(polygon_points[:point_count])

    if unique_points.shape[0] < 3:
        return 0.0

    # Sort points to form a proper polygon
    sorted_points = _sort_points_by_angle(unique_points)

    # Use area.py calculate_face_area with latitude adjustment
    x = sorted_points[:, 0]
    y = sorted_points[:, 1]
    z = sorted_points[:, 2]

    area, _ = calculate_face_area(
        x,
        y,
        z,
        quadrature_rule=quadrature_rule,
        order=order,
        latitude_adjusted_area=True,  # Key improvement: use latitude adjustment
    )

    return area


def _compute_face_band_weights(uxgrid, bands):
    """Compute overlap area between every face and every latitude band.

    Shared geometry kernel used by both zonal_mean and zonal_anomaly so the
    expensive intersection calculations are never duplicated.

    Parameters
    ----------
    uxgrid : Grid
    bands : array-like
        Latitude band edges in degrees, shape (n_bands + 1,)

    Returns
    -------
    W : ndarray, shape (n_face, n_bands)
        W[f, b] is the overlap area between face f and band b.
        Fully-contained faces carry their full face area; partially-overlapping
        faces carry the exact intersection area.
    """
    faces_edge_nodes_xyz = _get_cartesian_face_edge_nodes_array(
        uxgrid.face_node_connectivity.values,
        uxgrid.n_face,
        uxgrid.n_max_face_nodes,
        uxgrid.node_x.values,
        uxgrid.node_y.values,
        uxgrid.node_z.values,
    )
    n_nodes_per_face = uxgrid.n_nodes_per_face.values
    face_bounds_lat = uxgrid.face_bounds_lat.values
    face_areas = uxgrid.face_areas.values

    bands = np.asarray(bands, dtype=float)
    nb = bands.size - 1
    W = np.zeros((uxgrid.n_face, nb), dtype=float)

    for bi in range(nb):
        lat0 = float(np.clip(bands[bi], -90.0, 90.0))
        lat1 = float(np.clip(bands[bi + 1], -90.0, 90.0))
        if lat0 > lat1:
            lat0, lat1 = lat1, lat0

        z0 = np.sin(np.deg2rad(lat0))
        z1 = np.sin(np.deg2rad(lat1))
        zmin, zmax = (z0, z1) if z0 <= z1 else (z1, z0)

        fully_contained = uxgrid.get_faces_between_latitudes((lat0, lat1))
        mask = ~((face_bounds_lat[:, 1] < lat0) | (face_bounds_lat[:, 0] > lat1))
        all_overlapping = np.nonzero(mask)[0]

        if all_overlapping.size == 0:
            continue

        is_fully_contained = np.isin(all_overlapping, fully_contained)

        fc = all_overlapping[is_fully_contained]
        W[fc, bi] = face_areas[fc]

        for f in all_overlapping[~is_fully_contained]:
            nedge = n_nodes_per_face[f]
            W[f, bi] = _compute_band_overlap_area(
                faces_edge_nodes_xyz[f, :nedge], zmin, zmax
            )

    return W


def _compute_conservative_zonal_mean_bands(uxda, bands):
    """Compute conservative zonal mean over latitude bands.

    Parameters
    ----------
    uxda : UxDataArray
    bands : array-like
        Latitude band edges in degrees

    Returns
    -------
    result : array
        Zonal means for each band, with n_face axis replaced by n_bands
    """
    import dask.array as da

    bands = np.asarray(bands, dtype=float)
    if bands.ndim != 1 or bands.size < 2:
        raise ValueError("bands must be 1D with at least two edges")

    W = _compute_face_band_weights(uxda.uxgrid, bands)  # (n_face, n_bands)
    nb = W.shape[1]
    face_axis = uxda.get_axis_num("n_face")

    shape = list(uxda.shape)
    shape[face_axis] = nb
    if isinstance(uxda.data, da.Array):
        result = da.full(shape, np.nan, dtype=float)
    else:
        result = np.full(shape, np.nan, dtype=float)

    for bi in range(nb):
        overlapping = np.nonzero(W[:, bi] > 0)[0]
        if overlapping.size == 0:
            continue

        w = W[overlapping, bi]
        total = w.sum()
        if total == 0.0:
            continue

        data_slice = uxda.isel(n_face=overlapping, ignore_grid=True).data
        w_shape = [1] * data_slice.ndim
        w_shape[face_axis] = w.size
        weighted = (data_slice * w.reshape(w_shape)).sum(axis=face_axis) / total

        idx = [slice(None)] * result.ndim
        idx[face_axis] = bi
        result[tuple(idx)] = weighted

    return result


def _compute_zonal_anomaly(uxda, bands, conservative=False):
    """Compute zonal anomaly: each face value minus the mean of its latitude band.

    Parameters
    ----------
    uxda : UxDataArray
    bands : array-like
        Latitude band edges in degrees
    conservative : bool
        If True, uses area-weighted band means and blends across bands for
        faces that straddle a boundary, reusing the same weight matrix as
        zonal_mean so geometry is computed only once.
        If False, assigns each face to a band by centroid latitude.

    Returns
    -------
    ndarray
        Same shape as uxda, with the per-face band mean subtracted.
    """
    bands = np.asarray(bands, dtype=float)
    face_axis = uxda.get_axis_num("n_face")
    n_face = uxda.uxgrid.n_face
    nb = bands.size - 1

    if conservative:
        # Single geometry pass shared with zonal_mean
        W = _compute_face_band_weights(uxda.uxgrid, bands)  # (n_face, n_bands)

        # Band means
        band_means = np.full(nb, np.nan)
        for bi in range(nb):
            overlapping = np.nonzero(W[:, bi] > 0)[0]
            if overlapping.size == 0:
                continue
            w = W[overlapping, bi]
            total = w.sum()
            if total > 0:
                vals = uxda.isel(n_face=overlapping, ignore_grid=True).values
                band_means[bi] = (w * vals).sum() / total

        # Map band means back to faces; straddling faces get area-weighted blend
        face_totals = W.sum(axis=1)
        valid = face_totals > 0
        face_means = np.where(
            valid,
            np.where(
                valid,
                (
                    W * np.where(np.isnan(band_means), 0.0, band_means)[np.newaxis, :]
                ).sum(axis=1)
                / np.where(valid, face_totals, 1.0),
                np.nan,
            ),
            np.nan,
        )
    else:
        # Centroid-based: fast, no intersection geometry needed
        face_lats = uxda.uxgrid.face_lat.values
        band_indices = np.clip(np.digitize(face_lats, bands) - 1, 0, nb - 1)

        band_means = np.full(nb, np.nan)
        for bi in range(nb):
            mask = band_indices == bi
            if mask.any():
                band_means[bi] = float(
                    uxda.isel(
                        n_face=np.nonzero(mask)[0], ignore_grid=True
                    ).values.mean()
                )

        face_means = band_means[band_indices]

    # Broadcast face_means to match uxda shape (face axis may not be last)
    shape = [1] * uxda.ndim
    shape[face_axis] = n_face
    return uxda.values - face_means.reshape(shape)
