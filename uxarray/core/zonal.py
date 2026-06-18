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

    Returns a sparse per-band representation so memory scales with the number
    of faces that overlap each band (typically O(n_face) total) rather than
    O(n_face * n_bands), which would OOM on large grids with fine bands.

    Parameters
    ----------
    uxgrid : Grid
    bands : array-like
        Latitude band edges in degrees, shape (n_bands + 1,). Must be
        monotonic non-decreasing.

    Returns
    -------
    per_band : list of (indices, weights) tuples, length n_bands
        For band ``bi``: ``indices`` is an int ndarray of face indices that
        overlap the band, and ``weights`` is the corresponding overlap-area
        ndarray. Fully-contained faces carry their full face area; partially-
        overlapping faces carry the exact intersection area.
    """
    bands = np.asarray(bands, dtype=float)
    if bands.ndim != 1 or bands.size < 2:
        raise ValueError("bands must be 1D with at least two edges")
    if np.any(np.diff(bands) < 0):
        raise ValueError(
            f"bands must be monotonic non-decreasing; got diff(bands)={np.diff(bands)}"
        )

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

    nb = bands.size - 1
    per_band = []

    for bi in range(nb):
        lat0 = float(np.clip(bands[bi], -90.0, 90.0))
        lat1 = float(np.clip(bands[bi + 1], -90.0, 90.0))
        if lat0 > lat1:
            lat0, lat1 = lat1, lat0

        z0 = np.sin(np.deg2rad(lat0))
        z1 = np.sin(np.deg2rad(lat1))
        zmin, zmax = (z0, z1) if z0 <= z1 else (z1, z0)

        mask = ~((face_bounds_lat[:, 1] < lat0) | (face_bounds_lat[:, 0] > lat1))
        all_overlapping = np.nonzero(mask)[0]

        if all_overlapping.size == 0:
            per_band.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=float)))
            continue

        fully_contained = uxgrid.get_faces_between_latitudes((lat0, lat1))
        is_fully_contained = np.isin(all_overlapping, fully_contained)

        weights = np.empty(all_overlapping.size, dtype=float)

        fc_mask = is_fully_contained
        fc = all_overlapping[fc_mask]
        weights[fc_mask] = face_areas[fc]

        partial = all_overlapping[~fc_mask]
        partial_pos = np.nonzero(~fc_mask)[0]
        for pos, f in zip(partial_pos, partial):
            nedge = n_nodes_per_face[f]
            weights[pos] = _compute_band_overlap_area(
                faces_edge_nodes_xyz[f, :nedge], zmin, zmax
            )

        per_band.append((all_overlapping.astype(np.int64), weights))

    return per_band


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
    per_band = _compute_face_band_weights(uxda.uxgrid, bands)
    nb = len(per_band)
    face_axis = uxda.get_axis_num("n_face")

    if np.issubdtype(uxda.dtype, np.integer) or np.issubdtype(uxda.dtype, np.bool_):
        result_dtype = np.float64
    else:
        result_dtype = uxda.dtype

    shape = list(uxda.shape)
    shape[face_axis] = nb
    if isinstance(uxda.data, da.Array):
        result = da.full(shape, np.nan, dtype=result_dtype)
    else:
        result = np.full(shape, np.nan, dtype=result_dtype)

    for bi, (overlapping, w) in enumerate(per_band):
        if overlapping.size == 0:
            continue

        total = w.sum()
        if total == 0.0 or not np.isfinite(total):
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

    Preserves the input dtype (promoting only integer/bool inputs so NaNs can
    fit), the input shape (n_face axis stays in place even if it is not the
    last axis), and dask laziness when ``uxda`` is chunked.

    Parameters
    ----------
    uxda : UxDataArray
    bands : array-like
        Latitude band edges in degrees. Must be monotonic non-decreasing.
    conservative : bool
        If True, uses area-weighted band means and blends across bands for
        faces that straddle a boundary, reusing the same sparse weight kernel
        as zonal_mean so geometry is computed only once.
        If False, assigns each face to a band by centroid latitude.

    Returns
    -------
    array-like
        Same shape and axis order as ``uxda.data``. Returns a dask array when
        ``uxda.data`` is a dask array; otherwise a numpy array.
    """
    import dask.array as da

    bands = np.asarray(bands, dtype=float)
    if bands.ndim != 1 or bands.size < 2:
        raise ValueError("Band edges must be 1D with at least two values.")
    if np.any(np.diff(bands) < 0):
        raise ValueError(
            "Band edges must be monotonic non-decreasing; got "
            f"diff(bands)={np.diff(bands)}"
        )

    face_axis = uxda.get_axis_num("n_face")
    n_face = uxda.uxgrid.n_face
    nb = bands.size - 1
    is_dask = isinstance(uxda.data, da.Array)

    if np.issubdtype(uxda.dtype, np.integer) or np.issubdtype(uxda.dtype, np.bool_):
        out_dtype = np.float64
    else:
        out_dtype = uxda.dtype

    reduced_shape = list(uxda.shape)
    reduced_shape.pop(face_axis)

    def _reshape_along_face(w_1d):
        s = [1] * uxda.ndim
        s[face_axis] = w_1d.size
        return w_1d.reshape(s)

    if conservative:
        per_band = _compute_face_band_weights(uxda.uxgrid, bands)

        # Compute per-band means along the n_face axis, preserving other dims.
        # band_means is a list of length nb; entries are arrays with shape
        # reduced_shape (or None when no overlap). They are small relative to
        # uxda, so materializing them is cheap.
        band_means = [None] * nb
        face_totals = np.zeros(n_face, dtype=float)

        for bi, (overlapping, w) in enumerate(per_band):
            if overlapping.size == 0:
                continue
            total = w.sum()
            if total == 0.0 or not np.isfinite(total):
                continue
            face_totals[overlapping] += w
            data_slice = uxda.isel(n_face=overlapping, ignore_grid=True).data
            band_mean = (data_slice * _reshape_along_face(w)).sum(
                axis=face_axis
            ) / total
            if isinstance(band_mean, da.Array):
                band_mean = band_mean.compute()
            band_means[bi] = band_mean.astype(out_dtype, copy=False)

        # face_means_num[..., f, ...] = sum_b W[f,b] * band_mean[b]
        # This is the output-shaped per-face mean field. Built eagerly because
        # the scatter pattern is awkward in dask; uxda.data itself is not
        # touched so its laziness is preserved by the final subtract.
        face_means_num = np.zeros(uxda.shape, dtype=out_dtype)
        for bi, (overlapping, w) in enumerate(per_band):
            if overlapping.size == 0 or band_means[bi] is None:
                continue
            bm_expanded = np.expand_dims(band_means[bi], face_axis)
            contrib = bm_expanded * _reshape_along_face(w)
            idx = [slice(None)] * uxda.ndim
            idx[face_axis] = overlapping
            face_means_num[tuple(idx)] += contrib

        valid = face_totals > 0
        face_means = np.full(uxda.shape, np.nan, dtype=out_dtype)
        if valid.any():
            valid_idx = np.nonzero(valid)[0]
            idx = [slice(None)] * uxda.ndim
            idx[face_axis] = valid_idx
            face_means[tuple(idx)] = face_means_num[tuple(idx)] / _reshape_along_face(
                face_totals[valid_idx]
            )

    else:
        # Centroid-based: fast, no intersection geometry needed.
        face_lats = uxda.uxgrid.face_lat.values
        band_indices = np.clip(np.digitize(face_lats, bands) - 1, 0, nb - 1)

        # Compute per-band mean reducing only over the face axis. Build a
        # stack of shape (nb, *reduced_shape); preserve dask laziness.
        per_band_means = []
        for bi in range(nb):
            sel = np.nonzero(band_indices == bi)[0]
            if sel.size == 0:
                if is_dask:
                    per_band_means.append(
                        da.full(tuple(reduced_shape), np.nan, dtype=out_dtype)
                    )
                else:
                    per_band_means.append(
                        np.full(tuple(reduced_shape), np.nan, dtype=out_dtype)
                    )
            else:
                sub = uxda.isel(n_face=sel, ignore_grid=True).data
                per_band_means.append(sub.mean(axis=face_axis))

        if is_dask:
            band_means = da.stack(per_band_means, axis=0)
            face_means_face_first = band_means[band_indices]
        else:
            band_means = np.stack(per_band_means, axis=0)
            face_means_face_first = np.take(band_means, band_indices, axis=0)
        face_means = np.moveaxis(face_means_face_first, 0, face_axis)

    return uxda.data - face_means
