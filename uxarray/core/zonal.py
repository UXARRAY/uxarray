import numpy as np
from numba import njit

from uxarray.constants import ERROR_TOLERANCE
from uxarray.errors import DimensionError
from uxarray.grid.integrate import _zonal_face_weights, _zonal_face_weights_robust
from uxarray.grid.utils import _get_cartesian_face_edge_nodes_array_subset
from uxarray.utils.numba_math import (
    _numba_add3,
    _numba_cross3,
    _numba_div3_scalar,
    _numba_dot3,
    _numba_mul3_scalar,
    _numba_norm3,
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

    # Grid arrays are read once here and passed to the per-latitude subset
    # builder. The whole-grid (n_face, n_max, 2, 3) edge array is never
    # materialized; only the faces intersecting each latitude are built, so peak
    # memory scales with the candidate count (~1% of n_face) instead of n_face.
    face_node_connectivity = uxgrid.face_node_connectivity.values
    n_max_face_edges = uxgrid.n_max_face_nodes
    node_x = uxgrid.node_x.values
    node_y = uxgrid.node_y.values
    node_z = uxgrid.node_z.values

    bounds = uxgrid.bounds.values

    for i, lat in enumerate(latitudes):
        face_indices = uxda.uxgrid.get_faces_at_constant_latitude(lat)

        idx = [slice(None)] * result.ndim
        idx[face_axis] = i

        if face_indices.size == 0:
            # No intersecting faces for this latitude
            continue

        z = np.sin(np.deg2rad(lat))

        fe = _get_cartesian_face_edge_nodes_array_subset(
            face_indices,
            face_node_connectivity,
            n_nodes_per_face,
            n_max_face_edges,
            node_x,
            node_y,
            node_z,
        )

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
def _is_pole_point(p):
    """Whether ``p`` lies (numerically) on a pole, where longitude is undefined."""
    return p[0] * p[0] + p[1] * p[1] < ERROR_TOLERANCE * ERROR_TOLERANCE


@njit(cache=True)
def _lon_delta(a, b):
    """Signed longitude change from ``a`` to ``b`` in (-pi, pi].

    Computed from the horizontal projections of the two points, so it is
    unaffected by the antimeridian.
    """
    return np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


@njit(cache=True)
def _gca_z_dlon_integral(a, b):
    """Exact value of the line integral of ``z dlon`` along the minor great-circle arc a -> b.

    With ``n = a x b`` and ``t = n x p`` the (unnormalized) tangent at ``p``, an
    antiderivative along the arc is ``-arctan(t_z / n_z)``. The difference of the
    two arctangents is evaluated with a single ``arctan2`` so meridian arcs
    (``n_z == 0``) give 0 instead of dividing by zero.
    """
    n = _numba_cross3(a, b)
    ta_z = n[0] * a[1] - n[1] * a[0]
    tb_z = n[0] * b[1] - n[1] * b[0]
    return -np.arctan2(n[2] * (tb_z - ta_z), n[2] * n[2] + ta_z * tb_z)


@njit(cache=True)
def _edge_band_integral(a, b, z_min, z_max):
    """Line integral of ``(clip(z, z_min, z_max) - z_min) dlon`` along the arc a -> b.

    The arc is split where it crosses ``z = z_min`` and ``z = z_max`` (at most
    twice each), so on every piece the integrand is either a constant or ``z``
    itself, both of which integrate exactly.

    Returns the integral and whether any piece of the arc lies strictly inside
    the band.
    """
    a = (a[0], a[1], a[2])
    b = (b[0], b[1], b[2])
    n = _numba_cross3(a, b)
    sin_arc = _numba_norm3(n)
    if sin_arc == 0.0:
        return 0.0, False
    arc = np.arctan2(sin_arc, _numba_dot3(a, b))

    # Parametrize the arc as p(t) = a cos(t) + u sin(t), t in [0, arc], where u is
    # the unit tangent at a pointing toward b. Then z(t) = r cos(t - t0).
    u = _numba_div3_scalar(_numba_cross3(n, a), sin_arc)
    r = np.hypot(a[2], u[2])
    t0 = np.arctan2(u[2], a[2])

    ts = np.empty(6, dtype=np.float64)
    ts[0] = 0.0
    nt = 1
    for c in (z_min, z_max):
        if np.abs(c) < r:
            d = np.arccos(c / r)
            for t in (t0 - d, t0 + d):
                if t > np.pi:
                    t -= 2.0 * np.pi
                elif t <= -np.pi:
                    t += 2.0 * np.pi
                if 0.0 < t < arc:
                    ts[nt] = t
                    nt += 1
    ts[nt] = arc
    nt += 1
    ts[:nt].sort()

    total = 0.0
    inside = False
    p0 = a
    for i in range(1, nt):
        if i == nt - 1:
            p1 = b
        else:
            p1 = _numba_add3(
                _numba_mul3_scalar(a, np.cos(ts[i])),
                _numba_mul3_scalar(u, np.sin(ts[i])),
            )

        # Each piece lies entirely below, inside, or above the band; classify it
        # by its midpoint. The integrand is continuous in z, so a piece grazing a
        # band edge contributes (almost) the same under either classification.
        t_mid = 0.5 * (ts[i - 1] + ts[i])
        z_mid = a[2] * np.cos(t_mid) + u[2] * np.sin(t_mid)
        if z_mid > z_min:
            dlon = _lon_delta(p0, p1)
            if z_mid >= z_max:
                total += (z_max - z_min) * dlon
            else:
                total += _gca_z_dlon_integral(p0, p1) - z_min * dlon
                inside = True
        p0 = p1

    return total, inside


@njit(cache=True)
def _compute_band_overlap_area(face_edges_xyz, z_min, z_max):
    """Compute the exact overlap area between a face and a latitude band.

    The Lambert cylindrical equal-area map ``(lon, z)``, with ``z = sin(lat)``,
    preserves area and sends the band to the strip ``z_min <= z <= z_max``. By
    Green's theorem the area of the part of a face inside that strip is

        -(closed line integral of (clip(z, z_min, z_max) - z_min) dlon)

    taken around the face boundary, which is evaluated exactly on each
    great-circle edge (see :func:`_edge_band_integral`). No intersection
    polygon is constructed, so there is no vertex ordering to get wrong, and
    longitude only enters through local differences, so faces spanning the
    antimeridian need no special handling.

    Pole handling: at a vertex lying on a pole the boundary jumps between
    meridians, which contributes the integrand at the pole times that jump. A
    face enclosing a pole winds once around it in longitude; the strip region
    above (north) or below (south) its boundary then contributes
    ``2*pi*(z_max - z_min)`` or ``0`` respectively.

    Parameters
    ----------
    face_edges_xyz : ndarray
        Cartesian coordinates of the face's edge nodes on the unit sphere, shape
        (n_edges, 2, 3), with the edges in boundary order.
    z_min, z_max : float
        Z-coordinate bounds of the latitude band (z = sin(latitude)),
        ``z_min <= z_max``.

    Returns
    -------
    float
        Overlap area between the face and the latitude band.
    """
    n_edges = face_edges_xyz.shape[0]
    line_integral = 0.0
    total_dlon = 0.0
    z_sum = 0.0
    boundary_in_band = False

    for e in range(n_edges):
        a = face_edges_xyz[e, 0]
        b = face_edges_xyz[e, 1]
        z_sum += a[2]

        if _is_pole_point(a) or _is_pole_point(b):
            # Meridian arc to or from a pole: longitude is constant along it, so
            # it adds nothing to the integral, and z varies monotonically.
            z_lo = min(a[2], b[2])
            z_hi = max(a[2], b[2])
            boundary_in_band |= max(z_lo, z_min) < min(z_hi, z_max)
            if _is_pole_point(a):
                # Turning at the pole from the incoming to the outgoing meridian
                # sweeps longitude at constant z = a_z.
                jump = _lon_delta(face_edges_xyz[e - 1, 0], b)
                h_pole = min(max(a[2], z_min), z_max) - z_min
                line_integral += h_pole * jump
                total_dlon += jump
            continue

        total_dlon += _lon_delta(a, b)
        integral, inside = _edge_band_integral(a, b, z_min, z_max)
        line_integral += integral
        boundary_in_band |= inside

    area = -line_integral
    winding = int(np.round(total_dlon / (2.0 * np.pi)))
    if winding == 0:
        # Taking the magnitude makes the result independent of whether the
        # face is ordered counterclockwise or clockwise.
        area = np.abs(area)
    else:
        # The face encloses a pole. The winding direction combined with which
        # pole is enclosed gives the orientation (counterclockwise winds
        # eastward around the north pole and westward around the south pole).
        north = z_sum > 0.0
        orientation = np.sign(winding) if north else -np.sign(winding)
        area *= orientation
        if north:
            area += 2.0 * np.pi * (z_max - z_min)

    if not boundary_in_band:
        # The boundary never enters the band, so the band either misses the
        # face or lies entirely inside it (only possible for a face enclosing a
        # pole). Return that exactly instead of the near-cancelling sum, so a
        # face that merely touches the band gets a weight of exactly zero.
        full_band = 2.0 * np.pi * (z_max - z_min)
        if winding != 0 and area > 0.5 * full_band:
            return full_band
        return 0.0
    return max(area, 0.0)


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
        raise DimensionError(
            "bands must be 1D with at least two values; "
            f"got bands with ndim={bands.ndim}, size={bands.size}."
        )
    if np.any(np.diff(bands) < 0):
        raise ValueError(
            f"bands must be monotonic non-decreasing; got diff(bands)={np.diff(bands)}"
        )

    # Read grid arrays once; the per-band partial-face edge subsets are built
    # on demand below rather than materializing the whole-grid edge array.
    face_node_connectivity = uxgrid.face_node_connectivity.values
    n_max_face_edges = uxgrid.n_max_face_nodes
    node_x = uxgrid.node_x.values
    node_y = uxgrid.node_y.values
    node_z = uxgrid.node_z.values
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
        if partial.size:
            # Build edges for only the partially-overlapping faces of this band.
            fe_partial = _get_cartesian_face_edge_nodes_array_subset(
                partial,
                face_node_connectivity,
                n_nodes_per_face,
                n_max_face_edges,
                node_x,
                node_y,
                node_z,
            )
            for k, (pos, f) in enumerate(zip(partial_pos, partial)):
                nedge = n_nodes_per_face[f]
                weights[pos] = _compute_band_overlap_area(
                    fe_partial[k, :nedge], zmin, zmax
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
        raise DimensionError("Band edges must be 1D with at least two values.")
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
