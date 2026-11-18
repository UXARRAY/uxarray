import math

import numpy as np
import xarray as xr
from scipy import sparse


def _compute_azimuthal_mean(uxda, center_xyz, chord_radii):
    """Computes area-weighted means over the rings between successive bounding circles.

    Returns the means, with ``n_face`` replaced by ``radius`` in place (lazy
    when ``uxda`` is dask-backed), and the number of faces in each ring.
    """
    n_radius = chord_radii.size
    faces, ring_ids = _ring_membership(
        uxda.uxgrid._get_scipy_kd_tree(), center_xyz, chord_radii
    )
    hit_count = np.bincount(ring_ids, minlength=n_radius)

    # Rings are disjoint, so all per-ring means form one sparse
    # (n_radius, n_ring_faces) operator. Areas come from the parent grid, and
    # only the ring faces are gathered, so no per-ring subgrid is ever built.
    areas = uxda.uxgrid.face_areas.values[faces]
    ring_area = np.bincount(ring_ids, weights=areas, minlength=n_radius)
    weights = sparse.csr_matrix(
        (areas / ring_area[ring_ids], (ring_ids, np.arange(faces.size))),
        shape=(n_radius, faces.size),
    )

    # Only the data variable is used: the caller rebuilds the coords, and
    # loading them here would read any lazy ones once per contiguous read
    variable = uxda.variable
    if variable.chunks is not None:
        variable = variable.transpose(..., "n_face")
        sums = _dask_ring_sums(variable.data, faces, weights)
    else:
        if variable._in_memory:
            # one fancy index is cheaper than contiguous reads from memory
            ring_data = variable.isel(n_face=faces)
        else:
            ring_data = _read_faces(variable, faces)
        variable = ring_data.transpose(..., "n_face")
        sums = _ring_sums(variable.values, weights)

    # empty rings are NaN; np.where defers to dask for lazy sums
    means = xr.Variable(
        variable.dims[:-1] + ("radius",), np.where(hit_count == 0, np.nan, sums)
    )
    dims = ["radius" if dim == "n_face" else dim for dim in uxda.dims]
    return means.transpose(*dims), hit_count


def _ring_membership(kdtree, center_xyz, chord_radii):
    """Assigns each face to the first ring whose bounding circle holds it: r_(i-1) < d <= r_i.

    Returns the faces, in ascending order, and their ring indices.
    """
    # The running max keeps the rings past a peak in chord length (radii
    # beyond 180 degrees) empty, as their circles add no faces
    bounds = np.maximum.accumulate(chord_radii) ** 2

    # One padded query finds every candidate, so the tree's rounding can't drop
    # faces on the outer circle; the squared distances alone then bin them
    faces = np.array(
        kdtree.query_ball_point(
            center_xyz, chord_radii.max() * (1 + 1e-9), return_sorted=True
        ),
        dtype=np.int_,
    )
    dist2 = ((kdtree.data[faces] - center_xyz) ** 2).sum(axis=1)
    inside = dist2 <= bounds[-1]
    return faces[inside], np.searchsorted(bounds, dist2[inside])


def _read_faces(variable, faces):
    """Loads the given faces (ascending, unique) of a non-dask variable.

    A lazily backed (e.g. netCDF) variable reads indices face by face,
    while one slice spanning them all can read far more faces than are
    selected. So the faces are read as contiguous slices instead, bridging the
    smallest gaps between them for as long as at most twice the selected faces
    are read, which keeps the number of reads low.
    """
    if faces.size == 0:
        return variable.isel(n_face=slice(0, 0)).load()

    gaps = np.diff(faces) - 1
    by_size = np.argsort(gaps)
    n_bridged = np.searchsorted(np.cumsum(gaps[by_size]), faces.size, side="right")
    # one read per run of faces between the gaps left unbridged
    runs = np.split(faces, np.sort(by_size[n_bridged:]) + 1)
    pieces = [
        variable.isel(n_face=slice(run[0], run[-1] + 1))
        .load()
        .isel(n_face=run - run[0])
        for run in runs
    ]
    return xr.Variable.concat(pieces, "n_face")


def _dask_ring_sums(array, faces, weights):
    """Applies the sparse ring operator to the ring faces of each input
    ``n_face`` chunk of a dask array (``n_face`` last) and adds up the partial sums.
    """
    import dask.array as da

    n_radius = weights.shape[0]
    out_chunks = array.chunks[:-1] + ((n_radius,),)
    meta = np.array((), dtype=np.result_type(weights.dtype, array.dtype))
    # faces are ascending, so each chunk's ring faces are one slice faces[lo:hi],
    # and indexing them stays within that chunk
    edges = np.searchsorted(faces, np.cumsum((0,) + array.chunks[-1]))
    parts = [
        array[..., faces[lo:hi]].map_blocks(
            _ring_sums, weights=weights[:, lo:hi], chunks=out_chunks, meta=meta
        )
        for lo, hi in zip(edges[:-1], edges[1:])
        if lo < hi
    ]

    if not parts:
        shape = array.shape[:-1] + (n_radius,)
        return da.zeros(shape, chunks=out_chunks, dtype=meta.dtype)
    # stacking makes the sum a tree reduction, where a chain of + would fuse into
    # one task that needs every input chunk at once
    return da.stack(parts).sum(axis=0)


def _ring_sums(values, weights):
    """Applies the sparse ring operator along the trailing (ring face) dimension."""
    # NaN faces add nothing to the weighted sum but still count toward the ring
    # area, matching the skipna sum of the former per-ring weighted_mean()
    values = np.where(np.isnan(values), 0, values)
    # explicit leading size: -1 is ambiguous when no face falls in any ring
    flat = values.reshape(math.prod(values.shape[:-1]), values.shape[-1])
    return (flat @ weights.T).reshape(values.shape[:-1] + (weights.shape[0],))
