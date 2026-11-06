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

    # Read the span of ring faces contiguously, then pick them out of it: a
    # lazily backed (e.g. netCDF) variable reads fancy indices face by face
    start, stop = (faces[0], faces[-1] + 1) if faces.size else (0, 0)
    span = uxda.to_xarray().isel(n_face=slice(start, stop))
    if span.chunks is None:
        ring_data = span.load().isel(n_face=faces - start)
    else:
        # the sparse multiply spans every ring face, so keep them in a single chunk
        ring_data = span.isel(n_face=faces - start).chunk({"n_face": -1})

    means = xr.apply_ufunc(
        _apply_ring_weights,
        ring_data,
        input_core_dims=[["n_face"]],
        output_core_dims=[["radius"]],
        dask="parallelized",
        kwargs={"weights": weights},
        output_dtypes=[np.result_type(weights.dtype, ring_data.dtype)],
        # n_face is already one chunk; dask's rechunk divides by zero when it's empty
        dask_gufunc_kwargs={
            "output_sizes": {"radius": n_radius},
            "allow_rechunk": False,
        },
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


def _apply_ring_weights(values, weights):
    """Applies the sparse ring operator along the trailing (ring face) dimension."""
    # NaN faces add nothing to the weighted sum but still count toward the ring
    # area, matching the skipna sum of the former per-ring weighted_mean()
    values = np.where(np.isnan(values), 0, values)
    # explicit leading size: -1 is ambiguous when no face falls in any ring
    flat = values.reshape(math.prod(values.shape[:-1]), values.shape[-1])
    means = (weights @ flat.T).T
    means[:, np.diff(weights.indptr) == 0] = np.nan  # empty rings
    return means.reshape(values.shape[:-1] + (weights.shape[0],))
