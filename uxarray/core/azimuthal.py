import math

import numpy as np
import xarray as xr
from scipy import sparse


def _compute_azimuthal_mean(uxda, center_xyz, chord_radii):
    """Computes area-weighted means over the rings between successive bounding circles.

    Returns the means, with ``n_face`` replaced by ``radius`` in place (lazy
    when ``uxda`` is dask-backed), and the number of faces in each ring.
    """
    kdtree = uxda.uxgrid._get_scipy_kd_tree()
    n_radius = chord_radii.size

    # Each ring holds the faces newly inside its bounding circle: r_(i-1) < d <= r_i
    ring_faces = []
    faces_processed = np.array([], dtype=np.int_)
    for r_chord in chord_radii:
        within = np.array(kdtree.query_ball_point(center_xyz, r_chord), dtype=np.int_)
        if within.size:
            within.sort()

        ring = np.setdiff1d(within, faces_processed, assume_unique=True)
        ring_faces.append(ring)

        if ring.size:
            faces_processed = within  # cumulative set for next iteration

    hit_count = np.array([ring.size for ring in ring_faces], dtype=np.int_)

    # Rings are disjoint, so all per-ring means form one sparse
    # (n_radius, n_ring_faces) operator. Areas come from the parent grid, and
    # only the ring faces are gathered, so no per-ring subgrid is ever built.
    faces = np.concatenate(ring_faces)
    ring_ids = np.repeat(np.arange(n_radius), hit_count)
    areas = uxda.uxgrid.face_areas.values[faces]
    ring_area = np.bincount(ring_ids, weights=areas, minlength=n_radius)
    weights = sparse.csr_matrix(
        (areas / ring_area[ring_ids], (ring_ids, np.arange(faces.size))),
        shape=(n_radius, faces.size),
    )

    ring_data = uxda.isel(n_face=faces, ignore_grid=True).to_xarray()
    if ring_data.chunks is not None:
        # the sparse multiply spans every ring face, so keep them in a single chunk
        ring_data = ring_data.chunk({"n_face": -1})

    means = xr.apply_ufunc(
        _apply_ring_weights,
        ring_data,
        input_core_dims=[["n_face"]],
        output_core_dims=[["radius"]],
        dask="parallelized",
        kwargs={"weights": weights, "empty": hit_count == 0},
        output_dtypes=[np.result_type(weights.dtype, ring_data.dtype)],
        dask_gufunc_kwargs={"output_sizes": {"radius": n_radius}},
    )

    dims = ["radius" if dim == "n_face" else dim for dim in uxda.dims]
    return means.transpose(*dims).data, hit_count


def _apply_ring_weights(values, weights, empty):
    """Applies the sparse ring operator along the trailing (ring face) dimension."""
    # NaN faces add nothing to the weighted sum but still count toward the ring
    # area, matching the skipna sum of the former per-ring weighted_mean()
    values = np.where(np.isnan(values), 0, values)
    # explicit leading size: -1 is ambiguous when no face falls in any ring
    flat = values.reshape(math.prod(values.shape[:-1]), values.shape[-1])
    means = (weights @ flat.T).T
    means[:, empty] = np.nan
    return means.reshape(values.shape[:-1] + (weights.shape[0],))
