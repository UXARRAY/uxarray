import math

import numpy as np
import xarray as xr
from scipy import sparse

# Squared chord distances are at most 4, so this is ~1000 ulps: well past the
# kd-tree's rounding, yet faces this close to a ring's edge are rare
_TIE_TOLERANCE = 1e-12


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


def _ring_membership(kdtree, center_xyz, chord_radii):
    """Assigns each face to the first ring whose bounding circle holds it: r_(i-1) < d <= r_i.

    Returns the faces and their ring indices, ordered by ring and then face.
    """
    # The running max keeps the rings past a peak in chord length (radii
    # beyond 180 degrees) empty, as their circles add no faces
    chord_bounds = np.maximum.accumulate(chord_radii)
    chord_bounds = chord_bounds * chord_bounds

    # One query at the outermost circle, then bin by squared chord distance,
    # summed in the kd-tree's own order
    faces = np.array(
        kdtree.query_ball_point(center_xyz, chord_radii.max()), dtype=np.int_
    )
    diff = kdtree.data[faces] - center_xyz
    dist2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2]
    ring_ids = np.searchsorted(chord_bounds, dist2, side="left")
    # the outer query has the final say on whether a face is in the outermost ring
    outermost = np.searchsorted(chord_bounds, chord_bounds[-1], side="left")
    ring_ids = np.minimum(ring_ids, outermost)

    # The tree prunes whole nodes on distances that carry rounding of their
    # own, so within an ulp or so of a circle it can disagree with
    # dist2 <= r**2. Settle faces that close by asking the tree at that circle.
    near_circle = np.full(faces.size, -1)
    for circle in (ring_ids - 1, ring_ids):  # the circles just inside and outside
        circle = np.clip(circle, 0, outermost)
        # first circle of a plateau, where a face on that bound first enters
        circle = np.searchsorted(chord_bounds, chord_bounds[circle], side="left")
        near = np.abs(dist2 - chord_bounds[circle]) <= _TIE_TOLERANCE
        near_circle[near] = circle[near]
    # the outer query already settled the outermost circle
    tied_circles = np.unique(
        near_circle[(near_circle >= 0) & (near_circle < outermost)]
    )
    for circle in tied_circles:
        tied = near_circle == circle
        inside = np.isin(
            faces[tied], kdtree.query_ball_point(center_xyz, chord_radii[circle])
        )
        next_circle = np.searchsorted(chord_bounds, chord_bounds[circle], side="right")
        ring_ids[tied] = np.where(inside, circle, next_circle)

    order = np.lexsort((faces, ring_ids))
    return faces[order], ring_ids[order]


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
