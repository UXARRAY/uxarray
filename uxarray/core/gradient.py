import warnings

import numpy as np
from numba import njit, prange

from uxarray.constants import INT_FILL_VALUE
from uxarray.conventions.attrs import _drop_value_describing_attrs
from uxarray.core.utils import _apply_1dfunc_with_grid_core_dim
from uxarray.errors import DataCenteringError


def _calculate_edge_face_difference(d_var, edge_faces, n_edge):
    """Helper function for computing the aboslute difference between the data
    values on each face that saddle each edge.

    Edges with only a single neighbor will default to a value of zero.
    """
    dims = list(d_var.shape[:-1])
    dims.append(n_edge)

    edge_face_diff = np.zeros(dims)

    saddle_mask = edge_faces[:, 1] != INT_FILL_VALUE

    edge_face_diff[..., saddle_mask] = (
        d_var[..., edge_faces[saddle_mask, 0]] - d_var[..., edge_faces[saddle_mask, 1]]
    )

    return np.abs(edge_face_diff)


def _calculate_edge_node_difference(d_var, edge_nodes):
    """Helper function for computing the aboslute difference between the data
    values on each node that saddle each edge."""
    edge_node_diff = d_var[..., edge_nodes[:, 0]] - d_var[..., edge_nodes[:, 1]]

    return np.abs(edge_node_diff)


@njit(cache=True, inline="always")
def _compute_arc_length(lat_a, lat_b, lon_a, lon_b):
    dlat = np.radians(lat_b - lat_a)
    dlon = np.radians(lon_b - lon_a)

    lat_a = np.radians(lat_a)
    lat_b = np.radians(lat_b)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    distance = 2 * np.arcsin(np.sqrt(a))
    return distance


@njit(cache=True)
def _check_face_on_boundary(
    face_idx, face_node_connectivity, node_edge_connectivity, edge_face_connectivity
):
    bool_bdy = False
    for node_idx in face_node_connectivity[face_idx]:
        if node_idx != INT_FILL_VALUE:
            for edge_idx in node_edge_connectivity[node_idx]:
                if edge_idx != INT_FILL_VALUE:
                    if INT_FILL_VALUE in edge_face_connectivity[edge_idx]:
                        bool_bdy = True
    return bool_bdy


@njit(cache=True)
def _check_node_on_boundary_and_gather_node_neighbors(
    node_idx, node_edge_connectivity, edge_face_connectivity, edge_node_connectivity
):
    """Checks whether a node is on the boundary and returns a boolean
    Also, finds the neighboring nodes and returns a np array

    """

    node_neighbors = np.full(
        len(node_edge_connectivity[node_idx]), INT_FILL_VALUE, dtype=np.int64
    )
    num_node_neighbors = 0
    bool_bdy = False

    for edge_idx in node_edge_connectivity[node_idx]:
        if edge_idx != INT_FILL_VALUE:
            if INT_FILL_VALUE in edge_face_connectivity[edge_idx]:
                bool_bdy = True
            else:
                for other_node_idx in edge_node_connectivity[edge_idx]:
                    if other_node_idx != INT_FILL_VALUE and other_node_idx != node_idx:
                        node_neighbors[num_node_neighbors] = other_node_idx
                        num_node_neighbors += 1

    return bool_bdy, node_neighbors[0:num_node_neighbors]


def _compute_gradient(
    data,
    scale_by_radius=True,
    *,
    components=("zonal", "meridional"),
    _with_gradient_attrs=False,
    _keep_value_defining_attrs=False,
    **kw_apply_ufunc,
):
    """returns (zonal_gradient, meridional_gradient) as UxDataArrays, which are the
    zonal and meridional components of the gradient of the input UxDataArray.

    kw_apply_ufunc get passed to apply_ufunc for a function which takes `data` as input,
    and outputs grad_zonal, grad_meridional. E.g. it may be appropriate to provide
    kwargs like dask="parallelized" and output_dtypes=[float, float].

    components: tuple of str, default ("zonal", "meridional")
        which of "zonal" and "meridional" to compute; they are returned in the order
        given, always as a tuple (e.g. curl() only needs dv/dx and du/dy, so it asks
        for one component of each).
    _with_gradient_attrs: bool
        whether to add gradient-specific attributes to the output UxDataArrays.
        If False, just include units attribute. (False by default to avoid confusion if
        called externally as part of a different method. This adopts the stance that
        attributes should only be included if you are confident that they are appropriate.)
        If True, also include {"derivative": "gradient"}, "long_name", and "description".
    _keep_value_defining_attrs: bool, default False
        whether to keep value-defining attributes (like "standard_name", "long_name",
        "valid_min", "scale_factor"). Likely to produce misleading attributes when True,
        and also possibly data loss on netcdf read-back, so this is False by default.
    """
    uxgrid = data.uxgrid
    components = tuple(components)
    if not components or not set(components) <= {"zonal", "meridional"}:
        raise ValueError(
            "components must be a non-empty subset of ('zonal', 'meridional'); "
            f"got {components!r}"
        )
    # determine output dtype; necessary if inputs are parallelized.
    output_dtype = np.result_type(data.dtype, np.float64)
    kw_apply_ufunc.setdefault("output_dtypes", [output_dtype] * len(components))

    if data._face_centered():
        operators = dict(zip(("zonal", "meridional"), _face_gradient_operator(uxgrid)))
        selected = [operators[c] for c in components]

        def _kernel(values):
            # values: (..., n_face). The gradient is linear in the data, so every
            # slice shares one precomputed operator: one sparse product each.
            # The weights are float64, so the product is computed in float64
            # (or complex128) whatever the input precision.
            flat = values.reshape(-1, values.shape[-1]).T
            grads = tuple(
                (w @ flat).T.reshape(values.shape).astype(output_dtype, copy=False)
                for w in selected
            )
            return grads if len(grads) > 1 else grads[0]

        grads = _apply_1dfunc_with_grid_core_dim(
            _kernel,
            "n_face",
            data,
            n_outputs=len(components),
            vectorize=False,
            **kw_apply_ufunc,
        )
        if len(components) == 1:
            grads = (grads,)

    # TODO: Add support for this after merging face-centered implementation
    # elif data._node_centered():
    #     # Gradient of a Node-Centered Data Variable
    #     node_coords = np.array(
    #         [uxgrid.node_x.values, uxgrid.node_y.values, uxgrid.node_z.values]
    #     )
    #     node_lat = uxgrid.node_lat.values
    #     node_lon = uxgrid.node_lon.values
    #
    #     node_lon_rad = np.deg2rad(node_lon)
    #     node_lat_rad = np.deg2rad(node_lat)
    #     normal_lat = np.array(
    #         [
    #             -np.cos(node_lon_rad) * np.sin(node_lat_rad),
    #             -np.sin(node_lon_rad) * np.sin(node_lat_rad),
    #             np.cos(node_lat_rad),
    #         ]
    #     ).T
    #     normal_lon = np.array(
    #         [
    #             -np.sin(node_lon_rad),
    #             np.cos(node_lon_rad),
    #             np.zeros_like(node_lon_rad),
    #         ]
    #     ).T
    #
    #     grad_zonal, grad_meridional = _compute_gradients_on_nodes(
    #         data.values,
    #         uxgrid.n_node,
    #         node_coords,
    #         node_lat,
    #         node_lon,
    #         uxgrid.node_edge_connectivity.values,
    #         uxgrid.edge_face_connectivity.values,
    #         uxgrid.edge_node_connectivity.values,
    #         uxgrid.node_face_connectivity.values,
    #         normal_lat,
    #         normal_lon,
    #     )
    else:
        raise DataCenteringError(
            "_compute_gradient(data) is only supported for face-centered data; got "
            f"data.data_mapping={data.data_mapping!r}, data.sizes={dict(**data.sizes)}"
        )

    has_sphere_radius = "sphere_radius" in uxgrid._ds.attrs
    if scale_by_radius:
        if has_sphere_radius:
            radius = uxgrid.sphere_radius
            grads = tuple(grad / radius for grad in grads)
        else:
            warnings.warn(
                "scale_by_radius=True but the grid has no 'sphere_radius' "
                "attribute; result is left on the unit sphere. Set "
                "uxgrid.sphere_radius or pass scale_by_radius=False.",
                UserWarning,
                stacklevel=2,
            )

    base_units = data.attrs.get("units", "1")
    if "/" in base_units:  # put in (), e.g. (m/s)/m, not m/s/m.
        base_units = f"({base_units})"
    was_scaled = scale_by_radius and has_sphere_radius
    denom_units = "m" if was_scaled else "rad"
    grad_units = f"{base_units}/{denom_units}"

    results = []
    for component, grad in zip(components, grads):
        grad_da = grad.rename(f"{component}_gradient")

        # attrs bookkeeping
        if not _keep_value_defining_attrs:
            grad_da = _drop_value_describing_attrs(grad_da)
        grad_da.attrs["units"] = grad_units

        if _with_gradient_attrs:
            label = component.capitalize()  # "Zonal" / "Meridional"
            grad_da = grad_da.assign_attrs(
                {
                    "long_name": f"{label} gradient of {data.name}",
                    "derivative": "gradient",
                    "description": f"{label} component of gradient of {data.name}",
                }
            )
        results.append(grad_da)
    return tuple(results)


@njit(cache=True)
def _dual_cell_area(sx, sy, sz, angles, n):
    """Spherical area of the contour the Green-Gauss loop integrates around.

    ``sx``/``sy``/``sz`` hold the Cartesian centroids of the faces forming the
    contour in their first ``n`` entries; ``angles`` is a scratch buffer of at
    least ``n`` entries. All four are modified in place.

    The centroids arrive in connectivity order, which is not necessarily the
    order in which they trace the polygon, so they are sorted by azimuth about
    the contour centroid before the area is measured.

    Notes
    -----
    This replaces a call to :func:`uxarray.grid.area.calculate_face_area`, which
    integrated this area with 4th-order Gaussian quadrature: 16 quadrature
    points per fan triangle, each evaluating a spherical Jacobian. For a polygon
    bounded by great-circle arcs that integral has a closed form -- the
    spherical excess, via Van Oosterom and Strackee -- so the quadrature cost
    roughly 80x the exact answer (23.9 us vs 0.3 us per call) and was slightly
    less accurate.

    The fan triangles are summed *signed*, and the magnitude is taken once at
    the end. Dual cells are frequently non-convex -- a HealPix dual cell
    alternates near edge-neighbor centroids with far corner-neighbor centroids
    -- and for a non-convex polygon a fan from vertex 0 sweeps some triangles
    backwards. Those must cancel. The retired quadrature accumulated unsigned
    Jacobians instead, which double-counts the reversed triangles: measured
    against the exact ``d/dlat sin(lat) = cos(lat)`` solution on HealPix, the
    unsigned area leaves a worst-case gradient error of ~19% that does not
    improve with resolution (0.186 / 0.192 / 0.196 at z4 / z5 / z6), while the
    signed sum converges as expected (0.0127 / 0.0088 / 0.0051). Roughly 46% of
    z4 dual cells are affected, by up to 20% in area.

    Scale invariance is preserved from the quadrature, whose Jacobian divided
    the radius out. The excess formula has no such property, so the points are
    normalized onto the unit sphere first: some grids store
    ``face_x``/``face_y``/``face_z`` in meters rather than as unit vectors
    (e.g. the dyamond-30km test subset, at a radius of 6.37e6).
    """
    # The excess formula needs unit vectors; face centroids are not always stored
    # normalized.
    for i in range(n):
        inv_r = 1.0 / np.sqrt(sx[i] * sx[i] + sy[i] * sy[i] + sz[i] * sz[i])
        sx[i] *= inv_r
        sy[i] *= inv_r
        sz[i] *= inv_r

    # Contour centroid, used as the pole of the local azimuthal sort.
    cx = 0.0
    cy = 0.0
    cz = 0.0
    for i in range(n):
        cx += sx[i]
        cy += sy[i]
        cz += sz[i]
    inv_c = 1.0 / np.sqrt(cx * cx + cy * cy + cz * cz)
    cx *= inv_c
    cy *= inv_c
    cz *= inv_c

    # Build a local tangent basis at the centroid.
    if np.abs(cz) < 0.9:
        ax, ay, az = 0.0, 0.0, 1.0
    else:
        ax, ay, az = 1.0, 0.0, 0.0
    ex = ay * cz - az * cy
    ey = az * cx - ax * cz
    ez = ax * cy - ay * cx
    inv_e = 1.0 / np.sqrt(ex * ex + ey * ey + ez * ez)
    ex *= inv_e
    ey *= inv_e
    ez *= inv_e
    fx = cy * ez - cz * ey
    fy = cz * ex - cx * ez
    fz = cx * ey - cy * ex

    for i in range(n):
        angles[i] = np.arctan2(
            sx[i] * fx + sy[i] * fy + sz[i] * fz, sx[i] * ex + sy[i] * ey + sz[i] * ez
        )

    # Insertion sort by azimuth, carrying the coordinates along. n is bounded by
    # n_max_face_nodes * n_max_node_edges (order 10s), so this beats allocating
    # an argsort permutation and a second coordinate buffer per face.
    for i in range(1, n):
        key = angles[i]
        kx = sx[i]
        ky = sy[i]
        kz = sz[i]
        j = i - 1
        while j >= 0 and angles[j] > key:
            angles[j + 1] = angles[j]
            sx[j + 1] = sx[j]
            sy[j + 1] = sy[j]
            sz[j + 1] = sz[j]
            j -= 1
        angles[j + 1] = key
        sx[j + 1] = kx
        sy[j + 1] = ky
        sz[j + 1] = kz

    # Spherical excess of each fan triangle from vertex 0, summed signed so
    # that backward-swept triangles of a non-convex cell cancel.
    area = 0.0
    ax = sx[0]
    ay = sy[0]
    az = sz[0]
    for j in range(1, n - 1):
        bx = sx[j]
        by = sy[j]
        bz = sz[j]
        cx = sx[j + 1]
        cy = sy[j + 1]
        cz = sz[j + 1]
        triple = (
            ax * (by * cz - bz * cy)
            + ay * (bz * cx - bx * cz)
            + az * (bx * cy - by * cx)
        )
        denom = (
            1.0
            + (ax * bx + ay * by + az * bz)
            + (bx * cx + by * cy + bz * cz)
            + (cx * ax + cy * ay + cz * az)
        )
        area += 2.0 * np.arctan2(triple, denom)

    return np.abs(area)


# Entries of fixed-width scratch per build block: 2**22 * 24 B ~ 100 MB.
_OPERATOR_BUILD_BLOCK_ENTRIES = 2**22
# Grid variables the operator is built from.
_OPERATOR_SOURCES = (
    "face_x",
    "face_y",
    "face_z",
    "face_lat",
    "face_lon",
    "edge_face_connectivity",
    "face_node_connectivity",
    "node_edge_connectivity",
)


def _face_gradient_operator(uxgrid):
    """Return ``(w_zonal, w_meridional)``: sparse ``(n_face, n_face)`` operators
    such that ``w_zonal @ data`` is the zonal gradient of face-centered ``data``
    on the unit sphere (likewise meridional).

    The Green-Gauss gradient is linear in the data, and everything else it needs
    (stencil walk, normals, arc lengths, dual-cell areas) depends only on the
    grid. Building the operator once per grid and caching it means each data
    slice costs one sparse product instead of a full geometry pass.

    Rows whose contour is open (see :func:`_face_gradient_weights`) hold a
    single NaN weight, so they come out NaN for any data.

    The cache is keyed on the identity of the grid variables it was built from,
    so replacing any of them (property setters, ``construct_face_centers()``,
    ``normalize_cartesian_coordinates()``) rebuilds it. Editing their
    ``.values`` in place is not detected.
    """
    # Populate everything first: some grid properties are computed lazily, and
    # computing one (e.g. face_lat) can replace another (face_x).
    for name in _OPERATOR_SOURCES:
        getattr(uxgrid, name)
    sources = tuple(getattr(uxgrid, name).variable for name in _OPERATOR_SOURCES)
    cached = getattr(uxgrid, "_face_gradient_operator_cache", None)
    if cached is not None and all(a is b for a, b in zip(cached[0], sources)):
        return cached[1]

    from scipy import sparse

    # float64 geometry: some grids store float32 coordinates, and the weights
    # combine many products of them.
    face_lat = uxgrid.face_lat.values.astype(np.float64)
    face_lon = uxgrid.face_lon.values.astype(np.float64)
    face_lon_rad = np.deg2rad(face_lon)
    face_lat_rad = np.deg2rad(face_lat)
    normal_lat = np.array(
        [
            -np.cos(face_lon_rad) * np.sin(face_lat_rad),
            -np.sin(face_lon_rad) * np.sin(face_lat_rad),
            np.cos(face_lat_rad),
        ]
    ).T
    normal_lon = np.array(
        [
            -np.sin(face_lon_rad),
            np.cos(face_lon_rad),
            np.zeros_like(face_lon_rad),
        ]
    ).T

    face_coords = np.array(
        [uxgrid.face_x.values, uxgrid.face_y.values, uxgrid.face_z.values],
        dtype=np.float64,
    ).T
    face_node_connectivity = uxgrid.face_node_connectivity.values
    node_edge_connectivity = uxgrid.node_edge_connectivity.values
    edge_face_connectivity = uxgrid.edge_face_connectivity.values

    # The kernel writes fixed-width rows (max stencil + the face itself); build
    # them a block of faces at a time so scratch stays bounded on any grid,
    # keeping only the used slots of each row.
    n_face = uxgrid.n_face
    width = face_node_connectivity.shape[1] * node_edge_connectivity.shape[1] + 1
    block = max(1, _OPERATOR_BUILD_BLOCK_ENTRIES // width)
    index_dtype = np.int32 if n_face < np.iinfo(np.int32).max else np.int64
    cols, w_zonal, w_meridional, counts = [], [], [], []
    for start in range(0, n_face, block):
        stop = min(start + block, n_face)
        b_cols, b_zonal, b_meridional, b_counts = _face_gradient_weights(
            start,
            stop,
            face_coords,
            edge_face_connectivity,
            face_node_connectivity,
            node_edge_connectivity,
            face_lat,
            face_lon,
            normal_lon,
            normal_lat,
        )
        used = np.arange(width) < b_counts[:, None]
        cols.append(b_cols[used].astype(index_dtype))
        w_zonal.append(b_zonal[used])
        w_meridional.append(b_meridional[used])
        counts.append(b_counts)

    cols = np.concatenate(cols)
    indptr = np.concatenate(([0], np.cumsum(np.concatenate(counts)))).astype(
        index_dtype
    )
    shape = (n_face, n_face)
    operator = (
        sparse.csr_matrix((np.concatenate(w_zonal), cols, indptr), shape=shape),
        sparse.csr_matrix((np.concatenate(w_meridional), cols, indptr), shape=shape),
    )
    uxgrid._face_gradient_operator_cache = (sources, operator)
    return operator


@njit(cache=True, parallel=True, nogil=True)
def _face_gradient_weights(
    face_start,
    face_stop,
    face_coords,
    edge_face_connectivity,
    face_node_connectivity,
    node_edge_connectivity,
    face_lat,
    face_lon,
    normal_lon,
    normal_lat,
):
    """
    Weights of the horizontal gradient on faces, averaged over the cell constructed
    from connecting the centroids of the faces which share a common node with the face.

    Each contour edge between neighbor faces f1 and f2 contributes
    ``((d[f1] + d[f2]) / 2 - d[face]) * c`` to the face's gradient, so the gradient
    is ``sum(w[s] * d[s])`` over the face itself and its stencil faces.

    Covers faces ``face_start:face_stop``; row ``k`` is face ``face_start + k``.

    Returns
    -------
    cols, w_zonal, w_meridional : np.ndarray, shape (n_rows, max_stencil + 1)
        Fixed-width rows: column index and zonal/meridional weight. Slot 0 is the
        face itself; only the first ``counts[k]`` slots of row ``k`` are used.
    counts : np.ndarray, shape (n_rows,)

    Notes
    -----
    Combined ideas from:
    - Strategy (3) in Barth, Timothy, and Dennis Jespersen. "The design and application of upwind schemes on unstructured meshes." 27th Aerospace sciences meeting. 1989.
    - Equation (11) in Tomita, Hirofumi, et al. "Shallow water model on a modified icosahedral geodesic grid by using spring dynamics." Journal of Computational Physics 174.2 (2001): 579-613.
    """
    n_face_nodes = face_node_connectivity.shape[1]
    n_node_edges = node_edge_connectivity.shape[1]
    max_stencil = n_face_nodes * n_node_edges

    n_rows = face_stop - face_start
    cols = np.empty((n_rows, max_stencil + 1), dtype=np.int64)
    w_zonal = np.zeros((n_rows, max_stencil + 1))
    w_meridional = np.zeros((n_rows, max_stencil + 1))
    counts = np.empty(n_rows, dtype=np.int64)

    # Parallel across faces
    for row in prange(n_rows):
        face_idx = face_start + row
        # Centroids of the faces forming the contour, collected as the loop
        # walks it so the normalizing area matches the region integrated over.
        stencil_x = np.empty(max_stencil)
        stencil_y = np.empty(max_stencil)
        stencil_z = np.empty(max_stencil)
        angles = np.empty(max_stencil)
        n_stencil = 0
        cols[row, 0] = face_idx

        has_contribution = False

        # Green-Gauss only applies to a closed contour. If any edge in this
        # face's node neighborhood is missing a second face, the contour is
        # open and no enclosed area exists.
        contour_closed = True

        face_x = face_coords[face_idx, 0]
        face_y = face_coords[face_idx, 1]
        face_z = face_coords[face_idx, 2]

        for i in range(n_face_nodes):  # take each node on that face
            node_idx = face_node_connectivity[face_idx, i]
            if node_idx == INT_FILL_VALUE:
                continue

            for j in range(n_node_edges):  # grab each edge connected to that node
                edge_idx = node_edge_connectivity[node_idx, j]
                if edge_idx == INT_FILL_VALUE:
                    continue

                # edge_face_connectivity is always (n_edge, 2), so the two
                # neighbors can be compared directly.
                face1_idx = edge_face_connectivity[edge_idx, 0]
                face2_idx = edge_face_connectivity[edge_idx, 1]

                # Skip edges that lack a second face neighbor instead of
                # NaN-ing the entire face.  Fixes grids where
                # edge_face_connectivity has spurious INT_FILL_VALUE entries
                # (e.g. SCRIP-derived SE grids like ne120np4).  See #1452.
                if face1_idx == INT_FILL_VALUE or face2_idx == INT_FILL_VALUE:
                    contour_closed = False
                    continue

                # skip edges connected to the original face
                if face1_idx == face_idx or face2_idx == face_idx:
                    continue

                face1_x = face_coords[face1_idx, 0]
                face1_y = face_coords[face1_idx, 1]
                face1_z = face_coords[face1_idx, 2]
                face2_x = face_coords[face2_idx, 0]
                face2_y = face_coords[face2_idx, 1]
                face2_z = face_coords[face2_idx, 2]

                # compute normal pointing outwards from face
                cross_x = face1_y * face2_z - face1_z * face2_y
                cross_y = face1_z * face2_x - face1_x * face2_z
                cross_z = face1_x * face2_y - face1_y * face2_x
                norm = np.sqrt(
                    cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
                )
                if (
                    cross_x * (face1_x - face_x)
                    + cross_y * (face1_y - face_y)
                    + cross_z * (face1_z - face_z)
                ) > 0:
                    inv_norm = 1.0 / norm
                else:
                    inv_norm = -1.0 / norm

                # compute arc length between the two faces
                arc_length = _compute_arc_length(
                    face_lat[face1_idx],
                    face_lat[face2_idx],
                    face_lon[face1_idx],
                    face_lon[face2_idx],
                )

                # this edge adds (trapz - d[face]) * coef * cross to the gradient,
                # with trapz = (d[face1] + d[face2]) / 2; project cross now.
                coef = arc_length * inv_norm
                c_zonal = coef * (
                    cross_x * normal_lon[face_idx, 0]
                    + cross_y * normal_lon[face_idx, 1]
                    + cross_z * normal_lon[face_idx, 2]
                )
                c_meridional = coef * (
                    cross_x * normal_lat[face_idx, 0]
                    + cross_y * normal_lat[face_idx, 1]
                    + cross_z * normal_lat[face_idx, 2]
                )
                w_zonal[row, 0] -= c_zonal
                w_meridional[row, 0] -= c_meridional
                has_contribution = True

                for cand in (face1_idx, face2_idx):
                    slot = -1
                    for s in range(n_stencil):
                        if cols[row, s + 1] == cand:
                            slot = s
                            break
                    if slot < 0:
                        slot = n_stencil
                        cols[row, slot + 1] = cand
                        stencil_x[slot] = face_coords[cand, 0]
                        stencil_y[slot] = face_coords[cand, 1]
                        stencil_z[slot] = face_coords[cand, 2]
                        n_stencil += 1
                    w_zonal[row, slot + 1] += 0.5 * c_zonal
                    w_meridional[row, slot + 1] += 0.5 * c_meridional

        # The contour must be closed, and a polygon, before it encloses an area.
        # A lone NaN weight makes the row NaN for any data.
        if not has_contribution or not contour_closed or n_stencil < 3:
            w_zonal[row, 0] = np.nan
            w_meridional[row, 0] = np.nan
            counts[row] = 1
            continue

        inv_area = 1.0 / _dual_cell_area(
            stencil_x, stencil_y, stencil_z, angles, n_stencil
        )
        for s in range(n_stencil + 1):
            w_zonal[row, s] *= inv_area
            w_meridional[row, s] *= inv_area
        counts[row] = n_stencil + 1

    return cols, w_zonal, w_meridional, counts


# TODO: Add support for this after merging face-centered implementation
# @njit(cache=True, parallel=True)
# def _compute_gradients_on_nodes(
#     data,
#     n_node,
#     node_coords,
#     node_lat,
#     node_lon,
#     node_edge_connectivity,
#     edge_face_connectivity,
#     edge_node_connectivity,
#     node_face_connectivity,
#     normal_lat,
#     normal_lon,
# ):
#     """
#     Computes horizontal gradients on nodes averaged over the cell constructed from connecting neighboring nodes which share a common edge with the node.
#
#     Combined ideas from:
#         Strategy (3) in Barth, Timothy, and Dennis Jespersen. "The design and application of upwind schemes on unstructured meshes." 27th Aerospace sciences meeting. 1989.
#
#         Equation (11) in Tomita, Hirofumi, et al. "Shallow water model on a modified icosahedral geodesic grid by using spring dynamics." Journal of Computational Physics 174.2 (2001): 579-613.
#
#     Returns:
#         two np.ndarray: (n_node,) for zonal_grad & meridional_grad
#
#     """
#     gradients_nodes = np.zeros((n_node, 2))
#
#     for node_idx in prange(n_node):
#         gradient = np.zeros(3)
#
#         bool_bdy, node_neighbors = _check_node_on_boundary_and_gather_node_neighbors(
#             node_idx,
#             node_edge_connectivity,
#             edge_face_connectivity,
#             edge_node_connectivity,
#         )
#
#         node_neighbors = node_neighbors.astype(np.int64)
#
#         if not bool_bdy:  # if node is not on the boundary
#             for node1_idx in node_neighbors:
#                 for node2_idx in node_neighbors:
#                     if node1_idx > node2_idx:  # to avoid double counting
#                         if (
#                             np.intersect1d(
#                                 node_face_connectivity[node1_idx],
#                                 node_face_connectivity[node2_idx],
#                             ).size
#                             > 0
#                         ):  # check if nodes have a common face
#                             node1_coords = node_coords[:, node1_idx]
#                             node2_coords = node_coords[:, node2_idx]
#
#                             # compute normal that is pointing outwards from center node
#                             cross = np.cross(node1_coords, node2_coords)
#                             norm = np.linalg.norm(cross)
#                             if (
#                                 np.dot(cross, node1_coords - node_coords[:, node_idx])
#                                 > 0
#                             ):
#                                 normal = cross / norm
#                             else:
#                                 normal = -cross / norm
#
#                             # compute arc length between the two faces
#                             arc_length = _compute_arc_length(
#                                 node_lat[node1_idx],
#                                 node_lat[node2_idx],
#                                 node_lon[node1_idx],
#                                 node_lon[node2_idx],
#                             )
#
#                             # compute trapezoidal rule
#                             trapz = (data[node1_idx] + data[node2_idx]) / 2
#
#                             # add to the gradient (subtract correction term)
#                             gradient = (
#                                 gradient
#                                 + (trapz - data[node_idx]) * arc_length * normal
#                             )
#
#             # Normalize and project zonal and meridional components and store the result for the current node
#             gradients_nodes[node_idx, 0], gradients_nodes[node_idx, 1] = (
#                 _normalize_and_project_gradient(
#                     gradient,
#                     node_idx,
#                     normal_lat,
#                     normal_lon,
#                     node_coords,
#                     node_neighbors,
#                 )
#             )
#
#     return gradients_nodes[:, 0], gradients_nodes[:, 1]
