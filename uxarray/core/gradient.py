import warnings

import numpy as np
from numba import njit, prange

from uxarray.constants import INT_FILL_VALUE
from uxarray.errors import DataCenteringError, DimensionError


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


def _compute_gradient(data, scale_by_radius=True):
    from uxarray import UxDataArray

    uxgrid = data.uxgrid

    if data.ndim > 1:
        raise DimensionError(
            "Gradient currently requires 1D face-centered data. Consider "
            "reducing the dimension by selecting data across leading dimensions (e.g., `.isel(time=0)`, "
            "`.sel(lev=500)`, or `.mean('time')`). "
        )

    if data._face_centered():
        face_coords = np.array(
            [uxgrid.face_x.values, uxgrid.face_y.values, uxgrid.face_z.values]
        ).T
        face_lat = uxgrid.face_lat.values
        face_lon = uxgrid.face_lon.values

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

        grad_zonal, grad_meridional = _compute_gradients_on_faces(
            data.values,
            uxgrid.n_face,
            face_coords,
            uxgrid.edge_face_connectivity.values,
            uxgrid.face_node_connectivity.values,
            uxgrid.node_edge_connectivity.values,
            face_lat,
            face_lon,
            normal_lon,
            normal_lat,
        )

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
            "Computing the gradient is only supported for face-centered data variables."
        )

    has_sphere_radius = "sphere_radius" in uxgrid._ds.attrs
    if scale_by_radius:
        if has_sphere_radius:
            radius = uxgrid.sphere_radius
            grad_zonal = grad_zonal / radius
            grad_meridional = grad_meridional / radius
        else:
            warnings.warn(
                "scale_by_radius=True but the grid has no 'sphere_radius' "
                "attribute; result is left on the unit sphere. Set "
                "uxgrid.sphere_radius or pass scale_by_radius=False.",
                UserWarning,
                stacklevel=2,
            )

    base_units = data.attrs.get("units", "")
    if scale_by_radius and has_sphere_radius:
        grad_units = f"{base_units}/m" if base_units else "1/m"
    else:
        grad_units = f"{base_units}/rad" if base_units else "1/rad"

    # Zonal
    grad_zonal_da = UxDataArray(
        data=grad_zonal, name="zonal_gradient", dims=data.dims, uxgrid=uxgrid
    )
    grad_zonal_da.attrs["units"] = grad_units

    # Meridional
    grad_meridional_da = UxDataArray(
        data=grad_meridional, name="meridional_gradient", dims=data.dims, uxgrid=uxgrid
    )
    grad_meridional_da.attrs["units"] = grad_units

    return grad_zonal_da, grad_meridional_da


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

    Two properties of the retired quadrature are reproduced deliberately:

    - It accumulated *unsigned* Jacobians, i.e. an unsigned sum over fan
      triangles from vertex 0. Dual cells are often non-convex -- a HealPix dual
      cell alternates near edge-neighbor centroids with far corner-neighbor
      centroids -- and for those a signed excess sum differs from an unsigned
      one by ~1%. Summing ``abs`` per triangle matches the previous area to
      ~5e-14; a signed sum does not. This is a compatibility choice, not a claim
      that unsigned is the geometrically correct convention for a
      self-overlapping fan.
    - It was scale-invariant, because its Jacobian divided the radius out. The
      excess formula is not, so the points are normalized onto the unit sphere
      first: some grids store ``face_x``/``face_y``/``face_z`` in meters rather
      than as unit vectors (e.g. the dyamond-30km test subset, at a radius of
      6.37e6).

    Gradients therefore shift by up to ~5e-6 relative to the quadrature-based
    implementation. That gap is the retired quadrature's own truncation error;
    the closed form is the more accurate value.
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

    # Spherical excess of each fan triangle from vertex 0, summed unsigned.
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
        area += np.abs(2.0 * np.arctan2(triple, denom))

    return area


@njit(cache=True, parallel=True)
def _compute_gradients_on_faces(
    data,
    n_face,
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
    Computes horizontal gradients on faces averaged over the cell constructed from connecting the centroids of the faces which share a common node with the face.


    Parameters
    ----------
    data : np.ndarray
        Array containing the data to compute gradients on, must be face-centered
    n_face: int
        TODO
    face_coords: np.ndarray
        TODO

    Returns
    -------
    gradient_zonal: np.ndarray
        Zonal component of gradient ...
    gradient_meridional: np.ndarray
        Meridional component of gradient ...

    Notes
    -----

    Combined ideas from:
    - Strategy (3) in Barth, Timothy, and Dennis Jespersen. "The design and application of upwind schemes on unstructured meshes." 27th Aerospace sciences meeting. 1989.
    - Equation (11) in Tomita, Hirofumi, et al. "Shallow water model on a modified icosahedral geodesic grid by using spring dynamics." Journal of Computational Physics 174.2 (2001): 579-613.

    Returns:

        two np.ndarray: (n_face,) for zonal_grad & meridional_grad

    """

    gradient_zonal = np.empty(n_face)
    gradient_meridional = np.empty(n_face)

    n_face_nodes = face_node_connectivity.shape[1]
    n_node_edges = node_edge_connectivity.shape[1]
    max_stencil = n_face_nodes * n_node_edges

    # Parallel across faces
    for face_idx in prange(n_face):
        # Centroids of the faces forming the contour, collected as the loop
        # walks it so the normalizing area matches the region integrated over.
        stencil = np.empty(max_stencil, dtype=np.int64)
        stencil_x = np.empty(max_stencil)
        stencil_y = np.empty(max_stencil)
        stencil_z = np.empty(max_stencil)
        angles = np.empty(max_stencil)
        n_stencil = 0

        # Gradient accumulated component-wise to keep the inner loop free of
        # temporary arrays.
        grad_x = 0.0
        grad_y = 0.0
        grad_z = 0.0
        has_contribution = False

        # Green-Gauss only applies to a closed contour. If any edge in this
        # face's node neighborhood is missing a second face, the contour is
        # open and no enclosed area exists.
        contour_closed = True

        data_face = data[face_idx]
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

                # compute trapezoidal rule
                trapz = (data[face1_idx] + data[face2_idx]) / 2

                # add to the gradient (subtract correction term)
                weight = (trapz - data_face) * arc_length * inv_norm
                grad_x += weight * cross_x
                grad_y += weight * cross_y
                grad_z += weight * cross_z
                has_contribution = True

                for cand in (face1_idx, face2_idx):
                    seen = False
                    for s in range(n_stencil):
                        if stencil[s] == cand:
                            seen = True
                            break
                    if not seen:
                        stencil[n_stencil] = cand
                        stencil_x[n_stencil] = face_coords[cand, 0]
                        stencil_y[n_stencil] = face_coords[cand, 1]
                        stencil_z[n_stencil] = face_coords[cand, 2]
                        n_stencil += 1

        # The contour must be closed, and a polygon, before it encloses an area.
        if not has_contribution or not contour_closed or n_stencil < 3:
            gradient_zonal[face_idx] = np.nan
            gradient_meridional[face_idx] = np.nan
            continue

        area = _dual_cell_area(stencil_x, stencil_y, stencil_z, angles, n_stencil)

        # Normalize and project zonal and meridional components and store the result for the current face
        inv_area = 1.0 / area
        gradient_zonal[face_idx] = (
            grad_x * normal_lon[face_idx, 0]
            + grad_y * normal_lon[face_idx, 1]
            + grad_z * normal_lon[face_idx, 2]
        ) * inv_area
        gradient_meridional[face_idx] = (
            grad_x * normal_lat[face_idx, 0]
            + grad_y * normal_lat[face_idx, 1]
            + grad_z * normal_lat[face_idx, 2]
        ) * inv_area

    return gradient_zonal, gradient_meridional


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
