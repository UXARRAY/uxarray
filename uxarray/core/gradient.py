import numpy as np
from numba import njit, prange

from uxarray.constants import INT_FILL_VALUE
from uxarray.grid.area import calculate_face_area


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


@njit(cache=True)
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


def _compute_gradient(data):
    from uxarray import UxDataArray

    uxgrid = data.uxgrid

    if data.ndim > 1:
        raise ValueError(
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

        node_coords = np.array(
            [uxgrid.node_x.values, uxgrid.node_y.values, uxgrid.node_z.values]
        )

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
            node_coords,
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
        raise ValueError(
            "Computing the gradient is only supported for face-centered data variables."
        )

    # Zonal
    grad_zonal_da = UxDataArray(
        data=grad_zonal, name="zonal_gradient", dims=data.dims, uxgrid=uxgrid
    )

    # Meridional
    grad_meridional_da = UxDataArray(
        data=grad_meridional, name="meridional_gradient", dims=data.dims, uxgrid=uxgrid
    )

    return grad_zonal_da, grad_meridional_da


@njit(cache=True)
def _normalize_and_project_gradient(
    gradient, index, normal_lat, normal_lon, node_coords, node_neighbors
):
    area, _ = calculate_face_area(
        node_coords[0, node_neighbors].astype(np.float64),
        node_coords[1, node_neighbors].astype(np.float64),
        node_coords[2, node_neighbors].astype(np.float64),
    )

    gradient = gradient / area

    # projection to horizontal gradient
    zonal_grad = np.sum(gradient * normal_lon[index])
    meridional_grad = np.sum(gradient * normal_lat[index])

    return zonal_grad, meridional_grad


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
    node_coords,
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

    gradients_faces = np.full((n_face, 2), np.nan)

    # Parallel across faces
    for face_idx in prange(n_face):
        gradient = np.zeros(3)

        if not _check_face_on_boundary(
            face_idx,
            face_node_connectivity,
            node_edge_connectivity,
            edge_face_connectivity,
        ):  # check face is not on boundary
            for node_idx in face_node_connectivity[
                face_idx
            ]:  # take each node on that face
                if node_idx != INT_FILL_VALUE:
                    for edge_idx in node_edge_connectivity[
                        node_idx
                    ]:  # grab each edge connected to that node
                        if edge_idx != INT_FILL_VALUE:
                            if (
                                face_idx not in edge_face_connectivity[edge_idx]
                            ):  # check if edge connected to original face
                                face1_idx = edge_face_connectivity[edge_idx][0]
                                face2_idx = edge_face_connectivity[edge_idx][1]

                                face1_coords = face_coords[face1_idx]
                                face2_coords = face_coords[face2_idx]

                                # compute normal pointing outwards from face
                                cross = np.cross(face1_coords, face2_coords)
                                norm = np.linalg.norm(cross)
                                if (
                                    np.dot(cross, face1_coords - face_coords[face_idx])
                                    > 0
                                ):
                                    normal = cross / norm
                                else:
                                    normal = -cross / norm

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
                                gradient = (
                                    gradient
                                    + (trapz - data[face_idx]) * arc_length * normal
                                )

        else:
            gradient = np.full(3, np.nan)

        node_neighbors = face_node_connectivity[face_idx]
        node_neighbors = node_neighbors[node_neighbors != INT_FILL_VALUE]

        # Normalize and project zonal and meridional components and store the result for the current face
        gradients_faces[face_idx, 0], gradients_faces[face_idx, 1] = (
            _normalize_and_project_gradient(
                gradient, face_idx, normal_lat, normal_lon, node_coords, node_neighbors
            )
        )

    return gradients_faces[:, 0], gradients_faces[:, 1]


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
