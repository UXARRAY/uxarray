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


# @njit(cache=True)
# def _compute_arc_length(lat_a, lat_b, lon_a, lon_b):
#     '''
#     input: latitude and longitude in degrees
#     computes using law of cosines
#     Returns: arc length on unit sphere in radians
#     '''
#     radlon_a = np.deg2rad(lon_a)
#     radlon_b = np.deg2rad(lon_b)
#
#     radlat_a = np.deg2rad(lat_a)
#     radlat_b = np.deg2rad(lat_b)
#
#     # arc length
#     distance = np.arccos(
#         np.sin(radlat_a) * np.sin(radlat_b)
#         + np.cos(radlat_a) * np.cos(radlat_b) * np.cos(radlon_a - radlon_b)
#     )
#     return distance


@njit(cache=True)
def _compute_arc_length(lat_a, lat_b, lon_a, lon_b):
    """
    input: latitude and longitude in degrees

    Computes the haversine distance between two points on the unit sphere.

    returns: spherical arc length in radians
    """

    dlat = np.radians(lat_b - lat_a)
    dlon = np.radians(lon_b - lon_a)

    lat_a = np.radians(lat_a)
    lat_b = np.radians(lat_b)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    distance = 2 * np.arcsin(np.sqrt(a))
    return distance


@njit(cache=True)
def _check_face_on_boundary(
    face_idx: np.integer,
    face_edge_connectivity: np.ndarray,
    edge_face_connectivity: np.ndarray,
):
    """
    Checks if a face in on the boundary by checking the neighboring faces of its edges

    Returns: boolean
    True - boundary & False - not boundary

    """
    bool_bdy = False
    for edge_idx in face_edge_connectivity[face_idx]:
        if edge_idx != INT_FILL_VALUE:
            if INT_FILL_VALUE in edge_face_connectivity[edge_idx]:
                bool_bdy = True

    return bool_bdy


@njit(cache=True)
def _check_node_on_boundary_and_gather_node_neighbors(
    node_idx, node_edge_connectivity, edge_face_connectivity, edge_node_connectivity
):
    """
    Checks whether a node is on the boundary and returns a boolean
    Also, finds the neighboring nodes and returns a np array

    :returns
    bool_bdy - True on boundary, False not on boundary

    node_neighbors: np.array of all the neighboring nodes
    """

    # pre-allocate

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


@njit(cache=True, parallel=True)
def _compute_gradients_on_faces(
    data,
    n_face,
    face_coords,
    face_edge_connectivity,
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

    Combined ideas from:
        Strategy (3) in Barth, Timothy, and Dennis Jespersen. "The design and application of upwind schemes on unstructured meshes." 27th Aerospace sciences meeting. 1989.

        Equation (11) in Tomita, Hirofumi, et al. "Shallow water model on a modified icosahedral geodesic grid by using spring dynamics." Journal of Computational Physics 174.2 (2001): 579-613.

    Returns:
        two np.ndarray: (n_face,) for zonal_grad & meridional_grad

    """
    gradients_faces = np.zeros((n_face, 2))

    for face_idx in prange(n_face):
        gradient = np.zeros(3)

        if not _check_face_on_boundary(
            face_idx, face_edge_connectivity, edge_face_connectivity
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

                                # compute normal that is pointing outwards from face
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

        # divide gradient by the area of the face

        node_neighbors = face_node_connectivity[face_idx]
        node_neighbors = node_neighbors[~np.isin(node_neighbors, INT_FILL_VALUE)]

        [area, jacobian] = calculate_face_area(
            node_coords[0, node_neighbors].astype(np.float64),
            node_coords[1, node_neighbors].astype(np.float64),
            node_coords[2, node_neighbors].astype(np.float64),
        )

        gradient = gradient / area

        # projection to horizontal gradient

        zonal_grad = np.sum(gradient * normal_lon[face_idx])
        meridional_grad = np.sum(gradient * normal_lat[face_idx])

        gradients_faces[face_idx, 0] = zonal_grad
        gradients_faces[face_idx, 1] = meridional_grad

    return gradients_faces[:, 0], gradients_faces[:, 1]


@njit(cache=True, parallel=True)
def _compute_gradients_on_nodes(
    data,
    n_node,
    node_coords,
    node_lat,
    node_lon,
    node_edge_connectivity,
    edge_face_connectivity,
    edge_node_connectivity,
    node_face_connectivity,
    normal_lat,
    normal_lon,
):
    """
    Computes horizontal gradients on nodes averaged over the cell constructed from connecting neighboring nodes which share a common edge with the node.

    Combined ideas from:
        Strategy (3) in Barth, Timothy, and Dennis Jespersen. "The design and application of upwind schemes on unstructured meshes." 27th Aerospace sciences meeting. 1989.

        Equation (11) in Tomita, Hirofumi, et al. "Shallow water model on a modified icosahedral geodesic grid by using spring dynamics." Journal of Computational Physics 174.2 (2001): 579-613.

    Returns:
        two np.ndarray: (n_node,) for zonal_grad & meridional_grad

    """
    gradients_nodes = np.zeros((n_node, 2))

    for node_idx in prange(n_node):
        gradient = np.zeros(3)

        bool_bdy, node_neighbors = _check_node_on_boundary_and_gather_node_neighbors(
            node_idx,
            node_edge_connectivity,
            edge_face_connectivity,
            edge_node_connectivity,
        )

        node_neighbors = node_neighbors.astype(np.int64)

        if not bool_bdy:  # if node is not on the boundary
            for node1_idx in node_neighbors:
                for node2_idx in node_neighbors:
                    if node1_idx > node2_idx:  # to avoid double counting
                        if (
                            np.intersect1d(
                                node_face_connectivity[node1_idx],
                                node_face_connectivity[node2_idx],
                            ).size
                            > 0
                        ):  # check if nodes have a common face
                            node1_coords = node_coords[:, node1_idx]
                            node2_coords = node_coords[:, node2_idx]

                            # compute normal that is pointing outwards from center node
                            cross = np.cross(node1_coords, node2_coords)
                            norm = np.linalg.norm(cross)
                            if (
                                np.dot(cross, node1_coords - node_coords[:, node_idx])
                                > 0
                            ):
                                normal = cross / norm
                            else:
                                normal = -cross / norm

                            # compute arc length between the two faces
                            arc_length = _compute_arc_length(
                                node_lat[node1_idx],
                                node_lat[node2_idx],
                                node_lon[node1_idx],
                                node_lon[node2_idx],
                            )

                            # compute trapezoidal rule
                            trapz = (data[node1_idx] + data[node2_idx]) / 2

                            # add to the gradient
                            gradient = (
                                gradient
                                + (trapz - data[node_idx]) * arc_length * normal
                            )

            [area, jacobian] = calculate_face_area(
                node_coords[0, node_neighbors].astype(np.float64),
                node_coords[1, node_neighbors].astype(np.float64),
                node_coords[2, node_neighbors].astype(np.float64),
            )

            gradient = gradient / area

            # projection to horizontal gradient
            zonal_grad = np.sum(gradient * normal_lon[node_idx])

            meridional_grad = np.sum(gradient * normal_lat[node_idx])

            gradients_nodes[node_idx, 0] = zonal_grad
            gradients_nodes[node_idx, 1] = meridional_grad

    return gradients_nodes[:, 0], gradients_nodes[:, 1]


# TODO: Commented out for now
# def _calculate_grad_on_edge_from_faces(
#     d_var, edge_faces, n_edge, edge_face_distances, normalize: Optional[bool] = False
# ):
#     """Helper function for computing the horizontal gradient of a field on each
#     cell using values at adjacent cells.
#
#     The expression for calculating the gradient on each edge comes from
#     Eq. 22 in Ringler et al. (2010), J. Comput. Phys.
#
#     Code is adapted from
#     https://github.com/theweathermanda/MPAS_utilities/blob/main/mpas_calc_operators.py
#     """
#
#     # obtain all edges that saddle two faces
#     saddle_mask = edge_faces[:, 1] != INT_FILL_VALUE
#
#     grad = _calculate_edge_face_difference(d_var, edge_faces, n_edge)
#
#     grad[..., saddle_mask] = grad[..., saddle_mask] / edge_face_distances[saddle_mask]
#
#     if normalize:
#         grad = grad / np.linalg.norm(grad)
#
#     return grad
