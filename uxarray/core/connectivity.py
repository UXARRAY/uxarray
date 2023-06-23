import numpy as np

from uxarray.utils.constants import INT_DTYPE, INT_FILL_VALUE


def close_face_nodes(Mesh2_face_nodes, nMesh2_face, nMaxMesh2_face_nodes):
    """Closes (``Mesh2_face_nodes``) by inserting the first node index after
    the last non-fill-value node.

    Parameters
    ----------
    Mesh2_face_nodes : np.ndarray
        Connectivity array for constructing a face from its nodes
    nMesh2_face : constant
        Number of faces
    nMaxMesh2_face_nodes : constant
        Max number of nodes that compose a face

    Returns
    ----------
    closed : ndarray
        Closed (padded) Mesh2_face_nodes

    Example
    ----------
    Given face nodes with shape [2 x 5]
        [0, 1, 2, 3, FILL_VALUE]
        [4, 5, 6, 7, 8]
    Pads them to the following with shape [2 x 6]
        [0, 1, 2, 3, 0, FILL_VALUE]
        [4, 5, 6, 7, 8, 4]
    """

    # padding to shape [nMesh2_face, nMaxMesh2_face_nodes + 1]
    closed = np.ones((nMesh2_face, nMaxMesh2_face_nodes + 1),
                     dtype=INT_DTYPE) * INT_FILL_VALUE

    # set all non-paded values to original face nodee values
    closed[:, :-1] = Mesh2_face_nodes.copy()

    # instance of first fill value
    first_fv_idx_2d = np.argmax(closed == INT_FILL_VALUE, axis=1)

    # 2d to 1d index for np.put()
    first_fv_idx_1d = first_fv_idx_2d + (
        (nMaxMesh2_face_nodes + 1) * np.arange(0, nMesh2_face))

    # column of first node values
    first_node_value = Mesh2_face_nodes[:, 0].copy()

    # insert first node column at occurrence of first fill value
    np.put(closed.ravel(), first_fv_idx_1d, first_node_value)

    return closed


def build_polygon_verts(Mesh2_node_x, Mesh2_node_y, Mesh2_face_nodes,
                        nMesh2_face, nMaxMesh2_face_nodes, nNodes_per_face):

    closed_face_nodes = close_face_nodes(Mesh2_face_nodes, nMesh2_face,
                                         nMaxMesh2_face_nodes)

    polygon_verts = []
    for face_nodes, max_n_nodes in zip(Mesh2_face_nodes, nNodes_per_face):
        polygon_x = Mesh2_node_x[face_nodes[0:max_n_nodes]]
        polygon_y = Mesh2_node_y[face_nodes[0:max_n_nodes]]

        cur_polygon_verts = np.array([polygon_x, polygon_y])
        polygon_verts.append(cur_polygon_verts.T)

    return polygon_verts
