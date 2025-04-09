import numpy as np

from uxarray.constants import INT_FILL_VALUE
from uxarray.grid.utils import _swap_first_fill_value_with_last


def _construct_face_edge_nodes_cartesian(
    face_node_conn, n_face, n_max_face_edges, node_x, node_y, node_z
):
    """Construct an array to hold the edge Cartesian coordinates connectivity
    for multiple faces in a grid.

    Parameters
    ----------
    face_node_conn : np.ndarray
        An array of shape (n_face, n_max_face_edges) containing the node indices for each face. Accessed through `grid.face_node_connectivity.value`.
    n_face : int
        The number of faces in the grid. Accessed through `grid.n_face`.
    n_max_face_edges : int
        The maximum number of edges for any face in the grid. Accessed through `grid.n_max_face_edges`.
    node_x : np.ndarray
        An array of shape (n_nodes,) containing the x-coordinate values of the nodes. Accessed through `grid.node_x`.
    node_y : np.ndarray
        An array of shape (n_nodes,) containing the y-coordinate values of the nodes. Accessed through `grid.node_y`.
    node_z : np.ndarray
        An array of shape (n_nodes,) containing the z-coordinate values of the nodes. Accessed through `grid.node_z`.

    Returns
    -------
    face_edges_cartesian : np.ndarray
        An array of shape (n_face, n_max_face_edges, 2, 3) containing the Cartesian coordinates of the edges
        for each face. It might contain dummy values if the grid has holes.

    Examples
    --------
    >>> face_node_conn = np.array(
    ...     [
    ...         [0, 1, 2, 3, 4],
    ...         [0, 1, 3, 4, INT_FILL_VALUE],
    ...         [0, 1, 3, INT_FILL_VALUE, INT_FILL_VALUE],
    ...     ]
    ... )
    >>> n_face = 3
    >>> n_max_face_edges = 5
    >>> node_x = np.array([0, 1, 1, 0, 1, 0])
    >>> node_y = np.array([0, 0, 1, 1, 2, 2])
    >>> node_z = np.array([0, 0, 0, 0, 1, 1])
    >>> _get_cartesian_face_edge_nodes(
    ...     face_node_conn, n_face, n_max_face_edges, node_x, node_y, node_z
    ... )
    array([[[[    0,     0,     0],
         [    1,     0,     0]],

        [[    1,     0,     0],
         [    1,     1,     0]],

        [[    1,     1,     0],
         [    0,     1,     0]],

        [[    0,     1,     0],
         [    1,     2,     1]],

        [[    1,     2,     1],
         [    0,     0,     0]]],


       [[[    0,     0,     0],
         [    1,     0,     0]],

        [[    1,     0,     0],
         [    0,     1,     0]],

        [[    0,     1,     0],
         [    1,     2,     1]],

        [[    1,     2,     1],
         [    0,     0,     0]],

        [[INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE],
        [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE]]],


       [[[    0,     0,     0],
         [    1,     0,     0]],

        [[    1,     0,     0],
         [    0,     1,     0]],

        [[    0,     1,     0],
         [    0,     0,     0]],

        [[INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE],
         [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE]],

        [[INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE],
         [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE]]]])
    """

    # face_edge_connectivity (n_face, n_edge)

    # each edge should have a shape (2, 3)

    # Shift node connections to create edge connections
    face_node_conn_shift = np.roll(face_node_conn, -1, axis=1)

    # Construct edge connections by combining original and shifted node connections
    face_edge_conn = np.array([face_node_conn, face_node_conn_shift]).T.swapaxes(0, 1)

    # swap the first occurrence of INT_FILL_VALUE with the last value in each sub-array
    face_edge_conn = _swap_first_fill_value_with_last(face_edge_conn)

    # Get the indices of the nodes from face_edge_conn
    face_edge_conn_flat = face_edge_conn.reshape(-1)

    valid_mask = face_edge_conn_flat != INT_FILL_VALUE

    # Get the valid node indices
    valid_edges = face_edge_conn_flat[valid_mask]

    #  Create an array to hold the Cartesian coordinates of the edges
    face_edges_cartesian = np.full(
        (len(face_edge_conn_flat), 3), INT_FILL_VALUE, dtype=float
    )

    # Fill the array with the Cartesian coordinates of the edges
    face_edges_cartesian[valid_mask, 0] = node_x[valid_edges]
    face_edges_cartesian[valid_mask, 1] = node_y[valid_edges]
    face_edges_cartesian[valid_mask, 2] = node_z[valid_edges]

    return face_edges_cartesian.reshape(n_face, n_max_face_edges, 2, 3)


def _construct_face_edge_nodes_spherical(
    face_node_conn, n_face, n_max_face_edges, node_lon, node_lat
):
    """Construct an array to hold the edge latitude and longitude in radians
    connectivity for multiple faces in a grid.

    Parameters
    ----------
    face_node_conn : np.ndarray
        An array of shape (n_face, n_max_face_edges) containing the node indices for each face. Accessed through `grid.face_node_connectivity.value`.
    n_face : int
        The number of faces in the grid. Accessed through `grid.n_face`.
    n_max_face_edges : int
        The maximum number of edges for any face in the grid. Accessed through `grid.n_max_face_edges`.
    node_lon : np.ndarray
        An array of shape (n_nodes,) containing the longitude values of the nodes in degrees. Accessed through `grid.node_lon`.
    node_lat : np.ndarray
        An array of shape (n_nodes,) containing the latitude values of the nodes in degrees. Accessed through `grid.node_lat`.

    Returns
    -------
    face_edges_lonlat_rad : np.ndarray
        An array of shape (n_face, n_max_face_edges, 2, 2) containing the latitude and longitude coordinates
        in radians for the edges of each face. It might contain dummy values if the grid has holes.

    Notes
    -----
    If the grid has holes, the function will return an entry of dummy value faces_edges_coordinates[i] filled with INT_FILL_VALUE.
    """

    # Convert node coordinates to radians
    node_lon_rad = np.deg2rad(node_lon)
    node_lat_rad = np.deg2rad(node_lat)

    # Shift node connections to create edge connections
    face_node_conn_shift = np.roll(face_node_conn, -1, axis=1)

    # Construct edge connections by combining original and shifted node connections
    face_edge_conn = np.array([face_node_conn, face_node_conn_shift]).T.swapaxes(0, 1)

    # swap the first occurrence of INT_FILL_VALUE with the last value in each sub-array
    face_edge_conn = _swap_first_fill_value_with_last(face_edge_conn)

    # Get the indices of the nodes from face_edge_conn
    face_edge_conn_flat = face_edge_conn.reshape(-1)

    valid_mask = face_edge_conn_flat != INT_FILL_VALUE

    # Get the valid node indices
    valid_edges = face_edge_conn_flat[valid_mask]

    # Create an array to hold the latitude and longitude in radians for the edges
    face_edges_lonlat_rad = np.full(
        (len(face_edge_conn_flat), 2), INT_FILL_VALUE, dtype=float
    )

    # Fill the array with the latitude and longitude in radians for the edges
    face_edges_lonlat_rad[valid_mask, 0] = node_lon_rad[valid_edges]
    face_edges_lonlat_rad[valid_mask, 1] = node_lat_rad[valid_edges]

    return face_edges_lonlat_rad.reshape(n_face, n_max_face_edges, 2, 2)
