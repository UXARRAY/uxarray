import numpy as np
from uxarray.constants import INT_FILL_VALUE, MACHINE_EPSILON
import warnings
import uxarray.utils.computing as ac_utils

from numba import njit


@njit(cache=True)
def _angle_of_2_vectors(u, v):
    """Calculate the angle between two 3D vectors u and v in radians. Can be
    used to calcualte the span of a GCR.

    Parameters
    ----------
    u : numpy.ndarray (float)
        The first 3D vector.
    v : numpy.ndarray (float)
        The second 3D vector.

    Returns
    -------
    float
        The angle between u and v in radians.
    """
    v_norm_times_u = np.linalg.norm(v) * u
    u_norm_times_v = np.linalg.norm(u) * v
    vec_minus = v_norm_times_u - u_norm_times_v
    vec_sum = v_norm_times_u + u_norm_times_v
    angle_u_v_rad = 2 * np.arctan2(np.linalg.norm(vec_minus), np.linalg.norm(vec_sum))
    return angle_u_v_rad


def _inv_jacobian(x0, x1, y0, y1, z0, z1, x_i_old, y_i_old):
    """Calculate the inverse Jacobian matrix for a given set of parameters.

    Parameters
    ----------
    x0 : float
        Description of x0.
    x1 : float
        Description of x1.
    y0 : float
        Description of y0.
    y1 : float
        Description of y1.
    z0 : float
        Description of z0.
    z1 : float
        Description of z1.
    x_i_old : float
        Description of x_i_old.
    y_i_old : float
        Description of y_i_old.

    Returns
    -------
    numpy.ndarray or None
        The inverse Jacobian matrix if it is non-singular, or None if a singular matrix is encountered.

    Notes
    -----
    This function calculates the inverse Jacobian matrix based on the provided parameters. If the Jacobian matrix
    is singular, a warning is printed, and None is returned.
    """

    # d_dx = (x0 * x_i_old - x1 * x_i_old * z0 + y0 * y_i_old * z1 - y1 * y_i_old * z0 - y1 * y_i_old * z0)
    # d_dy = 2 * (x0 * x_i_old * z1 - x1 * x_i_old * z0 + y0 * y_i_old * z1 - y1 * y_i_old * z0)
    #
    # # row 1
    # J[0, 0] = y_i_old / d_dx
    # J[0, 1] = (x0 * z1 - z0 * x1) / d_dy
    # # row 2
    # J[1, 0] = x_i_old / d_dx
    # J[1, 1] = (y0 * z1 - z0 * y1) / d_dy

    # The Jacobian Matrix
    jacobian = [
        [ac_utils._fmms(y0, z1, z0, y1), ac_utils._fmms(x0, z1, z0, x1)],
        [2 * x_i_old, 2 * y_i_old],
    ]

    # First check if the Jacobian matrix is singular
    if np.linalg.matrix_rank(jacobian) < 2:
        warnings.warn("The Jacobian matrix is singular.")
        return None

    try:
        inverse_jacobian = np.linalg.inv(jacobian)
    except np.linalg.LinAlgError as e:
        # Print out the error message

        cond_number = np.linalg.cond(jacobian)
        print(f"Condition number: {cond_number}")
        print(f"Jacobian matrix:\n{jacobian}")
        print(f"An error occurred: {e}")
        raise

    return inverse_jacobian


def _newton_raphson_solver_for_gca_constLat(
    init_cart, gca_cart, max_iter=1000, verbose=False
):
    """Solve for the intersection point between a great circle arc and a
    constant latitude.

    Args:
        init_cart (np.ndarray): Initial guess for the intersection point.
        w0_cart (np.ndarray): First vector defining the great circle arc.
        w1_cart (np.ndarray): Second vector defining the great circle arc.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        np.ndarray or None: The intersection point or None if the solver fails to converge.
    """
    tolerance = MACHINE_EPSILON * 100
    w0_cart, w1_cart = gca_cart
    error = float("inf")
    constZ = init_cart[2]
    y_guess = np.array(init_cart[0:2])
    y_new = y_guess

    _iter = 0

    while error > tolerance and _iter < max_iter:
        f_vector = np.array(
            [
                np.dot(
                    np.cross(w0_cart, w1_cart),
                    np.array([y_guess[0], y_guess[1], constZ]),
                ),
                y_guess[0] * y_guess[0]
                + y_guess[1] * y_guess[1]
                + constZ * constZ
                - 1.0,
            ]
        )

        try:
            j_inv = _inv_jacobian(
                w0_cart[0],
                w1_cart[0],
                w0_cart[1],
                w1_cart[1],
                w0_cart[2],
                w1_cart[2],
                y_guess[0],
                y_guess[1],
            )

            if j_inv is None:
                return None
        except RuntimeError as e:
            print(f"Encountered an error: {e}")
            raise

        y_new = y_guess - np.matmul(j_inv, f_vector)
        error = np.max(np.abs(y_guess - y_new))
        y_guess = y_new

        if verbose:
            print(f"Newton method iter: {_iter}, error: {error}")
        _iter += 1

    return np.append(y_new, constZ)


def _swap_first_fill_value_with_last(arr):
    """Swap the first occurrence of INT_FILL_VALUE in each sub-array with the
    last value in the sub-array.

    Parameters:
    ----------
    arr (np.ndarray): A 3D numpy array where the swap will be performed.

    Returns:
    -------
    np.ndarray: The modified array with the swaps made.
    """
    # Find the indices of the first INT_FILL_VALUE in each sub-array
    mask = arr == INT_FILL_VALUE
    reshaped_mask = mask.reshape(arr.shape[0], -1)
    first_true_indices = np.argmax(reshaped_mask, axis=1)

    # If no INT_FILL_VALUE is found in a row, argmax will return 0, we need to handle this case
    first_true_indices[~np.any(reshaped_mask, axis=1)] = -1

    # Get the shape of the sub-arrays
    subarray_shape = arr.shape[1:]

    # Calculate the 2D indices within each sub-array
    valid_indices = first_true_indices != -1
    first_true_positions = np.unravel_index(
        first_true_indices[valid_indices], subarray_shape
    )

    # Create an index array for the last value in each sub-array
    last_indices = np.full((arr.shape[0],), subarray_shape[0] * subarray_shape[1] - 1)
    last_positions = np.unravel_index(last_indices, subarray_shape)

    # Swap the first INT_FILL_VALUE with the last value in each sub-array
    row_indices = np.arange(arr.shape[0])

    # Advanced indexing to swap values
    (
        arr[
            row_indices[valid_indices], first_true_positions[0], first_true_positions[1]
        ],
        arr[
            row_indices[valid_indices],
            last_positions[0][valid_indices],
            last_positions[1][valid_indices],
        ],
    ) = (
        arr[
            row_indices[valid_indices],
            last_positions[0][valid_indices],
            last_positions[1][valid_indices],
        ],
        arr[
            row_indices[valid_indices], first_true_positions[0], first_true_positions[1]
        ],
    )

    return arr


def _get_cartesian_face_edge_nodes(
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


def _get_lonlat_rad_face_edge_nodes(
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
