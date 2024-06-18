import numpy as np
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
import warnings
import uxarray.utils.computing as ac_utils


def _replace_fill_values(grid_var, original_fill, new_fill, new_dtype=None):
    """Replaces all instances of the current fill value (``original_fill``) in
    (``grid_var``) with (``new_fill``) and converts to the dtype defined by
    (``new_dtype``)

    Parameters
    ----------
    grid_var : np.ndarray
        grid variable to be modified
    original_fill : constant
        original fill value used in (``grid_var``)
    new_fill : constant
        new fill value to be used in (``grid_var``)
    new_dtype : np.dtype, optional
        new data type to convert (``grid_var``) to

    Returns
    ----------
    grid_var : xarray.Dataset
        Input Dataset with correct fill value and dtype
    """

    # locations of fill values
    if original_fill is not None and np.isnan(original_fill):
        fill_val_idx = np.isnan(grid_var)
    else:
        fill_val_idx = grid_var == original_fill

    # convert to new data type
    if new_dtype != grid_var.dtype and new_dtype is not None:
        grid_var = grid_var.astype(new_dtype)

    # ensure fill value can be represented with current integer data type
    if np.issubdtype(new_dtype, np.integer):
        int_min = np.iinfo(grid_var.dtype).min
        int_max = np.iinfo(grid_var.dtype).max
        # ensure new_fill is in range [int_min, int_max]
        if new_fill < int_min or new_fill > int_max:
            raise ValueError(
                f"New fill value: {new_fill} not representable by"
                f" integer dtype: {grid_var.dtype}"
            )

    # ensure non-nan fill value can be represented with current float data type
    elif np.issubdtype(new_dtype, np.floating) and not np.isnan(new_fill):
        float_min = np.finfo(grid_var.dtype).min
        float_max = np.finfo(grid_var.dtype).max
        # ensure new_fill is in range [float_min, float_max]
        if new_fill < float_min or new_fill > float_max:
            raise ValueError(
                f"New fill value: {new_fill} not representable by"
                f" float dtype: {grid_var.dtype}"
            )
    else:
        raise ValueError(
            f"Data type {grid_var.dtype} not supported" f"for grid variables"
        )

    # replace all zeros with a fill value
    grid_var[fill_val_idx] = new_fill

    return grid_var


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
    try:
        inverse_jacobian = np.linalg.inv(jacobian)
    except np.linalg.LinAlgError:
        raise warnings("Warning: Singular Jacobian matrix encountered.")
        return None

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
    tolerance = ERROR_TOLERANCE
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

        y_new = y_guess - np.matmul(j_inv, f_vector)
        error = np.max(np.abs(y_guess - y_new))
        y_guess = y_new

        if verbose:
            print(f"Newton method iter: {_iter}, error: {error}")
        _iter += 1

    return np.append(y_new, constZ)


def _get_cartesian_face_edge_nodes(
    face_node_conn, n_nodes_per_face, n_face, n_max_face_edges, node_x, node_y, node_z
):
    """Construct an array to hold the edge Cartesian coordinates connectivity
    for multiple faces in a grid.

    Parameters
    ----------
    face_node_conn : np.ndarray
        An array of shape (n_face, n_max_face_edges) containing the node indices for each face.
    n_nodes_per_face : np.ndarray
        An array of shape (n_face,) indicating the number of nodes for each face.
    n_face : int
        The number of faces in the grid.
    n_max_face_edges : int
        The maximum number of edges for any face in the grid.
    node_x : np.ndarray
        An array of shape (n_nodes,) containing the x-coordinate values of the nodes.
    node_y : np.ndarray
        An array of shape (n_nodes,) containing the y-coordinate values of the nodes.
    node_z : np.ndarray
        An array of shape (n_nodes,) containing the z-coordinate values of the nodes.

    Returns
    -------
    face_edges_cartesian : np.ndarray
        An array of shape (n_face, n_max_face_edges, 2, 3) containing the Cartesian coordinates of the edges
        for each face. It might contain dummy values if the grid has holes.
    """
    # Shift node connections to create edge connections
    face_node_conn_shift = np.roll(face_node_conn, -1, axis=1)

    # Close the loop for each face by connecting the last node to the first node
    for i, final_node_idx in enumerate(n_nodes_per_face):
        face_node_conn_shift[i, final_node_idx - 1] = face_node_conn[i, 0]

    # Construct edge connections by combining original and shifted node connections
    face_edge_conn = np.array([face_node_conn, face_node_conn_shift]).T.swapaxes(0, 1)

    # Reshape edge connections and create a mask for valid edges
    face_edge_conn_ravel = face_edge_conn.reshape((n_face * n_max_face_edges, 2))
    non_fill_value_mask = face_edge_conn_ravel[:, 0] != INT_FILL_VALUE

    # Extract Cartesian coordinates for the edge nodes using the mask
    edge_node_x_a = node_x[face_edge_conn_ravel[:, 0][non_fill_value_mask]]
    edge_node_y_a = node_y[face_edge_conn_ravel[:, 0][non_fill_value_mask]]
    edge_node_z_a = node_z[face_edge_conn_ravel[:, 0][non_fill_value_mask]]
    edge_node_x_b = node_x[face_edge_conn_ravel[:, 1][non_fill_value_mask]]
    edge_node_y_b = node_y[face_edge_conn_ravel[:, 1][non_fill_value_mask]]
    edge_node_z_b = node_z[face_edge_conn_ravel[:, 1][non_fill_value_mask]]

    # Initialize the final array and assign coordinates for valid edges
    face_edges_cartesian = np.full((n_face * n_max_face_edges, 2, 3), np.nan)
    face_edges_cartesian[:, 0][non_fill_value_mask] = np.vstack(
        [edge_node_x_a, edge_node_y_a, edge_node_z_a]
    ).T
    face_edges_cartesian[:, 1][non_fill_value_mask] = np.vstack(
        [edge_node_x_b, edge_node_y_b, edge_node_z_b]
    ).T

    # Reshape the final array to the desired shape
    face_edges_cartesian = face_edges_cartesian.reshape(
        (n_face, n_max_face_edges, 2, 3)
    )

    return face_edges_cartesian


def _get_lonlat_rad_face_edge_nodes(
    face_node_conn, n_nodes_per_face, n_face, n_max_face_edges, node_lon, node_lat
):
    """Construct an array to hold the edge latitude and longitude in radians
    connectivity for multiple faces in a grid.

    Parameters
    ----------
    face_node_conn : np.ndarray
        An array of shape (n_face, n_max_face_edges) containing the node indices for each face.
    n_nodes_per_face : np.ndarray
        An array of shape (n_face,) indicating the number of nodes for each face.
    n_face : int
        The number of faces in the grid.
    n_max_face_edges : int
        The maximum number of edges for any face in the grid.
    node_lon : np.ndarray
        An array of shape (n_nodes,) containing the longitude values of the nodes in degrees.
    node_lat : np.ndarray
        An array of shape (n_nodes,) containing the latitude values of the nodes in degrees.

    Returns
    -------
    face_edges_lonlat_rad : np.ndarray
        An array of shape (n_face, n_max_face_edges, 2, 2) containing the latitude and longitude coordinates
        in radians for the edges of each face. It might contain dummy values if the grid has holes.
    """
    # Shift node connections to create edge connections
    face_node_conn_shift = np.roll(face_node_conn, -1, axis=1)

    # Close the loop for each face by connecting the last node to the first node
    for i, final_node_idx in enumerate(n_nodes_per_face):
        face_node_conn_shift[i, final_node_idx - 1] = face_node_conn[i, 0]

    # Construct edge connections by combining original and shifted node connections
    face_edge_conn = np.array([face_node_conn, face_node_conn_shift]).T.swapaxes(0, 1)

    # Reshape edge connections and create a mask for valid edges
    face_edge_conn_ravel = face_edge_conn.reshape((n_face * n_max_face_edges, 2))
    non_fill_value_mask = face_edge_conn_ravel[:, 0] != INT_FILL_VALUE

    # Convert node coordinates to radians
    nod_lon_rad = np.deg2rad(node_lon)
    node_lat_rad = np.deg2rad(node_lat)

    # Extract longitude and latitude for the edge nodes using the mask
    edge_node_lon_a = nod_lon_rad[face_edge_conn_ravel[:, 0][non_fill_value_mask]]
    edge_node_lat_a = node_lat_rad[face_edge_conn_ravel[:, 0][non_fill_value_mask]]
    edge_node_lon_b = nod_lon_rad[face_edge_conn_ravel[:, 1][non_fill_value_mask]]
    edge_node_lat_b = node_lat_rad[face_edge_conn_ravel[:, 1][non_fill_value_mask]]

    # Initialize the final array and assign coordinates for valid edges
    face_edges_lonlat_rad = np.full((n_face * n_max_face_edges, 2, 2), np.nan)
    face_edges_lonlat_rad[:, 0][non_fill_value_mask] = np.vstack(
        [edge_node_lon_a, edge_node_lat_a]
    ).T
    face_edges_lonlat_rad[:, 1][non_fill_value_mask] = np.vstack(
        [edge_node_lon_b, edge_node_lat_b]
    ).T

    # Reshape the final array to the desired shape
    face_edges_lonlat_rad = face_edges_lonlat_rad.reshape(
        (n_face, n_max_face_edges, 2, 2)
    )

    return face_edges_lonlat_rad
