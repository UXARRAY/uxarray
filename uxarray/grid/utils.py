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


def _get_cartesian_single_face_edge_nodes(
    face_nodes, node_x_sliced, node_y_sliced, node_z_sliced
):
    """Construct an array to hold the edge Cartesian coordinates connectivity
    for a single face in a grid.

    Parameters
    ----------
    face_nodes : np.ndarray, shape (n_nodes_sliced,)
        The node indices for the face.
    node_x_sliced : np.ndarray, shape (n_nodes_sliced,)
        The values of Grid.node_x for the sliced portion.
    node_y_sliced : np.ndarray, shape (n_nodes_sliced,)
        The values of Grid.node_y for the sliced portion.
    node_z_sliced : np.ndarray, shape (n_nodes_sliced,)
        The values of Grid.node_z for the sliced portion.

    Returns
    -------
    cartesian_coordinates : np.ndarray
        An array of shape (n_edges_sliced, 2, 3) containing the Cartesian coordinates of the edges for the face.
    """

    # Create a mask that is True for all values not equal to INT_FILL_VALUE
    mask = face_nodes != INT_FILL_VALUE

    # Use the mask to select only the elements not equal to INT_FILL_VALUE
    face_nodes = face_nodes[mask]
    face_edges_connectivity = face_edges_connectivity = np.zeros(
        (len(face_nodes), 2), dtype=int
    )

    # Initialize the first edge
    face_edges_connectivity[0] = [face_nodes[0], face_nodes[1]]

    # Do the vectorized check for counter-clockwise order of edge nodes
    face_edges_connectivity[:, 0] = face_nodes
    face_edges_connectivity[:, 1] = np.roll(face_nodes, -1)

    # Ensure the last edge connects back to the first node to complete the loop
    face_edges_connectivity[-1] = [face_nodes[-1], face_nodes[0]]

    # Fetch coordinates for each node in the face edges
    nodes = face_edges_connectivity.flatten()
    coordinates = np.column_stack(
        (node_x_sliced[nodes], node_y_sliced[nodes], node_z_sliced[nodes])
    )
    cartesian_coordinates = coordinates.reshape(-1, 2, 3)

    return cartesian_coordinates


def _get_cartesian_face_edge_nodes(
    face_nodes_sliced, node_x_sliced, node_y_sliced, node_z_sliced
):
    """Construct an array to hold the edge Cartesian coordinates connectivity
    for multiple faces in a grid.

    This function processes sliced portions of the total grid data. Users must prepare the sliced versions of the data according to their needs before using this function.

    Parameters
    ----------
    face_nodes_sliced : list of np.ndarray
        Each element is an array of shape (n_nodes_sliced,) corresponding to the sliced face's node indices.
    node_x_sliced : np.ndarray, shape (n_nodes_sliced,)
        The values of Grid.node_x for the sliced portion, where n_nodes_sliced is the total number of nodes in the sliced portion.
    node_y_sliced : np.ndarray, shape (n_nodes_sliced,)
        The values of Grid.node_y for the sliced portion.
    node_z_sliced : np.ndarray, shape (n_nodes_sliced,)
        The values of Grid.node_z for the sliced portion.

    Returns
    -------
    faces_edges_coordinates : np.ndarray
        An array of shape (n_faces, n_edges, 2, 3) containing the Cartesian coordinates
        of the edges for each face.
    """

    # Use map function to apply the single face function to all faces
    faces_edges_coordinates = list(
        map(
            lambda face_nodes: _get_cartesian_single_face_edge_nodes(
                face_nodes, node_x_sliced, node_y_sliced, node_z_sliced
            ),
            face_nodes_sliced,
        )
    )

    return np.array(faces_edges_coordinates)


def _get_lonlat_rad_single_face_edge_nodes(
    face_nodes, node_lon_sliced, node_lat_sliced
):
    """Construct an array to hold the edge lat lon in radian connectivity for a
    single face in a grid.

    Parameters
    ----------
    face_nodes : np.ndarray, shape (n_nodes,)
        The node indices for the face.
    node_lon_sliced : np.ndarray, shape (n_nodes_sliced,)
        The values of Grid.node_lon, for the sliced portion.
    node_lat_sliced : np.ndarray, shape (n_nodes_sliced,)
        The values of Grid.node_lat, for the sliced portion.

    Returns
    -------
    lonlat_coordinates : np.ndarray, shape (n_edges, 2, 2)
        Face edge connectivity in latitude and longitude coordinates in radians.
    """

    # Create a mask that is True for all values not equal to INT_FILL_VALUE
    mask = face_nodes != INT_FILL_VALUE

    # Use the mask to select only the elements not equal to INT_FILL_VALUE
    face_nodes = face_nodes[mask]
    face_edges_connectivity = face_edges_connectivity = np.zeros(
        (len(face_nodes), 2), dtype=int
    )

    # Initialize the first edge
    face_edges_connectivity[0] = [face_nodes[0], face_nodes[1]]

    # Do the vectorized check for counter-clockwise order of edge nodes
    face_edges_connectivity[:, 0] = face_nodes
    face_edges_connectivity[:, 1] = np.roll(face_nodes, -1)

    # Ensure the last edge connects back to the first node to complete the loop
    face_edges_connectivity[-1] = [face_nodes[-1], face_nodes[0]]

    # Fetch coordinates for each node in the face edges
    nodes = face_edges_connectivity.flatten()
    lonlat_coordinates = np.column_stack(
        (
            np.mod(np.deg2rad(node_lon_sliced[nodes]), 2 * np.pi),
            np.deg2rad(node_lat_sliced[nodes]),
        )
    ).reshape(-1, 2, 2)

    return lonlat_coordinates


def _get_lonlat_rad_face_edge_nodes(
    face_nodes_sliced, node_lon_sliced, node_lat_sliced
):
    """Construct an array to hold the edge lat lon in radian connectivity for
    multiple faces in a grid.

    This function processes sliced portions of the total grid data. Users must prepare the sliced versions of the data according to their needs before using this function.

    Parameters
    ----------
    face_nodes_sliced : list of np.ndarray
        Each element is an array of shape (n_nodes_sliced,) corresponding to the sliced face's node indices.
    node_lon_sliced : np.ndarray, shape (n_nodes_sliced,)
        The values of Grid.node_lon for the sliced portion, where n_nodes_sliced is the total number of nodes in the sliced portion.
    node_lat_sliced : np.ndarray, shape (n_nodes_sliced,)
        The values of Grid.node_lat for the sliced portion.

    Returns
    -------
    faces_lonlat_coordinates : np.ndarray
        An array of shape (n_faces, n_edges, 2, 2) containing the latitude and longitude coordinates
        in radians for the edges of each face.
    """

    # Use map function to apply the single face function to all faces
    faces_lonlat_coordinates = list(
        map(
            lambda face_nodes: _get_lonlat_rad_single_face_edge_nodes(
                face_nodes, node_lon_sliced, node_lat_sliced
            ),
            face_nodes_sliced,
        )
    )

    return np.array(faces_lonlat_coordinates)