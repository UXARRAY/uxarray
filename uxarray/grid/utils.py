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


# TODO: Consider re-implementation in the future / better integration with API
def _get_cartesian_face_edge_nodes(
    face_nodes_ind, face_edges_ind, edge_nodes_grid, node_x, node_y, node_z
):
    """Construct an array to hold the edge Cartesian coordinates connectivity
    for a face in a grid.

    Parameters
    ----------
    face_nodes_ind : np.ndarray, shape (n_nodes,)
        The ith entry of Grid.face_node_connectivity, where n_nodes is the number of nodes in the face.
    face_edges_ind : np.ndarray, shape (n_edges,)
        The ith entry of Grid.face_edge_connectivity, where n_edges is the number of edges in the face.
    edge_nodes_grid : np.ndarray, shape (n_edges, 2)
        The entire Grid.edge_node_connectivity, where n_edges is the total number of edges in the grid.
    node_x : np.ndarray, shape (n_nodes,)
        The values of Grid.node_x, where n_nodes_total is the total number of nodes in the grid.
    node_y : np.ndarray, shape (n_nodes,)
        The values of Grid.node_y.
    node_z : np.ndarray, shape (n_nodes,)
        The values of Grid.node_z.

    Returns
    -------
    face_edges : np.ndarray, shape (n_edges, 2, 3)
        Face edge connectivity in Cartesian coordinates, where n_edges is the number of edges for the specific face.

    Notes
    -----
    - The function assumes that the inputs are well-formed and correspond to the same face.
    - The output array contains the Cartesian coordinates for each edge of the face.
    """

    # Create a mask that is True for all values not equal to INT_FILL_VALUE
    mask = face_edges_ind != INT_FILL_VALUE

    # Use the mask to select only the elements not equal to INT_FILL_VALUE
    valid_edges = face_edges_ind[mask]
    face_edges = edge_nodes_grid[valid_edges]

    # Ensure counter-clockwise order of edge nodes
    # Start with the first two nodes
    face_edges[0] = [face_nodes_ind[0], face_nodes_ind[1]]

    for idx in range(1, len(face_edges)):
        if face_edges[idx][0] != face_edges[idx - 1][1]:
            # Swap the node index in this edge if not in counter-clockwise order
            face_edges[idx] = face_edges[idx][::-1]

    # Fetch coordinates for each node in the face edges
    cartesian_coordinates = np.array(
        [
            [[node_x[node], node_y[node], node_z[node]] for node in edge]
            for edge in face_edges
        ]
    )

    return cartesian_coordinates


def _get_lonlat_rad_face_edge_nodes(
    face_nodes_ind, face_edges_ind, edge_nodes_grid, node_lon, node_lat
):
    """Construct an array to hold the edge lat lon in radian connectivity for a
    face in a grid.

    Parameters
    ----------
    face_nodes_ind : np.ndarray, shape (n_nodes,)
        The ith entry of Grid.face_node_connectivity, where n_nodes is the number of nodes in the face.
    face_edges_ind : np.ndarray, shape (n_edges,)
        The ith entry of Grid.face_edge_connectivity, where n_edges is the number of edges in the face.
    edge_nodes_grid : np.ndarray, shape (n_edges, 2)
        The entire Grid.edge_node_connectivity, where n_edges is the total number of edges in the grid.
    node_lon : np.ndarray, shape (n_nodes,)
        The values of Grid.node_lon.
    node_lat : np.ndarray, shape (n_nodes,)
        The values of Grid.node_lat, where n_nodes_total is the total number of nodes in the grid.
    Returns
    -------
    face_edges : np.ndarray, shape (n_edges, 2, 3)
        Face edge connectivity in Cartesian coordinates, where n_edges is the number of edges for the specific face.

    Notes
    -----
    - The function assumes that the inputs are well-formed and correspond to the same face.
    - The output array contains the Latitude and longitude coordinates in radian for each edge of the face.
    """

    # Create a mask that is True for all values not equal to INT_FILL_VALUE
    mask = face_edges_ind != INT_FILL_VALUE

    # Use the mask to select only the elements not equal to INT_FILL_VALUE
    valid_edges = face_edges_ind[mask]
    face_edges = edge_nodes_grid[valid_edges]

    # Ensure counter-clockwise order of edge nodes
    # Start with the first two nodes
    face_edges[0] = [face_nodes_ind[0], face_nodes_ind[1]]

    for idx in range(1, len(face_edges)):
        if face_edges[idx][0] != face_edges[idx - 1][1]:
            # Swap the node index in this edge if not in counter-clockwise order
            face_edges[idx] = face_edges[idx][::-1]

    # Fetch coordinates for each node in the face edges
    lonlat_coordinates = np.array(
        [
            [
                [
                    np.mod(np.deg2rad(node_lon[node]), 2 * np.pi),
                    np.deg2rad(node_lat[node]),
                ]
                for node in edge
            ]
            for edge in face_edges
        ]
    )

    return lonlat_coordinates
