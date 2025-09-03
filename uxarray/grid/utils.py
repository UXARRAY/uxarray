import numpy as np
import xarray as xr
from numba import njit

from uxarray.constants import INT_FILL_VALUE


@njit(cache=True)
def _small_angle_of_2_vectors(u, v):
    """
    Compute the smallest angle between two vectors using the new _angle_of_2_vectors.

    Parameters
    ----------
    u : numpy.ndarray
        The first 3D vector.
    v : numpy.ndarray
        The second 3D vector.

    Returns
    -------
    float
        The smallest angle between `u` and `v` in radians.
    """
    v_norm_times_u = np.linalg.norm(v) * u
    u_norm_times_v = np.linalg.norm(u) * v
    vec_minus = v_norm_times_u - u_norm_times_v
    vec_sum = v_norm_times_u + u_norm_times_v
    angle_u_v_rad = 2 * np.arctan2(np.linalg.norm(vec_minus), np.linalg.norm(vec_sum))
    return angle_u_v_rad


@njit(cache=True)
def _angle_of_2_vectors(u, v):
    """
    Calculate the angle between two 3D vectors `u` and `v` on the unit sphere in radians.

    This function computes the angle between two vectors originating from the center of a unit sphere.
    The result is returned in the range [0, 2π]. It can be used to calculate the span of a great circle arc (GCA).

    Parameters
    ----------
    u : numpy.ndarray
        The first 3D vector (float), originating from the center of the unit sphere.
    v : numpy.ndarray
        The second 3D vector (float), originating from the center of the unit sphere.

    Returns
    -------
    float
        The angle between `u` and `v` in radians, in the range [0, 2π].

    Notes
    -----
    - The direction of the angle (clockwise or counter-clockwise) is determined using the cross product of `u` and `v`.
    - Special cases such as vectors aligned along the same longitude are handled explicitly.
    """
    # Compute the cross product to determine the direction of the normal
    normal = np.cross(u, v)

    # Calculate the angle using arctangent of cross and dot products
    angle_u_v_rad = np.arctan2(np.linalg.norm(normal), np.dot(u, v))

    # Determine the direction of the angle
    normal_z = np.dot(normal, np.array([0.0, 0.0, 1.0]))
    if normal_z > 0:
        # Counterclockwise direction
        return angle_u_v_rad
    elif normal_z == 0:
        # Handle collinear vectors (same longitude)
        if u[2] > v[2]:
            return angle_u_v_rad
        elif u[2] < v[2]:
            return 2 * np.pi - angle_u_v_rad
        else:
            return 0.0  # u == v
    else:
        # Clockwise direction
        return 2 * np.pi - angle_u_v_rad


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


def _get_cartesian_face_edge_nodes_array(
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
    >>> _get_cartesian_face_edge_nodes_array(
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


def _get_lonlat_rad_face_edge_nodes_array(
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


@njit(cache=True)
def _get_cartesian_face_edge_nodes(
    face_idx, face_node_connectivity, n_edges_per_face, node_x, node_y, node_z
):
    """Computes the Cartesian Coordinates of the edge nodes that make up a given face.

    Parameters
    ----------
    face_idx : int
        The index of the face to construct the edge nodes
    face_node_connectivity : np.ndarray
        Face Node Connectivity array
    n_edges_per_face : np.ndarray
        Number of non-fill-value edges for each face
    node_x : np.ndarray
        Cartesian x coordinates
    node_y : np.ndarray
        Cartesian y coordinates
    node_z : np.ndarray
        Cartesian z coordinates

    Returns
    -------
    face_edge_nodes: np.ndarray
        Cartesian coordinates of the edge nodes that make up a given face
    """
    # Number non-fill-value edges
    n_edges = n_edges_per_face[face_idx]

    # Allocate data for face_edge_nodes
    face_edge_nodes = np.empty((n_edges, 2, 3), dtype=np.float64)

    start_nodes = face_node_connectivity[face_idx, 0:n_edges]
    end_nodes = np.roll(start_nodes, -1)

    # Assign x coordinates of start and end nodes
    face_edge_nodes[0:n_edges, 0, 0] = node_x[start_nodes]
    face_edge_nodes[0:n_edges, 1, 0] = node_x[end_nodes]

    # Assign y coordinates of start and end nodes
    face_edge_nodes[0:n_edges, 0, 1] = node_y[start_nodes]
    face_edge_nodes[0:n_edges, 1, 1] = node_y[end_nodes]

    # Assign z coordinates of start and end nodes
    face_edge_nodes[0:n_edges, 0, 2] = node_z[start_nodes]
    face_edge_nodes[0:n_edges, 1, 2] = node_z[end_nodes]

    return face_edge_nodes


@njit(cache=True)
def _get_spherical_face_edge_nodes(
    face_idx, face_node_connectivity, n_edges_per_face, node_lon, node_lat
):
    """Computes the Spherical Coordinates of the edge nodes that make up a given face.

    Parameters
    ----------
    face_idx : int
        The index of the face to construct the edge nodes
    face_node_connectivity : np.ndarray
        Face Node Connectivity array
    n_edges_per_face : np.ndarray
        Number of non-fill-value edges for each face
    node_lon : np.ndarray
        Longitude coordinates
    node_lat : np.ndarray
        Latitude coordinates

    Returns
    -------
    face_edge_nodes: np.ndarray
        Spherical coordinates of the edge nodes that make up a given face
    """
    # Number non-fill-value edges
    n_edges = n_edges_per_face[face_idx]

    # Allocate data for face_edge_nodes
    face_edge_nodes = np.empty((n_edges, 2, 2), dtype=np.float64)

    start_nodes = face_node_connectivity[face_idx, 0:n_edges]
    end_nodes = np.roll(start_nodes, -1)

    # Assign longitude coordinates of start and end nodes
    face_edge_nodes[0:n_edges, 0, 0] = node_lon[start_nodes]
    face_edge_nodes[0:n_edges, 1, 0] = node_lon[end_nodes]

    # Assign latitude coordinates of start and end nodes
    face_edge_nodes[0:n_edges, 0, 1] = node_lat[start_nodes]
    face_edge_nodes[0:n_edges, 1, 1] = node_lat[end_nodes]

    return face_edge_nodes


@njit(cache=True)
def all_elements_nan(arr):
    """Check if all elements in an array are np.nan."""
    for i in range(arr.shape[0]):
        if not np.isnan(arr[i]):
            return False
    return True


@njit(cache=True)
def any_close_lat(lat_pt, atol):
    """Check if the latitude point is close to either the North or South Pole."""
    return np.isclose(lat_pt, 0.5 * np.pi, atol) or np.isclose(
        lat_pt, -0.5 * np.pi, atol
    )


def make_setter(key: str):
    """Return a setter that assigns the value to self._ds[key] after type-checking."""

    def setter(self, value):
        if not isinstance(value, xr.DataArray):
            raise ValueError(f"{key} must be an xr.DataArray")
        self._ds[key] = value

    return setter
