import numpy as np
import xarray as xr
from numba import njit, prange

from uxarray.constants import (
    INT_DTYPE,
    INT_FILL_VALUE,
    MIN_ADAPTIVE_SORT_SIZE,
    MAX_SHIFTS_PER_EDGE,
)
from uxarray.utils.numba_math import (
    _numba_add3,
    _numba_mul3_scalar,
    _numba_norm3,
    _numba_sub3,
)


@njit(cache=True)
def _small_angle_of_2_vectors(u, v):
    """
    Compute the smallest angle between two vectors using the new _angle_of_2_vectors.

    Parameters
    ----------
    u : iterable of length 3
        The first 3D vector.
    v : iterable of length 3
        The second 3D vector.

    Returns
    -------
    float
        The smallest angle between `u` and `v` in radians.
    """
    u_times_v_norm = _numba_mul3_scalar(v, _numba_norm3(u))
    v_times_u_norm = _numba_mul3_scalar(u, _numba_norm3(v))
    vec_minus = _numba_sub3(u_times_v_norm, v_times_u_norm)
    vec_sum = _numba_add3(u_times_v_norm, v_times_u_norm)
    angle_u_v_rad = 2 * np.arctan2(_numba_norm3(vec_minus), _numba_norm3(vec_sum))
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


@njit(cache=True, parallel=True)
def _get_cartesian_face_edge_nodes_array_subset(
    face_indices,
    face_node_connectivity,
    n_nodes_per_face,
    n_max_face_edges,
    node_x,
    node_y,
    node_z,
):
    """Build the Cartesian edge array for a subset of faces.

    Parameters
    ----------
    face_indices : np.ndarray
        1D array of face indices to build, shape ``(n_selected,)``.
    face_node_connectivity : np.ndarray
        Face-node connectivity, shape ``(n_face, n_max_face_edges)``.
    n_nodes_per_face : np.ndarray
        Number of non-fill nodes (== edges) for each face, shape ``(n_face,)``.
    n_max_face_edges : int
        Maximum number of edges for any face in the grid.
    node_x, node_y, node_z : np.ndarray
        Cartesian node coordinates, each shape ``(n_node,)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_selected, n_max_face_edges, 2, 3)``. Unused edge
        slots are padded with ``INT_FILL_VALUE`` cast to float
    """
    n_selected = face_indices.shape[0]
    out = np.full(
        (n_selected, n_max_face_edges, 2, 3), INT_FILL_VALUE, dtype=np.float64
    )

    for i in prange(n_selected):
        f = face_indices[i]
        n_edges = n_nodes_per_face[f]
        for e in range(n_edges):
            n_start = face_node_connectivity[f, e]
            # Wrap the last edge back to the first node to close the polygon,
            # mirroring the np.roll(-1) used by the whole-grid builder.
            n_end = face_node_connectivity[f, (e + 1) % n_edges]
            out[i, e, 0, 0] = node_x[n_start]
            out[i, e, 0, 1] = node_y[n_start]
            out[i, e, 0, 2] = node_z[n_start]
            out[i, e, 1, 0] = node_x[n_end]
            out[i, e, 1, 1] = node_y[n_end]
            out[i, e, 1, 2] = node_z[n_end]

    return out


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
            raise TypeError(f"{key} must be an xr.DataArray")
        self._ds[key] = value

    return setter


# The following adaptive sorting algorithm is implemented for the sole purpose of
# sorting edge_node connectivity quickly. Other usages are not recommended.
#
# ---------------------------------------------------------------------------------
#
# Bucket sorting and searching for the counting sorts in ``uxarray.grid.connectivity``, which
# bucket half edges or edges on one of their two nodes and then order each bucket by the other.
# Nothing below knows about meshes: a bucket is a contiguous ``[bucket_start, bucket_end)`` slice
# of a key array, carrying an equally long ``payload`` array that every reordering moves in step
# so the two stay aligned.
#
# ``_sort_bucket`` orders one bucket and is the entry point for the sorts; the kernels beneath it
# are chosen by bucket size and are exposed only for testing. ``_build_pair_index`` runs the whole
# count/scatter/sort sequence for a caller starting from an ``(n, 2)`` array, and
# ``_search_bucket`` is the lookup that index is built for.
#
# NOTE: these are inlined into ``cache=True`` kernels in another module, and numba stamps its
# cache against the defining file alone, so editing them does not invalidate a caller's cached
# object. Clear ``uxarray/grid/__pycache__/*.nbi *.nbc`` after changing anything here.

# Smallest bucket worth watching for pathological input. A bucket of ``size`` holds at most
# ``size * (size - 1) / 2`` inversions, so at or below this size it cannot exceed the shift
# budget below and the bookkeeping would never pay for itself
MIN_ADAPTIVE_SORT_SIZE = 16

# Shifts per edge an insertion sort may spend on a bucket before it is abandoned for a heap
# sort. Insertion sort costs ``O(size + shifts)``, so a constant budget per edge keeps the
# adaptive path linear while leaving ample room for the near-sorted input it is chosen for
MAX_SHIFTS_PER_EDGE = 8


@njit(cache=True)
def _sort_bucket(end_node, payload, bucket_start, size):
    """Orders one bucket by ``end_node``, picking the sort that suits its size."""
    if size > MIN_ADAPTIVE_SORT_SIZE:
        # Large enough that a bad ordering would be worth catching, which only a
        # collapsed pole or a similarly degenerate node reaches
        _adaptive_sort_bucket(end_node, payload, bucket_start, size)
    elif size > 1:
        _insertion_sort_bucket(end_node, payload, bucket_start, size)


@njit(cache=True)
def _sift_down(end_node, payload, bucket_start, root, size):
    """Restores the max-heap property at ``root`` for a bucket keyed on ``end_node``."""
    while True:
        child = 2 * root + 1
        if child >= size:
            break

        if (
            child + 1 < size
            and end_node[bucket_start + child] < end_node[bucket_start + child + 1]
        ):
            child += 1

        if end_node[bucket_start + root] >= end_node[bucket_start + child]:
            break

        end_node[bucket_start + root], end_node[bucket_start + child] = (
            end_node[bucket_start + child],
            end_node[bucket_start + root],
        )
        payload[bucket_start + root], payload[bucket_start + child] = (
            payload[bucket_start + child],
            payload[bucket_start + root],
        )
        root = child


@njit(cache=True)
def _heap_sort_bucket(end_node, payload, bucket_start, size):
    """Sorts a bucket by ``end_node`` in place, in ``O(size * log(size))`` and without
    scratch space, for the rare bucket an insertion sort cannot finish cheaply."""
    for root in range(size // 2 - 1, -1, -1):
        _sift_down(end_node, payload, bucket_start, root, size)

    for end in range(size - 1, 0, -1):
        end_node[bucket_start], end_node[bucket_start + end] = (
            end_node[bucket_start + end],
            end_node[bucket_start],
        )
        payload[bucket_start], payload[bucket_start + end] = (
            payload[bucket_start + end],
            payload[bucket_start],
        )
        _sift_down(end_node, payload, bucket_start, 0, end)


@njit(cache=True)
def _insertion_sort_bucket(end_node, payload, bucket_start, size):
    """Sorts a bucket by ``end_node`` in place, in ``O(size + inversions)``."""
    for i in range(bucket_start + 1, bucket_start + size):
        key = end_node[i]
        key_payload = payload[i]

        j = i - 1
        while j >= bucket_start and end_node[j] > key:
            end_node[j + 1] = end_node[j]
            payload[j + 1] = payload[j]
            j -= 1
        end_node[j + 1] = key
        payload[j + 1] = key_payload


@njit(cache=True)
def _adaptive_sort_bucket(end_node, payload, bucket_start, size):
    """Sorts a large bucket by ``end_node`` in place, insertion sorting it unless it turns out
    to be badly ordered, in which case the partial work is abandoned for a heap sort.

    This is ``_insertion_sort_bucket``'s loop with a shift meter around it. The duplication is
    deliberate: metering every bucket instead of only the large ones measured ~5% slower
    end-to-end, because typical buckets hold a handful of edges and the per-element bookkeeping
    is a real fraction of that work. Keep the two in sync rather than merging them.
    """
    budget = MAX_SHIFTS_PER_EDGE * size
    shifts = 0

    for i in range(bucket_start + 1, bucket_start + size):
        key = end_node[i]
        key_payload = payload[i]

        j = i - 1
        while j >= bucket_start and end_node[j] > key:
            end_node[j + 1] = end_node[j]
            payload[j + 1] = payload[j]
            j -= 1
        end_node[j + 1] = key
        payload[j + 1] = key_payload

        shifts += i - 1 - j
        if shifts > budget:
            _heap_sort_bucket(end_node, payload, bucket_start, size)
            return


@njit(cache=True)
def _count_unique_in_bucket(end_node, bucket_start, bucket_end):
    """Number of distinct keys in an already sorted bucket, where equal keys are adjacent."""
    n_unique = 0
    previous_end_node = INT_FILL_VALUE

    for i in range(bucket_start, bucket_end):
        if end_node[i] != previous_end_node:
            n_unique += 1
            previous_end_node = end_node[i]

    return n_unique


@njit(cache=True)
def _is_lexicographically_sorted(pairs):
    """Whether every row of an ``(n, 2)`` array is ascending and the rows are themselves in
    nondecreasing lexicographic order."""
    previous_low = INT_FILL_VALUE
    previous_high = INT_FILL_VALUE

    for i in range(pairs.shape[0]):
        low = pairs[i, 0]
        high = pairs[i, 1]

        if low > high:
            return False
        if low < previous_low or (low == previous_low and high < previous_high):
            return False

        previous_low = low
        previous_high = high

    return True


@njit(cache=True)
def _count_pairs_per_bucket(pairs, n_bucket):
    """Bucket offsets keyed on each row's lower value: bucket ``a`` will occupy
    ``[bucket_offset[a], bucket_offset[a + 1])``."""
    bucket_offset = np.zeros(n_bucket + 1, dtype=INT_DTYPE)

    for i in range(pairs.shape[0]):
        bucket_offset[min(pairs[i, 0], pairs[i, 1]) + 1] += 1

    for a in range(n_bucket):
        bucket_offset[a + 1] += bucket_offset[a]

    return bucket_offset


@njit(cache=True)
def _build_pair_index(pairs, n_bucket):
    """Indexes an ``(n, 2)`` array of integer pairs so a pair can be looked up by value.

    Each row is bucketed on its lower value and each bucket ordered by its higher value, which is
    what lets :func:`_search_bucket` find a row with a single binary search. Rows that are already
    canonically ordered are indexed without being sorted again.

    Returns ``(bucket_offset, high, row)``: bucket ``a`` occupies
    ``[bucket_offset[a], bucket_offset[a + 1])``, ``high`` holds each entry's higher value, and
    ``row`` the index of ``pairs`` it came from. ``high`` is a copy rather than a column view, both
    to keep the search off a strided array and so that either path below returns the same arrays.
    """
    bucket_offset = _count_pairs_per_bucket(pairs, n_bucket)

    n_pair = pairs.shape[0]
    high = np.empty(n_pair, dtype=INT_DTYPE)
    row = np.empty(n_pair, dtype=INT_DTYPE)

    if _is_lexicographically_sorted(pairs):
        # Already grouped by lower value, ascending within each group, so the buckets are the
        # runs the counting pass just measured and no sorting is needed
        for i in range(n_pair):
            high[i] = pairs[i, 1]
            row[i] = i
        return bucket_offset, high, row

    for i in range(n_pair):
        pair_low = pairs[i, 0]
        pair_high = pairs[i, 1]
        if pair_low > pair_high:
            pair_low, pair_high = pair_high, pair_low

        slot = bucket_offset[pair_low]
        high[slot] = pair_high
        row[slot] = i
        bucket_offset[pair_low] = slot + 1

    # The scatter left each entry at its bucket's end, one slot right of where the convention
    # above wants it. One backward pass puts it back.
    for a in range(n_bucket, 0, -1):
        bucket_offset[a] = bucket_offset[a - 1]
    bucket_offset[0] = 0

    for a in range(n_bucket):
        bucket_start = bucket_offset[a]
        _sort_bucket(high, row, bucket_start, bucket_offset[a + 1] - bucket_start)

    return bucket_offset, high, row


@njit(cache=True)
def _search_bucket(high, row, bucket_start, bucket_end, key):
    """The ``row`` entry whose key is ``key`` within a sorted bucket, or ``INT_FILL_VALUE``
    when the bucket does not hold it."""
    low = bucket_start
    stop = bucket_end

    while low < stop:
        mid = (low + stop) // 2
        if high[mid] < key:
            low = mid + 1
        else:
            stop = mid

    if low < bucket_end and high[low] == key:
        return row[low]
    return INT_FILL_VALUE

#
# End edge_node connectivity sorting routines
#
# ------------------------------------------------------------------------------------------