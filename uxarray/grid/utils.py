import numpy as np
import xarray as xr
from uxarray.constants import INT_FILL_VALUE

from numba import njit


def make_setter(key: str):
    """Return a setter that assigns the value to self._ds[key] after type-checking."""

    def setter(self, value):
        if not isinstance(value, xr.DataArray):
            raise ValueError(f"{key} must be an xr.DataArray")
        self._ds[key] = value

    return setter


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


def _replace_fill_values(grid_var, original_fill, new_fill, new_dtype=None):
    """Replaces all instances of the current fill value (``original_fill``) in
    (``grid_var``) with (``new_fill``) and converts to the dtype defined by
    (``new_dtype``)

    Parameters
    ----------
    grid_var : xr.DataArray
        Grid variable to be modified
    original_fill : constant
        Original fill value used in (``grid_var``)
    new_fill : constant
        New fill value to be used in (``grid_var``)
    new_dtype : np.dtype, optional
        New data type to convert (``grid_var``) to

    Returns
    -------
    grid_var : xr.DataArray
        Modified DataArray with updated fill values and dtype
    """

    # Identify fill value locations
    if original_fill is not None and np.isnan(original_fill):
        # For NaN fill values
        fill_val_idx = grid_var.isnull()
        # Temporarily replace NaNs with a placeholder if dtype conversion is needed
        if new_dtype is not None and np.issubdtype(new_dtype, np.floating):
            grid_var = grid_var.fillna(0.0)
        else:
            # Choose an appropriate placeholder for non-floating types
            grid_var = grid_var.fillna(new_fill)
    else:
        # For non-NaN fill values
        fill_val_idx = grid_var == original_fill

    # Convert to the new data type if specified
    if new_dtype is not None and new_dtype != grid_var.dtype:
        grid_var = grid_var.astype(new_dtype)

    # Validate that the new_fill can be represented in the new_dtype
    if new_dtype is not None:
        if np.issubdtype(new_dtype, np.integer):
            int_min = np.iinfo(new_dtype).min
            int_max = np.iinfo(new_dtype).max
            if not (int_min <= new_fill <= int_max):
                raise ValueError(
                    f"New fill value: {new_fill} not representable by integer dtype: {new_dtype}"
                )
        elif np.issubdtype(new_dtype, np.floating):
            if not (
                np.isnan(new_fill)
                or (np.finfo(new_dtype).min <= new_fill <= np.finfo(new_dtype).max)
            ):
                raise ValueError(
                    f"New fill value: {new_fill} not representable by float dtype: {new_dtype}"
                )
        else:
            raise ValueError(f"Data type {new_dtype} not supported for grid variables")

    grid_var = grid_var.where(~fill_val_idx, new_fill)

    return grid_var
