import numpy as np
from uxarray.constants import ERROR_TOLERANCE
import warnings


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
            raise ValueError(f'New fill value: {new_fill} not representable by'
                             f' integer dtype: {grid_var.dtype}')

    # ensure non-nan fill value can be represented with current float data type
    elif np.issubdtype(new_dtype, np.floating) and not np.isnan(new_fill):
        float_min = np.finfo(grid_var.dtype).min
        float_max = np.finfo(grid_var.dtype).max
        # ensure new_fill is in range [float_min, float_max]
        if new_fill < float_min or new_fill > float_max:
            raise ValueError(f'New fill value: {new_fill} not representable by'
                             f' float dtype: {grid_var.dtype}')
    else:
        raise ValueError(f'Data type {grid_var.dtype} not supported'
                         f'for grid variables')

    # replace all zeros with a fill value
    grid_var[fill_val_idx] = new_fill

    return grid_var


def _fmms(a, b, c, d):
    """
    Calculate the difference of products using the FMA (fused multiply-add) operation: (a * b) - (c * d).

    This operation leverages the fused multiply-add operation when available on the system and rounds the result only once.
    The relative error of this operation is bounded by 1.5 ulps when no overflow and underflow occur.

    Parameters
    ----------
    a (float): The first value of the first product.
    b (float): The second value of the first product.
    c (float): The first value of the second product.
    d (float): The second value of the second product.

    Returns
    -------
    float: The difference of the two products.

    Example
    -------
    >>> _fmms(3.0,2.0,1.0,1.0)
    5.0

    Reference
    ---------
    Claude-Pierre Jeannerod, Nicolas Louvet, and Jean-Michel Muller, Further
    analysis of Kahan’s algorithm for the accurate computation of 2 x 2 determinants,
    Mathematics of Computation, vol. 82, no. 284, pp. 2245-2264, 2013.
    [Read more](https://ens-lyon.hal.science/ensl-00649347) (DOI: 10.1090/S0025-5718-2013-02679-8)
    """
    import pyfma
    cd = c * d
    err = pyfma.fma(-c, d, cd)
    dop = pyfma.fma(a, b, -cd)
    return dop + err


def cross_fma(v1, v2):
    """Calculate the cross product of two 3D vectors utilizing the fused
    multiply-add operation.

    Parameters
    ----------
    v1 (np.array): The first vector of size 3.
    v2 (np.array): The second vector of size 3.

    Returns
    -------
    np.array: The cross product vector of size 3.

    Example
    -------
    >>> v1 = np.array([1.0, 2.0, 3.0])
    >>> v2 = np.array([4.0, 5.0, 6.0])
    >>> cross_fma(v1, v2)
    array([-3.0, 6.0, -3.0])
    """
    x = _fmms(v1[1], v2[2], v1[2], v2[1])
    y = _fmms(v1[2], v2[0], v1[0], v2[2])
    z = _fmms(v1[0], v2[1], v1[1], v2[0])
    return np.array([x, y, z])


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
    jacobian = [[_fmms(y0, z1, z0, y1),
                 _fmms(x0, z1, z0, x1)], [2 * x_i_old, 2 * y_i_old]]
    try:
        inverse_jacobian = np.linalg.inv(jacobian)
    except np.linalg.LinAlgError:
        raise warnings("Warning: Singular Jacobian matrix encountered.")
        return None

    return inverse_jacobian


def _newton_raphson_solver_for_gca_constLat(init_cart,
                                            gca_cart,
                                            max_iter=1000,
                                            verbose=False):
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
    error = float('inf')
    constZ = init_cart[2]
    y_guess = np.array(init_cart[0:2])
    y_new = y_guess

    _iter = 0

    while error > tolerance and _iter < max_iter:
        f_vector = np.array([
            np.dot(np.cross(w0_cart, w1_cart),
                   np.array([y_guess[0], y_guess[1],
                             constZ])), y_guess[0] * y_guess[0] +
            y_guess[1] * y_guess[1] + constZ * constZ - 1.0
        ])

        j_inv = _inv_jacobian(w0_cart[0], w1_cart[0], w0_cart[1], w1_cart[1],
                              w0_cart[2], w1_cart[2], y_guess[0], y_guess[1])

        if j_inv is None:
            return None

        y_new = y_guess - np.matmul(j_inv, f_vector)
        error = np.max(np.abs(y_guess - y_new))
        y_guess = y_new

        if verbose:
            print(f"Newton method iter: {_iter}, error: {error}")
        _iter += 1

    return np.append(y_new, constZ)
