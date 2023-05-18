import gmpy2
from gmpy2 import mpfr
import numpy as np
from .constants import INT_DTYPE, INT_FILL_VALUE, FLOAT_PRECISION_BITS


def convert_to_mpfr(input_array, str_mode=True, precision=FLOAT_PRECISION_BITS):
    """
    Convert a numpy array to a list of mpfr numbers.
    The default precision of an mpfr is 53 bits - the same precision as Python’s `float` type.
    https://gmpy2.readthedocs.io/en/latest/mpfr.html

    Parameters
    ----------
    input_array : numpy array, float/string, shape is arbitrary
        The input array to be converted to mpfr. The input array should be float or string. If the input array is float,
        str_mode should be False. If the input array is string, str_mode should be True.

    str_mode : bool, optional
        If True, the input array should be string when passing into the function.
        If False, the input array should be float when passing into the function.
        str_mode is True by default and is recommended. Because to take advantage of the higher precision provided by
        the mpfr type, always pass constants as strings.
    precision : int, optional
        The precision of the mpfr numbers. The default precision of an mpfr is 53 bits - the same precision as Python’s `float` type.

    Returns
    ----------
    mpfr_array : numpy array, mpfr type, shape will be same as the input_array
        The output array with mpfr type, which supports correct
        rounding, selectable rounding modes, and many trigonometric, exponential, and special functions. A context
        manager is used to control precision, rounding modes, and the behavior of exceptions.
    """
    gmpy2.set_context(gmpy2.context())
    gmpy2.get_context().precision = precision

    # To take advantage of the higher precision provided by the mpfr type, always pass constants as strings.
    # https://gmpy2.readthedocs.io/en/latest/mpfr.html
    if not str_mode:
        # Cast the input 2D array to string array
        input_array = input_array.astype(str)
    else:
        flattened_array = np.ravel(input_array)
        if ~np.all([np.issubdtype(type(element), np.str_) for element in flattened_array]):
            raise ValueError('The input array should be string when str_mode is True.')

    # Then convert the input array to mpfr array
    mpfr_array = np.array([gmpy2.mpfr(x,precision) for x in input_array.ravel()]).reshape(input_array.shape)
    return mpfr_array


def unique_coordinates_mpfr(input_array_mpfr, precision=FLOAT_PRECISION_BITS):
    """
    Find the unique coordinates in the input array with mpfr numbers.
    The default precision of an mpfr is 53 bits - the same precision as Python’s `float` type.

    Parameters:
    ----------


    """

    # Reset the mpfr precision
    gmpy2.get_context().precision = precision

    # Check if the input_array is in th mpfr type
    try:
        # Flatten the input_array_mpfr to a 1D array so that we can check the type of each element
        input_array_mpfr_copy = np.ravel(input_array_mpfr)
        for i in range(len(input_array_mpfr_copy)):
            if type(input_array_mpfr_copy[i]) != gmpy2.mpfr:
                raise ValueError('The input array should be in the mpfr type. You can use convert_to_mpfr() to '
                                 'convert the input array to mpfr.')
    except Exception as e:
        raise e

    unique_arr = []
    inverse_indices = []
    m, n = input_array_mpfr.shape

    unique_dict = {}
    current_index = 0

    for i in range(m):
        format_string = "{0:+."+str(precision+1)+"Uf}"
        hashable_row = tuple(format_string.format(gmpy2.mpfr(x, precision)) for x in input_array_mpfr[i])

        if hashable_row not in unique_dict:
            unique_dict[hashable_row] = current_index
            unique_arr.append(input_array_mpfr[i])
            inverse_indices.append(current_index)
            current_index += 1
        else:
            inverse_indices.append(unique_dict[hashable_row])

    unique_arr = np.array(unique_arr)
    inverse_indices = np.array(inverse_indices)

    return unique_arr, inverse_indices
