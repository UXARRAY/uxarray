import gmpy2
from gmpy2 import mpfr, mpz
import numpy as np
import math
from uxarray.utils.constants import FLOAT_PRECISION_BITS, INT_FILL_VALUE_MPZ


def set_global_precision(global_precision=FLOAT_PRECISION_BITS):
    """Set the global precision of the mpfr numbers.
    Important Note:
    1. To avoid arithmetic overflow, the global precision should always be higher than any other precision speicified
    in the code.
    2. Modifying the precision by calling this function will modify all following codes running context until
    another call to this function.

    Parameters
    ----------
    global_precision : int, optional
        The global precision of the expected multiprecision float.
        The default precision of an mpfr is 53 bits - the same precision as Python’s `float` type.

    Returns
    -------
    None
    """

    gmpy2.get_context().precision = global_precision


def convert_to_multiprecision(input_array,
                              str_mode=True,
                              precision=FLOAT_PRECISION_BITS):
    """Convert a numpy array to a list of mpfr numbers.

    The default precision of an mpfr is 53 bits - the same precision as Python’s `float` type.
    https://gmpy2.readthedocs.io/en/latest/mpfr.html
    If the input array contains fill values INT_FILL_VALUE, the fill values will be converted to INT_FILL_VALUE_MPZ,
    which is the multi-precision integer representation of INT_FILL_VALUE.

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

    Raises
    ----------
    ValueError
        The input array should be string when str_mode is True, if not, raise
        ValueError('The input array should be string when str_mode is True.')
    """

    # To take advantage of the higher precision provided by the mpfr type, always pass constants as strings.
    # https://gmpy2.readthedocs.io/en/latest/mpfr.html
    flattened_array = np.ravel(input_array)
    mpfr_array = np.array(flattened_array, dtype=object)
    if not str_mode:
        # Cast the input 2D array to string array
        for idx, val in enumerate(flattened_array):
            if gmpy2.cmp(mpz(val), INT_FILL_VALUE_MPZ) == 0:
                mpfr_array[idx] = INT_FILL_VALUE_MPZ
            else:
                decimal_digit = precision_bits_to_decimal_digits(precision)
                format_str = "{0:+." + str(decimal_digit) + "f}"
                val_str = format_str.format(val)
                mpfr_array[idx] = mpfr(val_str, precision)

    else:

        if ~np.all([
                np.issubdtype(type(element), np.str_)
                for element in flattened_array
        ]):
            raise ValueError(
                'The input array should be string when str_mode is True.')
        # Then convert the input array to mpfr array
        for idx, val in enumerate(flattened_array):
            if val == "INT_FILL_VALUE":
                mpfr_array[idx] = INT_FILL_VALUE_MPZ
            else:
                mpfr_array[idx] = mpfr(val, precision)

    mpfr_array = mpfr_array.reshape(input_array.shape)

    return mpfr_array


def unique_coordinates_multiprecision(input_array_mpfr,
                                      precision=FLOAT_PRECISION_BITS):
    """Find the unique coordinates in the input array with mpfr numbers.

    The default precision of an mpfr is 53 bits - the same precision as Python’s `float` type.
    It can recognize the fill values INT_FILL_VALUE_MPZ, which is the multi-precision integer representation of
    INT_FILL_VALUE.

    Parameters:
    ----------
    input_array_mpfr : numpy.ndarray, gmpy2.mpfr type
        The input array containing mpfr numbers.

    precision : int, optional
        The precision in bits used for the mpfr calculations. Default is FLOAT_PRECISION_BITS.

    Returns:
    -------
    unique_arr ： numpy.ndarray, gmpy2.mpfr
        Array of unique coordinates in the input array.

    inverse_indices: numpy.ndarray, int
        The indices to reconstruct the original array from the unique array. Only provided if return_inverse is True.

    Raises
    ----------
    ValueError
        The input array should be string when str_mode is True, if not, raise
        ValueError('The input array should be string when str_mode is True.')
    """

    # Check if the input_array is in th mpfr type
    try:
        # Flatten the input_array_mpfr to a 1D array so that we can check the type of each element
        input_array_mpfr_copy = np.ravel(input_array_mpfr)
        for i in range(len(input_array_mpfr_copy)):
            if type(input_array_mpfr_copy[i]) != gmpy2.mpfr and type(
                    input_array_mpfr_copy[i]) != gmpy2.mpz:
                raise ValueError(
                    'The input array should be in the mpfr type. You can use convert_to_mpfr() to '
                    'convert the input array to mpfr.')
    except Exception as e:
        raise e

    unique_arr = []
    inverse_indices = []
    m, n = input_array_mpfr.shape
    unique_dict = {}
    current_index = 0

    for i in range(m):
        # We only need to check the first element of each row since the elements in the same row are the same type
        # (Either mpfr for valid coordinates or INT_FILL_VALUE_MPZ for fill values)
        if type(input_array_mpfr[i][0]) == gmpy2.mpfr:
            format_string = "{0:+." + str(precision + 1) + "Uf}"
        elif type(input_array_mpfr[i][0]) == gmpy2.mpz:
            format_string = "{:<+" + str(precision + 1) + "d}"
        else:
            raise ValueError(
                'The input array should be in the mpfr/mpz type. You can use convert_to_multiprecision() '
                'to convert the input array to multiprecision format.')
        hashable_row = tuple(
            format_string.format(x) for x in input_array_mpfr[i])

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


def decimal_digits_to_precision_bits(decimal_digits):
    """Convert the number of decimal digits to the number of bits of precision.

    Parameters
    ----------
    decimal_digits : int
        The number of decimal digits of precision

    Returns
    -------
    bits : int
        The number of bits of precision
    """
    bits = math.ceil(decimal_digits * math.log2(10))
    return bits


def precision_bits_to_decimal_digits(precision):
    """Convert the number of bits of precision to the number of decimal digits.

    Parameters
    ----------
    precision : int
        The number of bits of precision

    Returns
    -------
    decimal_digits : int
        The number of decimal digits of precision
    """
    # Compute the decimal digit count using gmpy2.log10()
    log10_2 = gmpy2.log10(gmpy2.mpfr(2))  # Logarithm base 10 of 2
    log10_precision = gmpy2.div(1,
                                log10_2)  # Logarithm base 10 of the precision
    decimal_digits = gmpy2.div(precision, log10_precision)

    # Convert the result to an integer
    decimal_digits = int(math.floor(decimal_digits))

    return decimal_digits
