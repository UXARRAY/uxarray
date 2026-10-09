"""
Purpose: testing routines from utils/numba_math.py
"""
import numpy as np
import pytest

from uxarray.utils.numba_math import (
    _numba_add3,
    _numba_add3_scalar,
    _numba_sub3,
    _numba_mul3,
    _numba_mul3_scalar,
    _numba_div3,
    _numba_div3_scalar,
    _numba_sqrt3,
    _numba_norm3,
    _numba_dot3,
    _numba_cross3,
)

def test_numba_add3():
    """ensure _numba_add3 and _numba_add3_scalar work as expected."""
    assert _numba_add3((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)) == (5.0, 7.0, 9.0)
    assert _numba_add3((1,2,3), (4.0, 5.0, 6)) == (5.0, 7.0, 9)
    assert _numba_add3([1,2,3], [4,5,-6]) == (5, 7, -3)
    assert _numba_add3(np.array([1,2,3]), np.array([4,5,6])) == (5, 7, 9)
    assert _numba_add3_scalar((1.0, 2.0, 3.0), 10.0) == (11.0, 12.0, 13.0)
    assert _numba_add3_scalar([1,2,3], 10.0) == (11.0, 12.0, 13.0)
    assert _numba_add3_scalar(np.array([1,2,3]), -10) == (-9, -8, -7)
    with pytest.raises(TypeError, match="can't unbox heterogeneous list"):
        _numba_add3_scalar([1,2,3.0], 10)   # numba doesn't like [1,2,3.0]
    assert _numba_add3_scalar(np.array([1,2,3]), 10) == (11, 12, 13)
    assert _numba_add3_scalar(np.array([1,2,3.0]), 10.0) == (11.0, 12.0, 13.0)

def test_numba_sub3():
    """ensure _numba_sub3 works as expected."""
    assert _numba_sub3((1.0, 2.0, 3.0), (4.0, 6.0, -5.0)) == (-3.0, -4.0, 8.0)
    assert _numba_sub3([1,2,3], [4,5,-6]) == (-3, -3, 9)
    assert _numba_sub3(np.array([1,2,3]), np.array([4, 6, -5])) == (-3, -4, 8)

def test_numba_mul3():
    """ensure _numba_mul3 and _numba_mul3_scalar work as expected."""
    assert _numba_mul3((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)) == (4.0, 10.0, 18.0)
    assert _numba_mul3([1,2,3], [4,5,-6]) == (4, 10, -18)
    assert _numba_mul3(np.array([1,2,3]), np.array([4,5,-6])) == (4, 10, -18)
    assert _numba_mul3_scalar((1.0, 2.0, 3.0), 10.0) == (10.0, 20.0, 30.0)
    assert _numba_mul3_scalar([1,2,3], -10) == (-10, -20, -30)
    assert _numba_mul3_scalar(np.array([1,2,3]), 10) == (10, 20, 30)

def test_numba_div3():
    """ensure _numba_div3 and _numba_div3_scalar work as expected."""
    assert _numba_div3((4.0, 10.0, 18.0), (2.0, 5.0, 6.0)) == (2.0, 2.0, 3.0)
    assert _numba_div3([4,10,18], [2,5,-6]) == (2, 2, -3)
    assert _numba_div3(np.array([4,10,18]), np.array([2,5,-6])) == (2, 2, -3)
    assert _numba_div3_scalar((10.0, 20.0, 30.0), 10.0) == (1.0, 2.0, 3.0)
    assert _numba_div3_scalar([10,20,30], -10) == (-1, -2, -3)
    assert _numba_div3_scalar(np.array([10,20,30]), 10) == (1, 2, 3)

def test_numba_sqrt3():
    """ensure _numba_sqrt3 works as expected."""
    assert _numba_sqrt3((4.0, 9.0, 16.0)) == (2.0, 3.0, 4.0)
    assert _numba_sqrt3([4,9,16]) == (2.0, 3.0, 4.0)
    assert _numba_sqrt3(np.array([4,9,16])) == (2.0, 3.0, 4.0)

def test_numba_norm3():
    """ensure _numba_norm3 works as expected."""
    assert _numba_norm3((3.0, 4.0, 12.0)) == 13.0
    assert _numba_norm3([0, 3, 4]) == 5.0
    assert _numba_norm3(np.array([0, -2, 0])) == 2.0

def test_numba_dot3():
    """ensure _numba_dot3 works as expected."""
    assert _numba_dot3((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)) == 32.0
    assert _numba_dot3([1,2,3], [4,5,-6]) == -4
    assert _numba_dot3(np.array([1,10,0]), np.array([0,-1,5])) == -10

def test_numba_cross3():
    """ensure _numba_cross3 works as expected."""
    assert _numba_cross3((1, 2, 30), (4, 0, 6)) == (12, 114, -8)
    assert _numba_cross3([0,1,0],[0,0,10]) == (10,0,0)
    assert _numba_cross3(np.array([1,10,0]), np.array([0,-1,5])) == (50, -5, -1)
