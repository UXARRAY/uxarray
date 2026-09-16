"""
Purpose: Test basic arithmetic operations on uxarray objects,
such as addition, multiplication, and some ufuncs.
(Not intended for testing specialized methods like spherical math operations.)

E.g., results should have the appropriate type,
and operations should raise errors if uxgrids of the operands are incompatible.
"""
import operator

import numpy as np
import pytest
import uxarray as ux
import xarray as xr

# Binary math operators which uxarray should guarantee work as expected,
# for operations involving UxDataArray and/or UxDataset objects.
# Created based on the list of binary ops supported for xarray objects.
# If needing to support more, (e.g. __matmul__), add them here.
MATH_BINARY_OPS = [
    operator.add, operator.sub, operator.mul, operator.pow,
    operator.truediv, operator.floordiv, operator.mod,
    operator.and_, operator.or_, operator.xor,
    operator.lshift, operator.rshift,
]

COMPARISON_BINARY_OPS = [
    operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne,
]

ALL_BINARY_OPS = MATH_BINARY_OPS + COMPARISON_BINARY_OPS

# spot-check a few binary ufuncs too:
BINARY_UFUNCS_TO_TEST = [
    np.add, np.subtract, np.multiply, np.divide, np.minimum, np.maximum,
]

# later tests will spot-check a few unary ufuncs too:
UNARY_UFUNCS_TO_TEST = [np.negative, np.absolute, np.square]


def test_binary_ops_and_binary_ufuncs_output_types():
    """Checks types of outputs behave as expected in binary operations,
    including in a few binary ufuncs like np.add and np.minimum.
    Includes regression test for issue #1695 (binary ops)
    and partial regression test for issue #1685 (but just binary ufuncs here)

    For "*" representing any binary operator, ideally would want
    uxarray-typed result whenever either input is uxarray-typed,
    and a Dataset whenever either input is a Dataset.

    The tests below incorporate the following checks (for most binary ops):
        (A) UxDataArray * UxDataArray --> UxDataArray
        (B) UxDataArray * UxDataset --> UxDataset
        (C) UxDataArray * xr.DataArray --> UxDataArray
        (D) UxDataArray * xr.Dataset --> UxDataset
        (E) UxDataset * UxDataArray --> UxDataset
        (F) UxDataset * UxDataset --> UxDataset
        (G) UxDataset * xr.DataArray --> UxDataset
        (H) UxDataset * xr.Dataset --> UxDataset
        (I) xr.DataArray * UxDataArray --> UxDataArray
        (J) xr.DataArray * UxDataset --> UxDataset or xr.Dataset (see below)
        (K) xr.Dataset * UxDataArray --> xr.Dataset (see below)
        (L) xr.Dataset * UxDataset --> UxDataset

    Case (K) explanation:
        The case of xr.Dataset * UxDataArray returns an xr.Dataset, instead of a UxDataset,
        not because it is desirable, but because it cannot be fixed in uxarray directly.
        xr.Dataset * UxDataArray will always go to xr.Dataset's relevant method (e.g. __mul__),
        which goes to xr.Dataset._binary_op, which sees that `other` is a DataArray subclass,
        and thus processes it as such, without giving that subclass a chance to do anything.

        Fixing this (without monkeypatching) would require xarray itself to call some sort of
        hook which subclasses could override, i.e. it would require editing xarray directly.
        Hopefully this case isn't particularly common, and when it does occur, can use the
        workaround of "put the uxarray object first" or explicitly cast to UxDataset beforehand.

    Case (J) explanation:
        Here, the output will be UxDataset if using a binary op (e.g. xrda + uxds),
        but xr.Dataset if using a ufunc (e.g. np.add(xrda, uxds)).

        This is because numpy's __array_ufunc__ sees that UxDataset is not a subclass of xr.DataArray,
        so it starts applying __array_ufunc__ of the inputs in order instead of dispatching directly
        to UxDataset.__array_ufunc__. This hits xr.DataArray's __array_ufunc__ first, which sees that
        `other` is a Dataset and moves forward instead of providing a hook or returning NotImplemented,
        so UxDataset never gets a chance to do anything.

        Again, fixing this (without monkeypatching) would require changes to xarray itself.
    """
    xarr = xr.DataArray([1, 2, 3, 4], dims='n_face')
    uarr0 = ux.tutorial.open_dataset('quad-hexagon')['t2m']
    # some spot checks right away for easier debugging if anything is very wrong;
    # also makes an example with "nice" small nonzero ints to use below in comprehensive tests,
    # small to avoid complaints from pow, nonzero to avoid complaints from div,
    # and ints to avoid complaints from lshift/rshift (which don't support floats).
    assert isinstance(uarr0 * xarr, ux.UxDataArray)
    assert isinstance(xarr * uarr0, ux.UxDataArray)
    uarr = uarr0 * 0 + 2 * xarr
    uarr = uarr.astype('int')
    assert isinstance(uarr, ux.UxDataArray)

    xds = xr.Dataset({'varname': xarr})
    uds = ux.UxDataset({'varname': uarr}, uxgrid=uarr.uxgrid)

    for op in ALL_BINARY_OPS + BINARY_UFUNCS_TO_TEST:
        # (A) UxDataArray * UxDataArray --> UxDataArray
        assert isinstance(op(uarr, uarr), ux.UxDataArray)
        # (B) UxDataArray * UxDataset --> UxDataset
        assert isinstance(op(uarr, uds), ux.UxDataset)
        # (C) UxDataArray * xr.DataArray --> UxDataArray
        assert isinstance(op(uarr, xarr), ux.UxDataArray)
        # (D) UxDataArray * xr.Dataset --> UxDataset
        assert isinstance(op(uarr, xds), ux.UxDataset)
        # (E) UxDataset * UxDataArray --> UxDataset
        assert isinstance(op(uds, uarr), ux.UxDataset)
        # (F) UxDataset * UxDataset --> UxDataset
        assert isinstance(op(uds, uds), ux.UxDataset)
        # (G) UxDataset * xr.DataArray --> UxDataset
        assert isinstance(op(uds, xarr), ux.UxDataset)
        # (H) UxDataset * xr.Dataset --> UxDataset
        assert isinstance(op(uds, xds), ux.UxDataset)
        # (I) xr.DataArray * UxDataArray --> UxDataArray
        assert isinstance(op(xarr, uarr), ux.UxDataArray)
        # (J) xr.DataArray * UxDataset --> UxDataset
        if op in ALL_BINARY_OPS:
            assert isinstance(op(xarr, uds), ux.UxDataset)
        else:
            assert isinstance(op(xarr, uds), xr.Dataset)
        # (K) xr.Dataset * UxDataArray --> xr.Dataset (see above for explanation)
        assert isinstance(op(xds, uarr), xr.Dataset)
        # (L) xr.Dataset * UxDataset --> UxDataset
        assert isinstance(op(xds, uds), ux.UxDataset)


def test_unary_ufuncs_output_types():
    """Checks types of outputs behave as expected in unary ufuncs.
    Partial regression test for issue #1685 (but just binary ufuncs here)
    """
    uds = ux.tutorial.open_dataset('quad-hexagon')
    uarr = uds['t2m']
    for ufunc in UNARY_UFUNCS_TO_TEST:
        assert isinstance(ufunc(uds), ux.UxDataset)
        assert isinstance(ufunc(uarr), ux.UxDataArray)


def test_multi_output_ufunc_output_types():
    """Spot check that output types behave as expected for multi-output ufuncs."""
    uds = ux.tutorial.open_dataset('quad-hexagon')
    uarr = uds['t2m']
    # np.divmod is a multi-output ufunc, returning a tuple of two outputs.
    out1, out2 = np.divmod(uds, 2)
    assert isinstance(out1, ux.UxDataset)
    assert isinstance(out2, ux.UxDataset)
    out1, out2 = np.divmod(uarr, 2)
    assert isinstance(out1, ux.UxDataArray)
    assert isinstance(out2, ux.UxDataArray)


def test_ops_check_grid_compatibility():
    """Ensure ops check that grids compare as equal before proceeding.
    E.g., uxarr0 + uxarr1 should crash if uxarr0.uxgrid != uxarr1.uxgrid.
    Regression test for issue #1718.
    """
    uarrA = ux.tutorial.open_dataset('quad-hexagon')['t2m']
    _tmp = ux.tutorial.open_dataset('outCSne30-timeseries')['psi']
    uarrB = _tmp.isel(n_face=[10,20,30,40])
    with pytest.raises(ux.errors.GridsMismatchError):
        uarrA + uarrB
    udsA = ux.UxDataset({'varname': uarrA}, uxgrid=uarrA.uxgrid)
    udsB = ux.UxDataset({'varname': uarrB}, uxgrid=uarrB.uxgrid)
    with pytest.raises(ux.errors.GridsMismatchError):
        udsA * udsB
    with pytest.raises(ux.errors.GridsMismatchError):
        np.add(uarrA, uarrB)
    with pytest.raises(ux.errors.GridsMismatchError):
        udsA - uarrB
    with pytest.raises(ux.errors.GridsMismatchError):
        uarrA / uarrA.to_xarray() + uarrB
