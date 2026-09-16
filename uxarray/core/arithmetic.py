"""
Purpose: ensures arithmetic for uxarray objects behaves as expected,
e.g. checks for compatible uxgrids (if more than one input has a uxgrid),
and ensure appropriate result types (e.g. xr.DataArray + UxDataArray --> UxDataArray).

Includes solutions for issues #1685, #1695, #1718:
(1685) Uxarray types dropped during ufuncs
    Solved by appropriately defining __array_ufunc__.
(1695) Uxarray types dropped during math with xarrays, depending on order of operations
    Solved by explicitly overriding dunder methods like __mul__,
    so Python's default behavior for dunder methods and subclasses triggers
    (which just requires `uxarray_obj.__mul__ is not xarray_obj.__mul__`)
    and dispatches to uxarray's methods instead of xarray's methods.
    (The overridden methods just call super()'s method, because they go to _binary_op,
    which goes to uxarray's _binary_op(), which handles types appropriately.
    _binary_op() already handles types appropriately because it uses
    _replace() for DataArrays, which UxDataArray already overrides, and it uses
    _calculate_binary_op() for Datasets, which UxDataset overrides.)
(1718) Math with multiple uxarray objects with different grids silently succeeds,
when data dims are otherwise compatible
    Solved by comparing uxgrids during _binary_op() if both inputs have a uxgrid,
    and by comparing uxgrids of all inputs during __array_ufunc__,
    if multiple inputs to __array_ufunc__ have a "uxgrid" attr.
    (The former fixes, e.g., objA + objB; the latter fixes, e.g., np.add(objA, objB))
"""

import operator

import xarray as xr

from uxarray.errors import GridsMismatchError


class UxSupportsArithmetic:
    """Handles arithmetic overrides for uxarray objects as needed,
    to ensure they behave as expected. E.g.:

    - check for compatible uxgrids (if more than one input has a uxgrid),
    - ensure appropriate result types (e.g. xr.DataArray + UxDataArray --> UxDataArray).


    Expects to be placed in inheritance order *before* xarray classes,
        e.g. UxDataArray(UxSupportsArithmetic, xr.DataArray),
        not xr.DataArray(UxSupportsArithmetic, UxDataArray),
    so that super() will resolve to methods from xarray.
    """

    # If strict type-checking gets turned on, it might complain about super() calls here.
    # The fix would probably be to define a typing.Protocol parent class to inherit from.

    __slots__ = ()

    @property
    def uxgrid(self):
        """(UxSupportsArithmetic expects subclasses to implement uxgrid)"""
        raise NotImplementedError(f"{type(self).__name__}.uxgrid")
        # below, checks for "has a uxgrid" are written as `isinstance(obj, UxSupportsArithmetic)`,
        # rather than checking for a uxgrid attribute, because the latter would be too permissive,
        # e.g. hasattr(xr.DataArray(7, coords={"uxgrid": "anything"}), "uxgrid") is True.

    def _binary_op(self, other, f, reflexive=False, **kw_super):
        """returns f(self, other) (or f(other, self), if `reflexive`) for f a binary operation,
        such as adding or multiplying. Like super()._binary_op, except that
        if `other` has a uxgrid, first ensure it is compatible with self.uxgrid.
        """
        if isinstance(other, UxSupportsArithmetic):
            if self.uxgrid != other.uxgrid:
                raise GridsMismatchError(
                    f"A.uxgrid != B.uxgrid during binary operation {f.__name__!r}, "
                    f"with type(A)={type(self).__name__}, type(B)={type(other).__name__}."
                )
        return super()._binary_op(other, f, reflexive=reflexive, **kw_super)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Like super().__array_ufunc__, except that if multiple inputs have a uxgrid,
        ensure they are all compatible, and if any outputs are xr.DataArray or xr.Dataset,
        convert to the appropriate uxarray type (UxDataArray or UxDataset).
        """
        from uxarray.core.dataarray import UxDataArray
        from uxarray.core.dataset import UxDataset

        uxgrids = [
            (i, obj.uxgrid)
            for i, obj in enumerate(inputs)
            if isinstance(obj, UxSupportsArithmetic)
        ]

        # At least one input is guaranteed to have a uxgrid, otherwise numpy
        # would not have delegated to this method during the ufunc call.
        # It might be self (often, the first input in inputs with a grid would be self),
        # but doesn't need to be --> use the first grid, rather than self's grid, as reference.
        j_ref, grid_ref = uxgrids[0]

        # check that all uxgrids are compatible:
        if len(uxgrids) > 1:
            for j, grid in uxgrids[1:]:
                if grid != grid_ref:
                    raise GridsMismatchError(
                        f"Multiple inputs to {ufunc.__name__!r} have incompatible uxgrids. "
                        f"Got inputs[{j}].uxgrid != inputs[{j_ref}].uxgrid, for inputs "
                        f"with types: type(inputs[{j}])={type(inputs[j])}, "
                        f"type(inputs[{j_ref}])={type(inputs[j_ref])}."
                    )
        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

        # convert any xr.DataArray / xr.Dataset outputs to appropriate uxarray type:
        _was_tuple = isinstance(result, tuple)
        result = list(result) if _was_tuple else [result]
        for i, obj in enumerate(result):
            if isinstance(obj, xr.DataArray):
                result[i] = UxDataArray(obj, uxgrid=grid_ref)
            elif isinstance(obj, xr.Dataset):
                result[i] = UxDataset(obj, uxgrid=grid_ref)
        result = tuple(result) if _was_tuple else result[0]
        return result

    # shadow dunder methods to ensure that, in case of ops with both uxarray and xarray,
    # Python's default behavior for dunder methods and subclasses triggers.
    # E.g., shadowing __add__ & __radd__ ensures that xarr + uxarr goes to uxarr.__radd__(xarr),
    # instead of xarr.__add__(uxarr), so the result can actually be a UxAarray.
    # (Just calling super()'s method is sufficient because it goes to self._binary_op,
    # which goes to uxarray's _binary_op because self will be the uxarray object.)

    def __add__(self, other):
        return super().__add__(other)

    def __radd__(self, other):
        return super().__radd__(other)

    def __sub__(self, other):
        return super().__sub__(other)

    def __rsub__(self, other):
        return super().__rsub__(other)

    def __mul__(self, other):
        return super().__mul__(other)

    def __rmul__(self, other):
        return super().__rmul__(other)

    def __pow__(self, other):
        return super().__pow__(other)

    def __rpow__(self, other):
        return super().__rpow__(other)

    def __truediv__(self, other):
        return super().__truediv__(other)

    def __rtruediv__(self, other):
        return super().__rtruediv__(other)

    def __floordiv__(self, other):
        return super().__floordiv__(other)

    def __rfloordiv__(self, other):
        return super().__rfloordiv__(other)

    def __mod__(self, other):
        return super().__mod__(other)

    def __rmod__(self, other):
        return super().__rmod__(other)

    def __and__(self, other):
        return super().__and__(other)

    def __rand__(self, other):
        return super().__rand__(other)

    def __or__(self, other):
        return super().__or__(other)

    def __ror__(self, other):
        return super().__ror__(other)

    def __xor__(self, other):
        return super().__xor__(other)

    def __rxor__(self, other):
        return super().__rxor__(other)

    def __lshift__(self, other):
        return super().__lshift__(other)

    def __rlshift__(self, other):
        # (this isn't in super() for some reason, so use _binary_op directly)
        return self._binary_op(other, operator.lshift, reflexive=True)

    def __rshift__(self, other):
        return super().__rshift__(other)

    def __rrshift__(self, other):
        # (this isn't in super() for some reason, so use _binary_op directly)
        return self._binary_op(other, operator.rshift, reflexive=True)

    # comparisons

    def __lt__(self, other):
        return super().__lt__(other)

    def __le__(self, other):
        return super().__le__(other)

    def __gt__(self, other):
        return super().__gt__(other)

    def __ge__(self, other):
        return super().__ge__(other)

    def __eq__(self, other):
        return super().__eq__(other)

    def __ne__(self, other):
        return super().__ne__(other)
