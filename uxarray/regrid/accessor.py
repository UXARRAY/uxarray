from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import uxarray.regrid.dataarray_regrid as dataarray_regrid

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray


class UxDataArrayRegridAccessor:
    _uxda: UxDataArray

    __slots__ = ("_uxda",)

    def __init__(self, uxda: UxDataArray) -> None:
        self._uxda = uxda

    @functools.wraps(dataarray_regrid._nearest_neighbor)
    def nearest_neighbor(self, *args, **kwargs):
        pass
