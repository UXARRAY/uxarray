import functools

from uxarray.core.dataarray import UxDataArray

from uxarray.regrid import nearest_neighbor


class UxDataArrayRegridAccessor:
    _uxda: UxDataArray

    __slots__ = ("_uxda",)

    def __init__(self, uxda: UxDataArray) -> None:
        self._uxda = uxda

    @functools.wraps(nearest_neighbor)
    def nearest_neighbor(self, *args, **kwargs):
        pass
