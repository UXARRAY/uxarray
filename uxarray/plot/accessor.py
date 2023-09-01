from uxarray.core.dataset import UxDataset
from uxarray.core.dataarray import UxDataArray
from uxarray.grid import Grid


class GridPlotAccessor:
    _uxgrid = Grid
    __slots__ = ("_uxgrid",)

    def __init__(self, uxgrid: Grid) -> None:
        self._uxgrid = uxgrid


class UxDataArrayPlotAccessor:
    _uxda = UxDataArray
    __slots__ = ("_uxgrid",)

    def __init__(self, uxda: UxDataArray) -> None:
        self._uxda = uxda


class UxDatasetPlotAccessor:
    _uxds = UxDataset
    __slots__ = ("_uxgrid",)

    def __init__(self, uxds: UxDataset) -> None:
        self._uxds = uxds
