from __future__ import annotations
from typing import TYPE_CHECKING, Any, overload

import functools

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray
    from uxarray.grid import Grid

import uxarray.plot.grid_plot as grid_plot


class GridPlotAccessor:
    _uxgrid: Grid
    __slots__ = ("_uxgrid",)

    def __init__(self, uxgrid: Grid) -> None:
        self._uxgrid = uxgrid

    @functools.wraps(grid_plot.plot)
    def __call__(self, **kwargs) -> Any:
        return grid_plot.plot(self._uxgrid, **kwargs)

    @functools.wraps(grid_plot.nodes)
    def nodes(self, **kwargs):
        return grid_plot.nodes(self._uxgrid, **kwargs)

    @functools.wraps(grid_plot.edges)
    def edges(self, **kwargs):
        return grid_plot.edges(self._uxgrid, **kwargs)


class UxDataArrayPlotAccessor:
    _uxda: UxDataArray
    __slots__ = ("_uxda",)

    def __init__(self, uxda: UxDataArray) -> None:
        self._uxda = uxda


class UxDatasetPlotAccessor:
    _uxds: UxDataset
    __slots__ = ("_uxds",)

    def __init__(self, uxds: UxDataset) -> None:
        self._uxds = uxds
