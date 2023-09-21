from __future__ import annotations
from typing import TYPE_CHECKING, Any, overload, Optional

import functools

import warnings

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray
    from uxarray.grid import Grid

import uxarray.plot.grid_plot as grid_plot
import uxarray.plot.dataarray_plot as dataarray_plot


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

    @functools.wraps(dataarray_plot.plot)
    def __call__(self, **kwargs) -> Any:
        return dataarray_plot.plot(self._uxda, **kwargs)

    @functools.wraps(dataarray_plot.raster)
    def raster(self,
               plot_height: Optional[int] = 300,
               plot_width: Optional[int] = 600,
               cmap: Optional[str] = "blue",
               agg: Optional[str] = "mean"):  # TODO: return hint
        """TODO: Docstring & additional params"""
        return dataarray_plot.raster(self._uxda, plot_height, plot_width, cmap,
                                     agg)


class UxDatasetPlotAccessor:
    _uxds: UxDataset
    __slots__ = ("_uxds",)

    def __init__(self, uxds: UxDataset) -> None:
        self._uxds = uxds

    def __call__(self, **kwargs) -> Any:
        warnings.warn("Plotting for UxDataset instances not yet supported.")
        pass
