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
    """Plotting Accessor for Grid, accessed through ``Grid.plot()`` or
    ``Grid.plot.specific_routine()``"""
    _uxgrid: Grid
    __slots__ = ("_uxgrid",)

    def __init__(self, uxgrid: Grid) -> None:
        self._uxgrid = uxgrid

    def __call__(self, **kwargs) -> Any:
        warnings.warn("Plotting for UxDataset instances not yet supported.")
        pass

    @functools.wraps(grid_plot.mesh)
    def mesh(self, projection=None, **kwargs):

        return grid_plot.mesh(self._uxgrid, projection, **kwargs)


class UxDataArrayPlotAccessor:
    """Plotting Accessor for UxDataArray, accessed through
    ``UxDataArray.plot()`` or ``UxDataArray.plot.specific_routine()``"""
    _uxda: UxDataArray
    __slots__ = ("_uxda",)

    def __init__(self, uxda: UxDataArray) -> None:
        self._uxda = uxda

    @functools.wraps(dataarray_plot.plot)
    def __call__(self, **kwargs) -> Any:
        return dataarray_plot.plot(self._uxda, **kwargs)

    @functools.wraps(dataarray_plot.datashade)
    def datashade(self,
                  *args,
                  method: Optional[str] = "polygon",
                  plot_height: Optional[int] = 300,
                  plot_width: Optional[int] = 600,
                  x_range: Optional[tuple] = (-180, 180),
                  y_range: Optional[tuple] = (-90, 90),
                  cmap: Optional[str] = "Blues",
                  agg: Optional[str] = "mean",
                  **kwargs):
        """Visualizes an unstructured grid data variable using data shading
        (rasterization + shading)

        Parameters
        ----------
        method: str, optional
            Selects which method to use for data shading
        plot_width, plot_height : int, optional
           Width and height of the output aggregate in pixels.
        x_range, y_range : tuple, optional
           A tuple representing the bounds inclusive space ``[min, max]`` along
           the axis.
        cmap: str, optional
            Colormap used for shading
        agg : str, optional
            Reduction to compute. Default is "mean", but can be one of "mean" or "sum"
        """
        return dataarray_plot.datashade(self._uxda, *args, method, plot_height,
                                        plot_width, x_range, y_range, cmap, agg,
                                        **kwargs)

    def polygons(self, *args, **kwargs):
        pass


class UxDatasetPlotAccessor:
    """Plotting Accessor for UxDataset, accessed through ``UxDataset.plot()``
    or ``UxDataset.plot.specific_routine()``"""
    _uxds: UxDataset
    __slots__ = ("_uxds",)

    def __init__(self, uxds: UxDataset) -> None:
        self._uxds = uxds

    def __call__(self, **kwargs) -> Any:
        warnings.warn("Plotting for UxDataset instances not yet supported.")
        pass
