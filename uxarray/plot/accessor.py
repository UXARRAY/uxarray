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
        return grid_plot.plot(self._uxgrid, **kwargs)

    @functools.wraps(grid_plot.mesh)
    def mesh(self,
             backend: Optional[str] = "bokeh",
             exclude_antimeridian: Optional[bool] = False,
             width: Optional[int] = 1000,
             height: Optional[int] = 500,
             **kwargs):
        """Vector Line Plot of the edges that make up each face.

        Parameters
        ----------
        backend: str
            Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
         exclude_antimeridian: bool,
            Whether to exclude edges that cross the antimeridian
        height: int
            Plot Height for Bokeh Backend
        width: int
            Plot Width for Bokeh Backend
        """
        return grid_plot.mesh(self._uxgrid,
                              backend=backend,
                              exclude_antimeridian=exclude_antimeridian,
                              width=width,
                              height=height,
                              **kwargs)

    @functools.wraps(grid_plot.mesh)
    def edges(self,
              backend: Optional[str] = "bokeh",
              exclude_antimeridian: Optional[bool] = False,
              width: Optional[int] = 1000,
              height: Optional[int] = 500,
              **kwargs):
        """Vector Line Plot of the edges that make up each face. Equivalent to
        ``Grid.plot.mesh()``

        Parameters
        ----------
        backend: str
            Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
         exclude_antimeridian: bool,
            Whether to exclude edges that cross the antimeridian
        height: int
            Plot Height for Bokeh Backend
        width: int
            Plot Width for Bokeh Backend
        """
        return grid_plot.mesh(self._uxgrid,
                              backend=backend,
                              exclude_antimeridian=exclude_antimeridian,
                              width=width,
                              height=height,
                              **kwargs)

    @functools.wraps(grid_plot.node_coords)
    def node_coords(self,
                    backend: Optional[str] = "bokeh",
                    width: Optional[int] = 1000,
                    height: Optional[int] = 500,
                    **kwargs):
        """Vector Point Plot of Nodes (latitude & longitude of the nodes that
        define the corners of each face)

        Parameters
        ----------
        backend: str
            Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
         exclude_antimeridian: bool,
            Whether to exclude edges that cross the antimeridian
        height: int
            Plot Height for Bokeh Backend
        width: int
            Plot Width for Bokeh Backend
        """

        return grid_plot.node_coords(self._uxgrid,
                                     backend=backend,
                                     width=width,
                                     height=height,
                                     **kwargs)

    @functools.wraps(grid_plot.node_coords)
    def nodes(self,
              backend: Optional[str] = "bokeh",
              width: Optional[int] = 1000,
              height: Optional[int] = 500,
              **kwargs):
        """Vector Point Plot of Nodes (latitude & longitude of the nodes that
        define the corners of each face). Alias of ``plot.node_coords``

        Parameters
        ----------
        backend: str
            Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
         exclude_antimeridian: bool,
            Whether to exclude edges that cross the antimeridian
        height: int
            Plot Height for Bokeh Backend
        width: int
            Plot Width for Bokeh Backend
        """

        return grid_plot.node_coords(self._uxgrid,
                                     backend=backend,
                                     width=width,
                                     height=height,
                                     **kwargs)

    @functools.wraps(grid_plot.face_coords)
    def face_coords(self,
                    backend: Optional[str] = "bokeh",
                    width: Optional[int] = 1000,
                    height: Optional[int] = 500,
                    **kwargs):
        """Vector Point Plot of Face Coordinates (latitude & longitude of the
        centroid of each face)

        Parameters
        ----------
        backend: str
            Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
         exclude_antimeridian: bool,
            Whether to exclude edges that cross the antimeridian
        height: int
            Plot Height for Bokeh Backend
        width: int
            Plot Width for Bokeh Backend
        """

        return grid_plot.face_coords(self._uxgrid,
                                     backend=backend,
                                     width=width,
                                     height=height,
                                     **kwargs)

    @functools.wraps(grid_plot.face_coords)
    def face_centers(self,
                     backend: Optional[str] = "bokeh",
                     width: Optional[int] = 1000,
                     height: Optional[int] = 500,
                     **kwargs):
        """Vector Point Plot of Face Coordinates (latitude & longitude of the
        centroid of each face). Alias of ``plot.face_coords``

        Parameters
        ----------
        backend: str
            Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
         exclude_antimeridian: bool,
            Whether to exclude edges that cross the antimeridian
        height: int
            Plot Height for Bokeh Backend
        width: int
            Plot Width for Bokeh Backend
        """

        return grid_plot.face_coords(self._uxgrid,
                                     backend=backend,
                                     width=width,
                                     height=height,
                                     **kwargs)

    @functools.wraps(grid_plot.edge_coords)
    def edge_coords(self,
                    backend: Optional[str] = "bokeh",
                    width: Optional[int] = 1000,
                    height: Optional[int] = 500,
                    **kwargs):
        """Vector Point Plot of Edge Coordinates (latitude & longitude of the
        centroid of each edge)

        Parameters
        ----------
        backend: str
            Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
         exclude_antimeridian: bool,
            Whether to exclude edges that cross the antimeridian
        height: int
            Plot Height for Bokeh Backend
        width: int
            Plot Width for Bokeh Backend
        """

        return grid_plot.edge_coords(self._uxgrid,
                                     backend=backend,
                                     width=width,
                                     height=height,
                                     **kwargs)

    @functools.wraps(grid_plot.edge_coords)
    def edge_centers(self,
                     backend: Optional[str] = "bokeh",
                     width: Optional[int] = 1000,
                     height: Optional[int] = 500,
                     **kwargs):
        """Vector Point Plot of Edge Coordinates (latitude & longitude of the
        centroid of each edge). Alias of ``plot.edge_coords``

        Parameters
        ----------
        backend: str
            Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
         exclude_antimeridian: bool,
            Whether to exclude edges that cross the antimeridian
        height: int
            Plot Height for Bokeh Backend
        width: int
            Plot Width for Bokeh Backend
        """

        return grid_plot.edge_coords(self._uxgrid,
                                     backend=backend,
                                     width=width,
                                     height=height,
                                     **kwargs)


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

    @functools.wraps(dataarray_plot.rasterize)
    def rasterize(self,
                  method: Optional[str] = "polygon",
                  backend: Optional[str] = "bokeh",
                  exclude_antimeridian: Optional[bool] = False,
                  pixel_ratio: Optional[float] = 1.0,
                  dynamic: Optional[bool] = False,
                  precompute: Optional[bool] = True,
                  projection: Optional[ccrs] = None,
                  width: Optional[int] = 1000,
                  height: Optional[int] = 500,
                  colorbar: Optional[bool] = True,
                  cmap: Optional[str] = "Blues",
                  aggregator: Optional[str] = "mean",
                  interpolation: Optional[str] = "linear",
                  npartitions: Optional[int] = 1,
                  cache: Optional[bool] = True,
                  override: Optional[bool] = False,
                  size: Optional[int] = 5,
                  **kwargs):
        """Raster plot of a data variable residing on an unstructured grid
        element.

        Parameters
        ----------
        method: str
            Selects what type of element to rasterize (point, trimesh, polygon).
        backend: str
            Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
        projection: ccrs
             Custom projection to transform (lon, lat) coordinates for rendering
        pixel_ratio: float
            Determines the resolution of the outputted raster.
        cache: bool
            Determines where computed elements (i.e. points, polygons) should be cached internally for subsequent plotting
            calls

        Notes
        -----
        For further information about supported keyword arguments, please refer to the [Holoviews Documentation](https://holoviews.org/_modules/holoviews/operation/datashader.html#rasterize)
        or run holoviews.help(holoviews.operation.datashader.rasterize).
        """

        return dataarray_plot.rasterize(
            self._uxda,
            method=method,
            backend=backend,
            exclude_antimeridian=exclude_antimeridian,
            pixel_ratio=pixel_ratio,
            dynamic=dynamic,
            precompute=precompute,
            projection=projection,
            width=width,
            height=height,
            colorbar=colorbar,
            cmap=cmap,
            aggregator=aggregator,
            interpolation=interpolation,
            npartitions=npartitions,
            cache=cache,
            override=override,
            size=size,
            **kwargs)

    @functools.wraps(dataarray_plot.polygons)
    def polygons(self,
                 backend: Optional[str] = "bokeh",
                 exclude_antimeridian: Optional[bool] = False,
                 width: Optional[int] = 1000,
                 height: Optional[int] = 500,
                 colorbar: Optional[bool] = True,
                 cmap: Optional[str] = "Blues",
                 cache: Optional[bool] = True,
                 override: Optional[bool] = False,
                 **kwargs):
        """Vector polygon plot shaded using a face-centered data variable.

        Parameters
        ----------
        backend: str
            Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
        exclude_antimeridian: bool,
            Whether to exclude faces that cross the antimeridian (Polygon Raster Only)
        height: int
            Plot Height for Bokeh Backend
        width: int
            Plot Width for Bokeh Backend
        """

        return dataarray_plot.polygons(
            self._uxda,
            backend=backend,
            exclude_antimeridian=exclude_antimeridian,
            width=width,
            height=height,
            colorbar=colorbar,
            cmap=cmap,
            cache=cache,
            override=override,
            **kwargs)

    @functools.wraps(dataarray_plot.points)
    def points(self,
               backend: Optional[str] = "bokeh",
               projection: Optional = None,
               width: Optional[int] = 1000,
               height: Optional[int] = 500,
               colorbar: Optional[bool] = True,
               cmap: Optional[str] = "Blues",
               **kwargs):
        """Vector Point Plot of a Data Variable Mapped to either Node, Edge, or
        Face Coordinates.

        Parameters
        ----------
        backend: str
            Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
         exclude_antimeridian: bool,
            Whether to exclude edges that cross the antimeridian
        height: int
            Plot Height for Bokeh Backend
        width: int
            Plot Width for Bokeh Backend
        """

        return dataarray_plot.points(self._uxda,
                                     backend=backend,
                                     projection=projection,
                                     width=width,
                                     height=height,
                                     colorbar=colorbar,
                                     cmap=cmap,
                                     **kwargs)


class UxDatasetPlotAccessor:
    """Plotting Accessor for UxDataset, accessed through ``UxDataset.plot()``
    or ``UxDataset.plot.specific_routine()``"""
    _uxds: UxDataset
    __slots__ = ("_uxds",)

    def __init__(self, uxds: UxDataset) -> None:
        self._uxds = uxds

    def __call__(self, **kwargs) -> Any:
        warnings.warn(
            "Plotting for UxDataset instances not yet supported. Did you mean to plot a data variable, i.e. uxds['data_variable'].plot()"
        )
        pass
