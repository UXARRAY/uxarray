from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from uxarray.grid import Grid

import numpy as np
import holoviews as hv
from holoviews import opts

from holoviews.operation.datashader import rasterize as hds_rasterize


def plot(grid: Grid, **kwargs):
    """Default Plotting Method for Grid."""
    return mesh(grid, **kwargs)


def mesh(uxgrid: Grid,
         backend: Optional[str] = "bokeh",
         exclude_antimeridian: Optional[bool] = False,
         width: Optional[int] = 1000,
         height: Optional[int] = 500,
         **kwargs):
    """Vector Line Plot.

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

    gdf = uxgrid.to_geodataframe(exclude_antimeridian=exclude_antimeridian)

    hv_paths = hv.Path(gdf)

    if backend == "matplotlib":
        # use holoviews matplotlib backend
        hv.extension("matplotlib")

        return hv_paths.opts(**kwargs)

    elif backend == "bokeh":
        # use holoviews bokeh backend
        hv.extension("bokeh")
        return hv_paths.opts(width=width, height=height, **kwargs)


def nodes(uxgrid: Grid,
          backend: Optional[str] = "bokeh",
          width: Optional[int] = 1000,
          height: Optional[int] = 500,
          **kwargs):
    """Vector Node Plot.

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

    hv_points = hv.Points(np.array([uxgrid.node_lon, uxgrid.node_lat]).T)

    if backend == "matplotlib":
        # use holoviews matplotlib backend
        hv.extension("matplotlib")

        return hv_points.opts(**kwargs)

    elif backend == "bokeh":
        # use holoviews bokeh backend
        hv.extension("bokeh")
        return hv_points.opts(width=width, height=height, **kwargs)


def rasterize(uxgrid: Grid,
              method: Optional[str] = "mesh",
              backend: Optional[str] = "bokeh",
              exclude_antimeridian: Optional[bool] = False,
              pixel_ratio: Optional[float] = 1.0,
              dynamic: Optional[bool] = False,
              precompute: Optional[bool] = True,
              projection: Optional[ccrs] = None,
              width: Optional[int] = 1000,
              height: Optional[int] = 500,
              aggregator: Optional[str] = "mean",
              interpolation: Optional[str] = "linear",
              npartitions: Optional[int] = 1,
              cache: Optional[bool] = True,
              **kwargs):
    """Rasterized Plot of an Unstructured Grid Element ("mesh" or "nodes")

    Parameters
    ----------
    method: str
        Selects what type of element to rasterize ("mesh" or "nodes")
    backend: str
        Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
    exclude_antimeridian: bool,
        Whether to exclude faces that cross the antimeridian (Polygon Raster Only)
    projection: ccrs
         Custom projection to transform (lon, lat) coordinates for rendering
    pixel_ratio: float
        Determines the resolution of the outputted raster.
    height: int
        Plot Height for Bokeh Backend
    width: int
        Plot Width for Bokeh Backend
    cache: bool
            Determines where computed elements (i.e. points, polygons) should be cached internally for subsequent plotting
            calls

    Notes
    -----
    For further information about supported keyword arguments, please refer to the [Holoviews Documentation](https://holoviews.org/_modules/holoviews/operation/datashader.html#rasterize)
    or run holoviews.help(holoviews.operation.datashader.rasterize).
    """

    if method == "mesh":
        gdf = uxgrid.to_geodataframe(exclude_antimeridian=exclude_antimeridian)
        geometry = hv.Path(gdf)
    elif method == "node":
        geometry = hv.Points(np.array([uxgrid.node_lon, uxgrid.node_lat]).T)
    else:
        raise ValueError(f"Unsupported method {method}.")

    if backend == "matplotlib":
        # use holoviews matplotlib backend
        hv.extension("matplotlib")
        raster = hds_rasterize(geometry,
                               pixel_ratio=pixel_ratio,
                               dynamic=dynamic,
                               precompute=precompute,
                               aggregator=aggregator,
                               interpolation=interpolation).opts(
                                   colorbar=colorbar, cmap=cmap, **kwargs)
    elif backend == "bokeh":
        # use holoviews bokeh backend
        hv.extension("bokeh")
        raster = hds_rasterize(geometry,
                               pixel_ratio=pixel_ratio,
                               dynamic=dynamic,
                               precompute=precompute,
                               aggregator=aggregator,
                               interpolation=interpolation).opts(
                                   width=width,
                                   height=height,
                                   colorbar=colorbar,
                                   cmap=cmap,
                                   **kwargs)

    return raster
