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
