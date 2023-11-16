from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from uxarray.grid import Grid

import numpy as np
import holoviews as hv
from holoviews import opts


def plot(grid: Grid, **kwargs):
    """Default Plotting Method for Grid."""
    return mesh(grid, **kwargs)


def mesh(uxgrid: Grid,
         backend: Optional[str] = "bokeh",
         exclude_antimeridian: Optional[bool] = False,
         width: Optional[int] = 1000,
         height: Optional[int] = 500,
         xlabel: Optional[str] = "Longitude",
         ylabel: Optional[str] = "Latitude",
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

    gdf = uxgrid.to_geodataframe(exclude_antimeridian=exclude_antimeridian)

    hv_paths = hv.Path(gdf)

    if backend == "matplotlib":
        # use holoviews matplotlib backend
        hv.extension("matplotlib")

        return hv_paths.opts(**kwargs)

    elif backend == "bokeh":
        # use holoviews bokeh backend
        hv.extension("bokeh")
        return hv_paths.opts(width=width,
                             height=height,
                             xlabel=xlabel,
                             ylabel=ylabel,
                             **kwargs)


def node_coords(uxgrid: Grid,
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

    return _plot_coords_as_points(element="node",
                                  uxgrid=uxgrid,
                                  backend=backend,
                                  width=width,
                                  height=height,
                                  **kwargs)


def face_coords(uxgrid: Grid,
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

    return _plot_coords_as_points(element="face",
                                  uxgrid=uxgrid,
                                  backend=backend,
                                  width=width,
                                  height=height,
                                  **kwargs)


def edge_coords(uxgrid: Grid,
                backend: Optional[str] = "bokeh",
                width: Optional[int] = 1000,
                height: Optional[int] = 500,
                **kwargs):
    """Vector Point Plot of Edge Coordinates (latitude & longitude of the
    center of each edge)

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

    return _plot_coords_as_points(element="edge",
                                  uxgrid=uxgrid,
                                  backend=backend,
                                  width=width,
                                  height=height,
                                  **kwargs)


def _plot_coords_as_points(element,
                           uxgrid: Grid,
                           backend: Optional[str] = "bokeh",
                           width: Optional[int] = 1000,
                           height: Optional[int] = 500,
                           xlabel: Optional[str] = "Longitude",
                           ylabel: Optional[str] = "Latitude",
                           **kwargs):
    """Helper function for plotting coordinates (node, edge, face) as Points
    with Holoviews."""

    if element == "node":
        hv_points = hv.Points(
            np.array([uxgrid.node_lon, uxgrid.node_lat.values]).T)
    elif element == "face":
        hv_points = hv.Points(
            np.array([uxgrid.face_lon, uxgrid.face_lat.values]).T)
    elif element == "edge":
        hv_points = hv.Points(
            np.array([uxgrid.edge_lon, uxgrid.edge_lat.values]).T)
    else:
        raise ValueError("Invalid element selected.")

    if backend == "matplotlib":
        # use holoviews matplotlib backend
        hv.extension("matplotlib")
        return hv_points.opts(**kwargs)

    elif backend == "bokeh":
        # use holoviews bokeh backend
        hv.extension("bokeh")
        return hv_points.opts(width=width,
                              height=height,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              **kwargs)
