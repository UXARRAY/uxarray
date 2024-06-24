from __future__ import annotations

import matplotlib

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray

# consider making these required depndencies
from cartopy import crs as ccrs
import dask.dataframe as dd
import holoviews as hv
from holoviews.operation.datashader import rasterize as hds_rasterize

import numpy as np

import pandas as pd

import uxarray.plot.utils


def plot(uxda, **kwargs):
    """Default plotting method for a ``UxDataArray``."""

    if uxda._face_centered():
        return rasterize(uxda, method="polygon", **kwargs)

    elif uxda._edge_centered() or uxda._node_centered():
        return rasterize(uxda, method="point", **kwargs)

    else:
        raise ValueError(
            "Plotting variables on unstructured grids requires the data variable to be mapped to either the nodes, edges, or faces."
        )


def datashade(
    uxda: UxDataArray,
    *args,
    method: Optional[str] = "polygon",
    plot_height: Optional[int] = 300,
    plot_width: Optional[int] = 600,
    x_range: Optional[tuple] = (-180, 180),
    y_range: Optional[tuple] = (-90, 90),
    cmap: Optional[str] = "Blues",
    agg: Optional[str] = "mean",
    **kwargs,
):
    """Visualizes an unstructured grid data variable using data shading
    (rasterization + shading).

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
    import datashader as ds
    import datashader.transfer_functions as tf

    cvs = ds.Canvas(plot_width, plot_height, x_range, y_range)
    gdf = uxda.to_geodataframe()

    if agg == "mean":
        _agg = ds.mean
    elif agg == "sum":
        _agg = ds.sum
    else:
        raise ValueError("Invalid agg")

    aggregated = cvs.polygons(gdf, geometry="geometry", agg=_agg(uxda.name))

    # support mpl colormaps
    try:
        _cmap = matplotlib.colormaps[cmap]
    except KeyError:
        _cmap = cmap

    return tf.shade(aggregated, cmap=_cmap, **kwargs)


def rasterize(
    uxda: UxDataArray,
    method: Optional[str] = "point",
    backend: Optional[str] = "bokeh",
    exclude_antimeridian: Optional[bool] = True,
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
    **kwargs,
):
    """Rasterized Plot of a Data Variable Residing on an Unstructured Grid.

    Parameters
    ----------
    method: str
        Selects what type of element to rasterize (point, trimesh, polygon), with "point" being the only currently
        implemented method.
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

    if method == "point":
        # perform point rasterization
        raster = _point_raster(
            uxda=uxda,
            backend=backend,
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
            size=size,
            **kwargs,
        )
    elif method == "polygon":
        raster = _polygon_raster(
            uxda=uxda,
            backend=backend,
            exclude_antimeridian=exclude_antimeridian,
            dynamic=dynamic,
            precompute=precompute,
            width=width,
            height=height,
            colorbar=colorbar,
            cmap=cmap,
            aggregator=aggregator,
            interpolation=interpolation,
            pixel_ratio=pixel_ratio,
            cache=cache,
            override=override,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported method: {method}.")

    return raster


def _point_raster(
    uxda: UxDataArray,
    backend: Optional[str] = "bokeh",
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
    xlabel: Optional[str] = "Longitude",
    ylabel: Optional[str] = "Latitude",
    size: Optional[int] = 5.0,
    **kwargs,
):
    """Implementation of Point Rasterization."""

    if "clabel" not in kwargs:
        # set default label for color bar
        kwargs["clabel"] = uxda.name

    if uxda._face_centered():
        # data mapped to face centroid coordinates
        lon, lat = uxda.uxgrid.face_lon.values, uxda.uxgrid.face_lat.values
    elif uxda._node_centered():
        # data mapped to face corner coordinates
        lon, lat = uxda.uxgrid.node_lon.values, uxda.uxgrid.node_lat.values
    elif uxda._edge_centered():
        # data mapped to face corner coordinates
        lon, lat = uxda.uxgrid.edge_lon.values, uxda.uxgrid.edge_lat.values
    else:
        raise ValueError(
            f"The Dimension of Data Variable {uxda.name} is not Node or Face centered."
        )

    if projection is not None:
        # apply projection to coordinates
        lon, lat, _ = projection.transform_points(ccrs.PlateCarree(), lon, lat).T

    uxarray.plot.utils.backend.assign(backend=backend)

    point_dict = {"lon": lon, "lat": lat, "var": uxda.data}
    point_df = pd.DataFrame.from_dict(point_dict)
    point_ddf = dd.from_pandas(point_df, npartitions=npartitions)

    # construct a holoviews points object
    if backend == "matplotlib":
        points = hv.Points(point_ddf, ["lon", "lat"]).opts(s=size)
    else:
        points = hv.Points(point_ddf, ["lon", "lat"]).opts(size=size)

    if backend == "matplotlib":
        # use holoviews matplotlib backend
        raster = hds_rasterize(
            points,
            pixel_ratio=pixel_ratio,
            dynamic=dynamic,
            precompute=precompute,
            aggregator=aggregator,
            interpolation=interpolation,
        ).opts(
            colorbar=colorbar,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )
    elif backend == "bokeh":
        # use holoviews bokeh backend
        raster = hds_rasterize(
            points,
            pixel_ratio=pixel_ratio,
            dynamic=dynamic,
            precompute=precompute,
            aggregator=aggregator,
            interpolation=interpolation,
        ).opts(
            width=width,
            height=height,
            colorbar=colorbar,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Invalid backend selected. Expected one of ['matplotlib', 'bokeh'] but received {backend}."
        )

    return raster


def _polygon_raster(
    uxda: UxDataArray,
    backend: Optional[str] = "bokeh",
    exclude_antimeridian: Optional[bool] = True,
    pixel_ratio: Optional[float] = 1.0,
    dynamic: Optional[bool] = False,
    precompute: Optional[bool] = True,
    width: Optional[int] = 1000,
    height: Optional[int] = 500,
    colorbar: Optional[bool] = True,
    cmap: Optional[str] = "Blues",
    aggregator: Optional[str] = "mean",
    interpolation: Optional[str] = "linear",
    xlabel: Optional[str] = "Longitude",
    ylabel: Optional[str] = "Latitude",
    cache: Optional[bool] = True,
    override: Optional[bool] = False,
    **kwargs,
):
    """Implementation of Polygon Rasterization."""

    if "clabel" not in kwargs:
        # set default label for color bar
        kwargs["clabel"] = uxda.name

    gdf = uxda.to_geodataframe(
        exclude_antimeridian=exclude_antimeridian, cache=cache, override=override
    )

    hv_polygons = hv.Polygons(gdf, vdims=[uxda.name])

    uxarray.plot.utils.backend.assign(backend=backend)

    if backend == "matplotlib":
        # use holoviews matplotlib backend
        raster = hds_rasterize(
            hv_polygons,
            pixel_ratio=pixel_ratio,
            dynamic=dynamic,
            precompute=precompute,
            aggregator=aggregator,
            interpolation=interpolation,
        ).opts(
            colorbar=colorbar,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )
    elif backend == "bokeh":
        # use holoviews bokeh backend
        raster = hds_rasterize(
            hv_polygons,
            pixel_ratio=pixel_ratio,
            dynamic=dynamic,
            precompute=precompute,
            aggregator=aggregator,
            interpolation=interpolation,
        ).opts(
            width=width,
            height=height,
            colorbar=colorbar,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Invalid backend selected. Expected one of ['matplotlib', 'bokeh'] but received {backend}."
        )

    return raster


def polygons(
    uxda: UxDataArray,
    backend: Optional[str] = "bokeh",
    exclude_antimeridian: Optional[bool] = True,
    projection: Optional = None,
    width: Optional[int] = 1000,
    height: Optional[int] = 500,
    colorbar: Optional[bool] = True,
    cmap: Optional[str] = "Blues",
    xlabel: Optional[str] = "Longitude",
    ylabel: Optional[str] = "Latitude",
    cache: Optional[bool] = True,
    override: Optional[bool] = False,
    **kwargs,
):
    """Vector Polygon Plot of a Data Variable Residing on an Unstructured Grid.

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
    if "clabel" not in kwargs:
        # set default label for color bar
        kwargs["clabel"] = uxda.name

    gdf = uxda.to_geodataframe(
        exclude_antimeridian=exclude_antimeridian, cache=cache, override=override
    )

    hv_polygons = hv.Polygons(gdf, vdims=[uxda.name])

    uxarray.plot.utils.backend.assign(backend=backend)
    if backend == "matplotlib":
        # use holoviews matplotlib backend

        return hv_polygons.opts(
            colorbar=colorbar, xlabel=xlabel, ylabel=ylabel, cmap=cmap, **kwargs
        )

    elif backend == "bokeh":
        # use holoviews bokeh backend
        return hv_polygons.opts(
            width=width,
            height=height,
            colorbar=colorbar,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )


def points(
    uxda: UxDataArray,
    backend: Optional[str] = "bokeh",
    width: Optional[int] = 1000,
    height: Optional[int] = 500,
    colorbar: Optional[bool] = True,
    cmap: Optional[str] = "Blues",
    projection=None,
    **kwargs,
):
    """Vector Point Plot of a Data Variable Mapped to either Node, Edge, or
    Face Coordinates."""

    if uxda.values.ndim > 1:
        raise ValueError(
            f"Data Variable must be 1-dimensional, with shape {uxda.uxgrid.n_node}, {uxda.uxgrid.n_edge}, {uxda.uxgrid.n_face} "
            f"for node-centered, edge-centered, or face-centered data respectively."
        )

    if uxda._node_centered():
        return _plot_data_as_points(
            element="node",
            uxda=uxda,
            backend=backend,
            width=width,
            height=height,
            colorbar=colorbar,
            cmap=cmap,
            projection=projection,
            **kwargs,
        )
    elif uxda._face_centered():
        return _plot_data_as_points(
            element="face",
            uxda=uxda,
            backend=backend,
            width=width,
            height=height,
            colorbar=colorbar,
            cmap=cmap,
            projection=projection,
            **kwargs,
        )
    elif uxda._edge_centered():
        return _plot_data_as_points(
            element="edge",
            uxda=uxda,
            backend=backend,
            width=width,
            height=height,
            colorbar=colorbar,
            cmap=cmap,
            projection=projection,
            **kwargs,
        )
    else:
        raise ValueError("Data Variable is not mapped to nodes, edges, or faces.")


def _plot_data_as_points(
    element,
    uxda: UxDataArray,
    backend: Optional[str] = "bokeh",
    width: Optional[int] = 1000,
    height: Optional[int] = 500,
    colorbar: Optional[bool] = True,
    cmap: Optional[str] = "Blues",
    xlabel: Optional[str] = "Longitude",
    ylabel: Optional[str] = "Latitude",
    projection=None,
    **kwargs,
):
    """Helper function for plotting data variables as Points, either on the
    Nodes, Face Centers, or Edge Centers."""

    from holoviews import Points

    if "clabel" not in kwargs:
        # set default label for color bar
        kwargs["clabel"] = uxda.name

    uxgrid = uxda.uxgrid
    if element == "node":
        lon, lat = uxgrid.node_lon.values, uxgrid.node_lat.values
    elif element == "face":
        lon, lat = uxgrid.face_lon.values, uxgrid.face_lat.values
    elif element == "edge":
        lon, lat = uxgrid.edge_lon.values, uxgrid.edge_lat.values
    else:
        raise ValueError("Invalid element selected.")

    if projection is not None:
        lon, lat, _ = projection.transform_points(ccrs.PlateCarree(), lon, lat).T

    verts = np.column_stack([lon, lat, uxda.values])
    hv_points = Points(verts, vdims=["z"])

    uxarray.plot.utils.backend.assign(backend=backend)

    if backend == "matplotlib":
        # use holoviews matplotlib backend
        return hv_points.opts(
            color="z",
            colorbar=colorbar,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )

    elif backend == "bokeh":
        # use holoviews bokeh backend
        return hv_points.opts(
            color="z",
            width=width,
            height=height,
            colorbar=colorbar,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )
