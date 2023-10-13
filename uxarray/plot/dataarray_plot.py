from __future__ import annotations

import matplotlib

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray

# consider making these required depndencies
from cartopy import crs as ccrs
import dask.dataframe as dd
import dask.array as da
import holoviews as hv
from holoviews.operation.datashader import rasterize as hds_rasterize
from holoviews import opts

import pandas as pd


def plot(uxda, **kwargs):
    """Default Plotting Method for UxDataArray."""
    return rasterize(uxda, **kwargs)


def datashade(uxda: UxDataArray,
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

    aggregated = cvs.polygons(gdf, geometry='geometry', agg=_agg(uxda.name))

    # support mpl colormaps
    try:
        _cmap = matplotlib.colormaps[cmap]
    except KeyError:
        _cmap = cmap

    return tf.shade(aggregated, cmap=_cmap, **kwargs)


def _point_raster(uxda: UxDataArray,
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
                  **kwargs):
    """Implementation of Point Rasterization."""

    if uxda._face_centered():
        # data mapped to face centroid coordinates
        lon = uxda.uxgrid.Mesh2_face_x.values
        lat = uxda.uxgrid.Mesh2_face_y.values
    elif uxda._node_centered():
        # data mapped to face corner coordinates
        lon = uxda.uxgrid.Mesh2_node_x.values
        lat = uxda.uxgrid.Mesh2_node_y.values
    else:
        raise ValueError(
            f"The Dimension of Data Variable {uxda.name} is not Node or Face centered."
        )

    # determine whether we need to recompute points, typically when a new projection is selected
    recompute = True
    if uxda._face_centered() == "center":
        if uxda.uxgrid._centroid_points_df_proj[
                0] is not None and uxda.uxgrid._centroid_points_df_proj[
                    1] == projection:
            recompute = False
            points_df = uxda.uxgrid._centroid_points_df_proj[0]

    else:
        if uxda.uxgrid._corner_points_df_proj[
                0] is not None and uxda.uxgrid._corner_points_df_proj[
                    1] == projection:
            recompute = False
            points_df = uxda.uxgrid._corner_points_df_proj[0]

    if recompute:
        # need to recompute points and/or projection
        if projection is not None:
            lon, lat, _ = projection.transform_points(ccrs.PlateCarree(), lon,
                                                      lat).T

        point_dict = {"lon": lon, "lat": lat, "var": uxda.values}

        # Construct Dask DataFrame
        point_ddf = dd.from_dict(data=point_dict, npartitions=npartitions)

        points = hv.Points(point_ddf, ['lon', 'lat'])

        # cache computed points & projection
        if cache:
            if uxda._face_centered() == "center":
                uxda.uxgrid._centroid_points_df_proj[0] = point_ddf
                uxda.uxgrid._centroid_points_df_proj[1] = projection
            else:
                uxda.uxgrid._corner_points_df_proj[0] = point_ddf
                uxda.uxgrid._corner_points_df_proj[1] = projection

    else:
        # use existing cached points & projection
        points_df['var'] = pd.Series(uxda.values)
        points = hv.Points(points_df, ['lon', 'lat'])

    if backend == "matplotlib":
        # use holoviews matplotlib backend
        hv.extension("matplotlib")
        raster = hds_rasterize(points,
                               pixel_ratio=pixel_ratio,
                               dynamic=dynamic,
                               precompute=precompute,
                               aggregator=aggregator,
                               interpolation=interpolation).opts(
                                   colorbar=colorbar, cmap=cmap, **kwargs)
    elif backend == "bokeh":
        # use holoviews bokeh backend
        hv.extension("bokeh")
        raster = hds_rasterize(points,
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

    else:
        raise ValueError(
            f"Invalid backend selected. Expected one of ['matplotlib', 'bokeh'] but received {backend}."
        )

    return raster


def rasterize(uxda: UxDataArray,
              method: Optional[str] = "point",
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
              **kwargs):
    """Performs an unstructured grid rasterization for visualuzation.

    Parameters
    ----------
    method: str
        Selects what type of element to rasterize (point, trimesh, polygon), with "point" being the only currently
        implemented method.
    backend: str
        Selects whether to use Holoview's "matplotlib" or "bokeh" backend for rendering plots
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
        raster = _point_raster(uxda, backend, pixel_ratio, dynamic, precompute,
                               projection, width, height, colorbar, cmap,
                               aggregator, interpolation, npartitions, cache,
                               **kwargs)
    elif method == "polygon":
        raise ValueError(f"Polygon Rasterization not yet implemented.")
    elif method == "trimesh":
        raise ValueError(f"Trimesh Rasterization not yet implemented.")
    else:
        raise ValueError(f"Unsupported method {method}.")

    return raster
