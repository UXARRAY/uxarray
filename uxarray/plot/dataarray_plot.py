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
from holoviews import opts


def plot(uxda, **kwargs):
    """Default Plotting Method for UxDataArray."""
    return datashade(uxda, **kwargs)


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


def rasterize(uxda: UxDataArray,
              *args,
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
              **kwargs):

    if uxda.face_centered():
        # data mapped to face centroid coordinates
        data_mapping = "center"
        lon = uxda.uxgrid.Mesh2_face_x.values
        lat = uxda.uxgrid.Mesh2_face_y.values
        # Node-centered data
    elif uxda.node_centered():
        # data mapped to face corner coordinates
        data_mapping = "corner"
        lon = uxda.uxgrid.Mesh2_node_x.values
        lat = uxda.uxgrid.Mesh2_node_y.values
    else:
        raise ValueError(
            "Issue with data. It is neither face-centered nor node-centered!")

    if method == "point":

        recompute = True
        if data_mapping == "center":
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
            if projection is not None:
                lon, lat, _ = projection.transform_points(
                    ccrs.PlateCarree(), lon, lat).T

            point_dict = {"lon": lon, "lat": lat, "var": uxda.values}

            # Construct Dask DataFrame
            point_ddf = dd.from_dict(data=point_dict, npartitions=npartitions)

            points = hv.Points(point_ddf, ['lon', 'lat'])

            # cache computed points
            if data_mapping == "center":
                uxda.uxgrid._centroid_points_df_proj[0] = point_ddf
                uxda.uxgrid._centroid_points_df_proj[1] = projection
            else:
                uxda.uxgrid._corner_points_df_proj[0] = point_ddf
                uxda.uxgrid._corner_points_df_proj[1] = projection

        else:
            # assign a new data variable to existing points
            points_df.replace({'var': uxda.values})
            points = hv.Points(points_df, ['lon', 'lat'])

        if backend == "matplotlib":
            # TODO
            hv.extension("matplotlib")
            raster = hds_rasterize(points,
                                   pixel_ratio=pixel_ratio,
                                   dynamic=dynamic,
                                   precompute=precompute,
                                   aggregator=aggregator,
                                   interpolation=interpolation).opts(
                                       colorbar=colorbar, cmap=cmap, **kwargs)
            pass
        elif backend == "bokeh":
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
            raise ValueError

        return raster

    elif method == "polygon":
        pass
    elif method == "trimesh":
        pass
    else:
        raise ValueError
