from __future__ import annotations

import matplotlib
from cartopy import crs as ccrs

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray


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
              colorbar=True,
              cmap='coolwarm',
              width=1000,
              height=500,
              tools=['hover'],
              projection: Optional[ccrs] = None,
              aggregator='mean',
              interpolation='linear',
              precompute=True,
              dynamic=False,
              npartitions: Optional[int] = 1,
              **kwargs):
    """Visualizes an unstructured grid data variable using rasterization.

    Parameters
    ----------
    projection: cartopy.crs, optional
            Custom projection to transform the axis coordinates during display. Defaults to None.
    npartitions: int, optional
            Number of partions for Dask DataFrane construction
    """
    import dask.dataframe as dd
    import holoviews as hv
    from holoviews.operation.datashader import rasterize as hds_rasterize

    face_centered_data=True
    node_centered_data=True

    # Face-centered data
    if (uxda.uxgrid.nMesh2_face in uxda.shape and
            uxda.uxgrid.nMesh2_node not in uxda.shape):
        lon = uxda.uxgrid.Mesh2_face_x.values
        lat = uxda.uxgrid.Mesh2_face_y.values
    # Node-centered data
    elif (uxda.uxgrid.nMesh2_node in uxda.shape and
          uxda.uxgrid.nMesh2_face not in uxda.shape):
        lon = uxda.uxgrid.Mesh2_node_x.values
        lat = uxda.uxgrid.Mesh2_node_y.values
    else:
        raise ValueError("Issue with data. It is neither face-centered nor node-centered!")

    # Construct a point dictionary
    if projection is None:
        point_dict = {"lon": lon,
                      "lat": lat,
                      "var": uxda.values}
    # Transform axis coords w.r.t projection, if any
    else:
        xPCS, yPCS, _ = projection.transform_points(ccrs.PlateCarree(),
                                                    lon,
                                                    lat).T
        point_dict = {"lon": xPCS,
                      "lat": yPCS,
                      "var": uxda.values}

    # Construct Dask DataFrame
    point_ddf = dd.from_dict(data=point_dict, npartitions=npartitions)

    points = hv.Points(point_ddf, ['lon', 'lat'])

    # Rasterize
    raster = hds_rasterize(points,
                           aggregator=aggregator,
                           interpolation=interpolation,
                           precompute=precompute,
                           dynamic=dynamic,
                           **kwargs)

    return raster.opts(width=width, height=height, tools=tools, colorbar=colorbar, cmap=cmap)
