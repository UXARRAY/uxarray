from __future__ import annotations

import matplotlib

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray

from cartopy import crs as ccrs
import dask.dataframe as dd
import holoviews as hv
import geoviews as gv
from holoviews.operation.datashader import rasterize as hds_rasterize


import pandas as pd

import uxarray.plot.utils


def rasterize(
    uxda: UxDataArray,
    method: Optional[str] = "point",
    backend: Optional[str] = "bokeh",
    periodic_elements: Optional[str] = "exclude",
    exclude_antimeridian: Optional[bool] = None,
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
            periodic_elements=periodic_elements,
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
            projection=projection,
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

    if "clabel" not in kwargs and uxda.name is not None:
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

    uxarray.plot.utils.backend.assign(backend)
    current_backend = hv.Store.current_backend

    point_dict = {"lon": lon, "lat": lat, "var": uxda.data}
    point_df = pd.DataFrame.from_dict(point_dict)
    point_ddf = dd.from_pandas(point_df, npartitions=npartitions)

    # construct a holoviews points object
    if current_backend == "matplotlib":
        points = hv.Points(point_ddf, ["lon", "lat"]).opts(s=size)
    else:
        points = hv.Points(point_ddf, ["lon", "lat"]).opts(size=size)

    if current_backend == "matplotlib":
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
    elif current_backend == "bokeh":
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
    periodic_elements: Optional[str] = "exclude",
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
    projection: Optional[ccrs] = None,
    **kwargs,
):
    """Implementation of Polygon Rasterization."""

    if "clabel" not in kwargs and uxda.name is not None:
        # set default label for color bar
        kwargs["clabel"] = uxda.name

    gdf = uxda.to_geodataframe(
        projection=projection,
        periodic_elements=periodic_elements,
        exclude_antimeridian=exclude_antimeridian,
        cache=cache,
        override=override,
    )

    uxarray.plot.utils.backend.assign(backend)
    current_backend = hv.Store.current_backend

    if current_backend == "matplotlib":
        _polygons = gv.Polygons(
            gdf,
            vdims=[uxda.name if uxda.name is not None else "var"],
        )
    else:
        # GeoViews Issue with Projections:
        _polygons = hv.Polygons(
            gdf, vdims=[uxda.name if uxda.name is not None else "var"]
        )

    if current_backend == "matplotlib":
        # use holoviews matplotlib backend
        raster = hds_rasterize(
            _polygons,
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
    elif current_backend == "bokeh":
        # use holoviews bokeh backend
        raster = hds_rasterize(
            _polygons,
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
