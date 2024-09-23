from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import functools


if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray
    from uxarray.grid import Grid

import uxarray.plot.grid_plot as grid_plot
import uxarray.plot.dataarray_plot as dataarray_plot
import uxarray.plot.utils

import warnings

import cartopy.crs as ccrs

import numpy as np
import pandas as pd

import holoviews as hv
import hvplot.pandas

from holoviews import Points


class GridPlotAccessor:
    """Plotting Accessor for Grid, accessed through ``Grid.plot()`` or
    ``Grid.plot.specific_routine()``"""

    _uxgrid: Grid
    __slots__ = ("_uxgrid",)

    def __init__(self, uxgrid: Grid) -> None:
        self._uxgrid = uxgrid

    def __call__(self, **kwargs) -> Any:
        return self.edges(**kwargs)

    def points(self, element="nodes", backend=None, **kwargs):
        """TODO:"""

        uxarray.plot.utils.backend.assign(backend)

        if element in ["nodes", "corner nodes", "node_latlon"]:
            lon, lat = self._uxgrid.node_lon.values, self._uxgrid.node_lat.values
        elif element in ["faces", "face centers", "face_latlon"]:
            lon, lat = self._uxgrid.face_lon.values, self._uxgrid.face_lat.values
        elif element in ["edges", "edge centers", "edge_latlon"]:
            lon, lat = self._uxgrid.edge_lon.values, self._uxgrid.edge_lat.values
        else:
            raise ValueError("TODO: ")

        verts = {"lon": lon, "lat": lat}

        points_df = pd.DataFrame.from_dict(verts)

        return points_df.hvplot.points("lon", "lat", **kwargs)

    def nodes(self, backend=None, **kwargs):
        """TODO:"""
        return self.points(element="nodes", backend=backend, **kwargs)

    def node_coords(self, backend=None, **kwargs):
        """TODO:"""
        return self.points(element="nodes", backend=backend, **kwargs)

    def edge_coords(self, backend=None, **kwargs):
        """TODO:"""
        return self.points(element="edges", backend=backend, **kwargs)

    def edge_centers(self, backend=None, **kwargs):
        """TODO:"""
        return self.points(element="edges", backend=backend, **kwargs)

    def facecoords(self, backend=None, **kwargs):
        """TODO:"""
        return self.points(element="faces", backend=backend, **kwargs)

    def face_centers(self, backend=None, **kwargs):
        """TODO:"""
        return self.points(element="faces", backend=backend, **kwargs)

    def edges(self, periodic_elements="exclude", backend=None, **kwargs):
        """TODO:"""

        uxarray.plot.utils.backend.assign(backend)

        if "rasterize" not in kwargs:
            kwargs["rasterize"] = False
        if "projection" not in kwargs:
            kwargs["projection"] = ccrs.PlateCarree()
        if "clabel" not in kwargs:
            kwargs["clabel"] = "edges"
        if "crs" not in kwargs:
            kwargs["crs"] = ccrs.PlateCarree()

        gdf = self._uxgrid.to_geodataframe(periodic_elements=periodic_elements)[
            ["geometry"]
        ]

        return gdf.hvplot.paths(geo=True, **kwargs)

    def mesh(self, periodic_elements="exclude", backend=None, **kwargs):
        """TODO:"""
        return self.edges(periodic_elements, backend, **kwargs)


class UxDataArrayPlotAccessor:
    """Plotting Accessor for UxDataArray, accessed through
    ``UxDataArray.plot()`` or ``UxDataArray.plot.specific_routine()``"""

    _uxda: UxDataArray
    __slots__ = ("_uxda",)

    def __init__(self, uxda: UxDataArray) -> None:
        self._uxda = uxda

    def __call__(self, **kwargs) -> Any:
        if self._uxda._face_centered():
            # polygons for face-centered data
            return self.polygons(**kwargs)
        else:
            # points for node and edge centered data
            return self.points(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """When a function that isn't part of the class is invoked (i.e.
        uxda.plot.hist), an attempt is made to try and call Xarray's
        implementation of that function if it exsists."""

        # reference to xr.DataArray.plot accessor
        xarray_plot_accessor = super(type(self._uxda), self._uxda).plot

        if hasattr(xarray_plot_accessor, name):
            # call xarray plot method if it exists
            # use inline backend to reset configuration if holoviz methods were called before
            uxarray.plot.utils.backend.reset_mpl_backend()
            return getattr(xarray_plot_accessor, name)
        else:
            raise AttributeError(f"Unsupported Plotting Method: '{name}'")

    def polygons(self, periodic_elements="exclude", backend=None, *args, **kwargs):
        uxarray.plot.utils.backend.assign(backend)

        if "rasterize" not in kwargs:
            kwargs["rasterize"] = True
        if "projection" not in kwargs:
            kwargs["projection"] = ccrs.PlateCarree()
        if "clabel" not in kwargs and self._uxda.name is not None:
            kwargs["clabel"] = self._uxda.name
        if "crs" not in kwargs:
            kwargs["crs"] = ccrs.PlateCarree()

        gdf = self._uxda.to_geodataframe(periodic_elements=periodic_elements)

        return gdf.hvplot.polygons(
            c=self._uxda.name if self._uxda.name is not None else "var",
            geo=True,
            *args,
            **kwargs,
        )

    def points(self, backend=None, *args, **kwargs):
        uxarray.plot.utils.backend.assign(backend)

        uxgrid = self._uxda.uxgrid
        data_mapping = self._uxda.data_mapping

        if data_mapping == "nodes":
            lon, lat = uxgrid.node_lon.values, uxgrid.node_lat.values
        elif data_mapping == "faces":
            lon, lat = uxgrid.face_lon.values, uxgrid.face_lat.values
        elif data_mapping == "edges":
            lon, lat = uxgrid.edge_lon.values, uxgrid.edge_lat.values
        else:
            raise ValueError("TODO: ")

        verts = {"lon": lon, "lat": lat, "z": self._uxda.values}

        points_df = pd.DataFrame.from_dict(verts)

        return points_df.hvplot.points("lon", "lat", c="z", *args, **kwargs)

    @functools.wraps(dataarray_plot.rasterize)
    def rasterize(
        self,
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

        warnings.warn(
            "``UxDataArray.plot.rasterize()`` will be deprecated in a future release. Please use "
            "``UxDataArray.plot.polygons(rasterize=True)`` or ``UxDataArray.plot.points(rasterize=True)``",
            DeprecationWarning,
            stacklevel=2,
        )

        return dataarray_plot.rasterize(
            self._uxda,
            method=method,
            backend=backend,
            periodic_elements=periodic_elements,
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
            **kwargs,
        )


class UxDatasetPlotAccessor:
    """Plotting Accessor for UxDataset, accessed through ``UxDataset.plot()``
    or ``UxDataset.plot.specific_routine()``"""

    _uxds: UxDataset
    __slots__ = ("_uxds",)

    def __init__(self, uxds: UxDataset) -> None:
        self._uxds = uxds

    def __call__(self, **kwargs) -> Any:
        raise ValueError(
            "UxDataset.plot cannot be called directly. Use an explicit plot method, "
            "e.g uxds.plot.scatter(...)"
        )

    def __getattr__(self, name: str) -> Any:
        """When a function that isn't part of the class is invoked (i.e.
        uxds.plot.scatter), an attempt is made to try and call Xarray's
        implementation of that function if it exists."""

        # reference to xr.Dataset.plot accessor
        xarray_plot_accessor = super(type(self._uxds), self._uxds).plot

        if hasattr(xarray_plot_accessor, name):
            # call xarray plot method if it exists
            # # use inline backend to reset configuration if holoviz methods were called before
            uxarray.plot.utils.backend.reset_mpl_backend()

            return getattr(xarray_plot_accessor, name)
        else:
            raise AttributeError(f"Unsupported Plotting Method: '{name}'")
