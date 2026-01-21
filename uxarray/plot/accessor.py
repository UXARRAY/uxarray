from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd

import uxarray.plot.utils

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray
    from uxarray.core.dataset import UxDataset
    from uxarray.grid import Grid

import hvplot.pandas
import hvplot.xarray

from uxarray.plot.utils import backend as plotting_backend


class GridPlotAccessor:
    """Plotting accessor for :class:`Grid`.

    You can call :func:`Grid.plot`
    which plots the edges, or you can specify a routine.
    For example:

    - :func:`Grid.plot.edges`
    - :func:`Grid.plot.nodes`
    - :func:`Grid.plot.face_centers`

    See examples in :doc:`/user-guide/plotting`.
    """

    _uxgrid: Grid
    __slots__ = ("_uxgrid",)

    def __init__(self, uxgrid: Grid) -> None:
        self._uxgrid = uxgrid

    def __call__(self, **kwargs) -> Any:
        return self.edges(**kwargs)

    def points(self, element="nodes", backend=None, **kwargs):
        """Generate a point plot based on the specified grid element type
        (nodes, faces, or edges).

        This function retrieves longitude and latitude values for the specified element type
        from the grid (e.g., node, face, or edge locations), converts them into a pandas
        DataFrame, and creates a point plot using `hvplot`. The backend for plotting can
        also be specified, and additional plotting options are accepted through `kwargs`.

        Parameters
        ----------
        element : str, optional, default="nodes"
            The type of grid element for which to retrieve points. Options include:
            - "nodes" or "corner nodes" or "node_latlon" for grid nodes,
            - "faces" or "face centers" or "face_latlon" for grid face centers,
            - "edges" or "edge centers" or "edge_latlon" for grid edges.
        backend : str or None, optional
            Plotting backend to use. One of ['matplotlib', 'bokeh']. Equivalent to running holoviews.extension(backend)
        **kwargs : dict
            Additional keyword arguments passed to `hvplot.points`. For a full list of supported arguments, please
            refer to https://hvplot.holoviz.org/user_guide/Customization.html

        Returns
        -------
        gdf.hvplot.points : hvplot.points
            A point plot of the selected coordinate

        Raises
        ------
        ValueError
            If the provided `element` is not one of the accepted options.
        """

        plotting_backend.assign(backend)

        if element in ["nodes", "corner nodes", "node_latlon"]:
            lon, lat = self._uxgrid.node_lon.values, self._uxgrid.node_lat.values
        elif element in ["faces", "face centers", "face_latlon"]:
            lon, lat = self._uxgrid.face_lon.values, self._uxgrid.face_lat.values
        elif element in ["edges", "edge centers", "edge_latlon"]:
            lon, lat = self._uxgrid.edge_lon.values, self._uxgrid.edge_lat.values
        else:
            raise ValueError(f"Unsupported element {element}")

        verts = {"lon": lon, "lat": lat}

        points_df = pd.DataFrame.from_dict(verts)

        return points_df.hvplot.points("lon", "lat", **kwargs)

    def nodes(self, backend=None, **kwargs):
        """Generate a point plot for the grid corner nodes.

        This function is a convenience wrapper around the `points` method, specifically
        for plotting the grid nodes. It retrieves the longitude and latitude values
        corresponding to the grid nodes and generates a point plot using `hvplot`. The
        backend for plotting can also be specified, and additional plotting options
        are accepted through `kwargs`.

        Parameters
        ----------
        backend : str or None, optional
            Plotting backend to use. One of ['matplotlib', 'bokeh']. Equivalent to running holoviews.extension(backend)
        **kwargs : dict
            Additional keyword arguments passed to `hvplot.points`. For a full list of supported arguments, please
            refer to https://hvplot.holoviz.org/user_guide/Customization.html

        Returns
        -------
        gdf.hvplot.points : hvplot.points
            A point plot of the corner node coordinates
        """

        return self.points(element="nodes", backend=backend, **kwargs)

    def node_coords(self, backend=None, **kwargs):
        return self.points(element="nodes", backend=backend, **kwargs)

    node_coords.__doc__ = nodes.__doc__

    def corner_nodes(self, backend=None, **kwargs):
        return self.points(element="nodes", backend=backend, **kwargs)

    corner_nodes.__doc__ = nodes.__doc__

    def edge_coords(self, backend=None, **kwargs):
        """Wrapper for ``Grid.plot.points(element='edge centers')``

        Parameters
        ----------
        backend : str or None, optional
            Plotting backend to use. One of ['matplotlib', 'bokeh']. Equivalent to running holoviews.extension(backend)
        **kwargs : dict
            Additional keyword arguments passed to `hvplot.points`. For a full list of supported arguments, please
            refer to https://hvplot.holoviz.org/user_guide/Customization.html

        Returns
        -------
        gdf.hvplot.points : hvplot.points
            A point plot of the edge center coordinates
        """
        return self.points(element="edges", backend=backend, **kwargs)

    def edge_centers(self, backend=None, **kwargs):
        return self.points(element="edges", backend=backend, **kwargs)

    edge_centers.__doc__ = edge_coords.__doc__

    def face_coords(self, backend=None, **kwargs):
        """Wrapper for ``Grid.plot.points(element='face centers')``

        Parameters
        ----------
        backend : str or None, optional
            Plotting backend to use. One of ['matplotlib', 'bokeh']. Equivalent to running holoviews.extension(backend)
        **kwargs : dict
            Additional keyword arguments passed to `hvplot.points`. For a full list of supported arguments, please
            refer to https://hvplot.holoviz.org/user_guide/Customization.html

        Returns
        -------
        gdf.hvplot.points : hvplot.points
            A point plot of the face center coordinates
        """
        return self.points(element="faces", backend=backend, **kwargs)

    def face_centers(self, backend=None, **kwargs):
        return self.points(element="faces", backend=backend, **kwargs)

    face_centers.__doc__ = face_coords.__doc__

    def edges(
        self,
        periodic_elements="exclude",
        backend=None,
        engine="spatialpandas",
        **kwargs,
    ):
        """Plots the edges of a Grid.

        This function plots the edges of the grid as geographical paths using `hvplot`.
        The plot can ignore, exclude, or split periodic elements based on the provided option.
        It automatically sets default values for rasterization, projection, and labeling,
        which can be overridden by passing additional keyword arguments. The backend for
        plotting can also be specified.

        Parameters
        ----------
        periodic_elements : str, optional, default="exclude"
            Specifies whether to include or exclude periodic elements in the grid.
            Options are:
            - "exclude": Exclude periodic elements,
            - "ignore": Include periodic elements without any corrections
            - "split": Split periodic elements.
        backend : str or None, optional
            Plotting backend to use. One of ['matplotlib', 'bokeh']. Equivalent to running holoviews.extension(backend)
        engine: str, optional
            Engine to use for GeoDataFrame construction. One of ['spatialpandas', 'geopandas']
        **kwargs : dict
            Additional keyword arguments passed to `hvplot.paths`. These can include:
            - "rasterize" (bool): Whether to rasterize the plot (default: False),
            - "projection" (ccrs.Projection): The map projection to use (default: `ccrs.PlateCarree()`),
            - "clabel" (str): Label for the edges (default: "edges"),
            - "crs" (ccrs.Projection): Coordinate reference system for the plot (default: `ccrs.PlateCarree()`).
            A full list can be found at https://hvplot.holoviz.org/user_guide/Customization.html

        Returns
        -------
        gdf.hvplot.paths : hvplot.paths
            A paths plot of the edges of the unstructured grid
        """
        import cartopy.crs as ccrs

        plotting_backend.assign(backend)

        if "rasterize" not in kwargs:
            kwargs["rasterize"] = False
        if "projection" not in kwargs:
            kwargs["projection"] = ccrs.PlateCarree()
        if "clabel" not in kwargs:
            kwargs["clabel"] = "edges"
        if "crs" not in kwargs:
            if "projection" in kwargs:
                central_longitude = kwargs["projection"].proj4_params["lon_0"]
            else:
                central_longitude = 0.0
            kwargs["crs"] = ccrs.PlateCarree(central_longitude=central_longitude)

        gdf = self._uxgrid.to_geodataframe(
            periodic_elements=periodic_elements,
            projection=kwargs.get("projection"),
            engine=engine,
            project=False,
        )

        return gdf.hvplot.paths(geo=True, **kwargs)

    def mesh(self, periodic_elements="exclude", backend=None, **kwargs):
        return self.edges(periodic_elements, backend, **kwargs)

    mesh.__doc__ = edges.__doc__

    def face_degree_distribution(
        self,
        backend=None,
        xlabel="Number of Nodes per Face",
        ylabel="Count",
        title="Face Degree Distribution",
        show_grid=True,
        xaxis="bottom",
        yaxis="left",
        xrotation=0,
        **kwargs,
    ):
        """Plots the distribution of the number of nodes per face as a bar
        plot."""

        plotting_backend.assign(backend)

        n_nodes_per_face = self._uxgrid.n_nodes_per_face.values

        nodes_series = pd.Series(n_nodes_per_face, name="number_of_nodes")
        counts = nodes_series.value_counts().sort_index()
        df = counts.reset_index()
        df.columns = ["number_of_nodes", "count"]
        df["number_of_nodes"] = df["number_of_nodes"].astype(str)

        return df.hvplot.bar(
            x="number_of_nodes",
            y="count",
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            **kwargs,
        ).opts(
            xrotation=xrotation,
            show_grid=show_grid,
            xaxis=xaxis,
            yaxis=yaxis,
        )

    def face_area_distribution(
        self,
        backend=None,
        xlabel="Face Area",
        ylabel="Count",
        title="Face Area Distribution",
        show_grid=True,
        xaxis="bottom",
        yaxis="left",
        xrotation=0,
        bins=30,  # Number of bins for the histogram
        **kwargs,
    ):
        """Plots a histogram of the face areas using hvplot."""
        # Assign the plotting backend if provided
        plotting_backend.assign(backend)

        # Extract face areas from the grid
        face_areas = self._uxgrid.face_areas.values

        # Create a pandas Series from face areas
        face_areas_series = pd.Series(face_areas, name="face_area")

        # Plot the histogram using hvplot
        histogram = face_areas_series.hvplot.hist(
            bins=bins, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs
        ).opts(
            xrotation=xrotation,
            show_grid=show_grid,
            xaxis=xaxis,
            yaxis=yaxis,
        )

        return histogram


class UxDataArrayPlotAccessor:
    """Plotting Accessor for :class:`UxDataArray`.

    You can call :func:`UxDataArray.plot`
    which auto-selects polygons for face-centered data
    or points for node- and edge-centered data,
    or you can specify a routine:

    - :func:`UxDataArray.plot.polygons`
    - :func:`UxDataArray.plot.points`
    - :func:`UxDataArray.plot.line`
    - :func:`UxDataArray.plot.scatter`

    See examples in :doc:`/user-guide/plotting`.
    """

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

    def polygons(
        self,
        periodic_elements: str | None = "exclude",
        backend: str | None = None,
        engine: str | None = "spatialpandas",
        rasterize: bool | None = True,
        dynamic: bool | None = False,
        projection=None,
        xlabel: str | None = "Longitude",
        ylabel: str | None = "Latitude",
        *args,
        **kwargs,
    ):
        """Generated a shaded polygon plot.

        This function plots the faces of an unstructured grid shaded with a face-centered data variable using hvplot.
        It allows for rasterization, projection settings, and labeling of the data variable to be
        customized through keyword arguments. The backend for plotting can also be specified.
        If a data array has a name, its name is used for color label.

        Parameters
        ----------
        periodic_elements : str, optional, default="exclude"
            Specifies whether to include or exclude periodic elements in the grid.
            Options are:
            - "exclude": Exclude periodic elements,
            - "split": Split periodic elements.
            - "ignore": Include periodic elements without any corrections
        backend : str or None, optional
            Plotting backend to use. One of ['matplotlib', 'bokeh']. Equivalent to running holoviews.extension(backend)
        engine: str, optional
            Engine to use for GeoDataFrame construction. One of ['spatialpandas', 'geopandas']
        rasterize: bool, optional
            Whether to rasterize the plot (default: True)
        projection: ccrs.Projection, optional
            The map projection to use.
        *args : tuple
            Additional positional arguments to be passed to `hvplot.polygons`.
        **kwargs : dict
            Additional keyword arguments passed to `hvplot.polygons`. For additional customization, please refer to https://hvplot.holoviz.org/user_guide/Customization.html

        Returns
        -------
        gdf.hvplot.polygons : hvplot.polygons
            A shaded polygon plot
        """
        import cartopy.crs as ccrs

        plotting_backend.assign(backend)

        if dynamic and (projection is not None or kwargs.get("geo", None) is True):
            warnings.warn(
                "Projections with dynamic plots may display incorrectly or update improperly. "
                "Consider using static plots instead. See: github.com/holoviz/geoviews/issues/762"
            )

        if projection is not None:
            kwargs["projection"] = projection
            kwargs["geo"] = True
            if "crs" not in kwargs:
                central_longitude = projection.proj4_params["lon_0"]
                kwargs["crs"] = ccrs.PlateCarree(central_longitude=central_longitude)

        if "clabel" not in kwargs and self._uxda.name is not None:
            kwargs["clabel"] = self._uxda.name

        gdf = self._uxda.to_geodataframe(
            periodic_elements=periodic_elements,
            projection=kwargs.get("projection"),
            engine=engine,
            project=False,
        )

        return gdf.hvplot.polygons(
            c=self._uxda.name if self._uxda.name is not None else "var",
            rasterize=rasterize,
            dynamic=dynamic,
            xlabel=xlabel,
            ylabel=ylabel,
            *args,
            **kwargs,
        )

    def points(self, backend=None, *args, **kwargs):
        """Generate a point plot based on the specified grid element type
        (nodes, faces, or edges) shaded with the data mapped to those elements.

        This function retrieves longitude and latitude values for the specified element type
        from the grid (e.g., node, face, or edge locations), converts them into a pandas
        DataFrame, and creates a point plot using `hvplot`. The points are shaded with the data that is mapped
        to the selected element.

        Parameters
        ----------
        backend : str or None, optional
            Plotting backend to use. One of ['matplotlib', 'bokeh']. Equivalent to running holoviews.extension(backend)
        **kwargs : dict
            Additional keyword arguments passed to `hvplot.points`. For a full list of supported arguments, please
            refer to https://hvplot.holoviz.org/user_guide/Customization.html

        Returns
        -------
        gdf.hvplot.points : hvplot.points
            A shaded point plot

        Raises
        ------
        ValueError
            If the data is not mapped to the nodes, edges, or faces.
        """

        plotting_backend.assign(backend)

        uxgrid = self._uxda.uxgrid
        data_mapping = self._uxda.data_mapping

        if data_mapping == "nodes":
            lon, lat = uxgrid.node_lon.values, uxgrid.node_lat.values
        elif data_mapping == "faces":
            lon, lat = uxgrid.face_lon.values, uxgrid.face_lat.values
        elif data_mapping == "edges":
            lon, lat = uxgrid.edge_lon.values, uxgrid.edge_lat.values
        else:
            raise ValueError(
                "Data is not mapped to the nodes, edges, or faces of the grid."
            )

        verts = {"lon": lon, "lat": lat, "z": self._uxda.values}

        points_df = pd.DataFrame.from_dict(verts)

        return points_df.hvplot.points("lon", "lat", c="z", *args, **kwargs)

    def line(self, backend=None, *args, **kwargs):
        """Wrapper for ``hvplot.line()``"""

        plotting_backend.assign(backend)
        da = self._uxda.to_xarray()
        return da.hvplot.line(*args, **kwargs)

    def scatter(self, backend=None, *args, **kwargs):
        """Wrapper for ``hvplot.scatter()``"""

        plotting_backend.assign(backend)
        da = self._uxda.to_xarray()
        return da.hvplot.scatter(*args, **kwargs)


class UxDatasetPlotAccessor:
    """Plotting accessor for :class:`UxDataset`."""

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
