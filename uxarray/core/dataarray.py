from __future__ import annotations

import warnings
from html import escape
from typing import TYPE_CHECKING, Any, Hashable, Literal, Mapping
from warnings import warn

import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from xarray.core import dtypes
from xarray.core.options import OPTIONS
from xarray.core.utils import UncachedAccessor

import uxarray
from uxarray.core.aggregation import _uxda_grid_aggregate
from uxarray.core.gradient import (
    _calculate_edge_face_difference,
    _calculate_edge_node_difference,
    _compute_gradient,
)
from uxarray.core.utils import _map_dims_to_ugrid
from uxarray.core.zonal import (
    _compute_conservative_zonal_mean_bands,
    _compute_non_conservative_zonal_mean,
)
from uxarray.cross_sections import UxDataArrayCrossSectionAccessor
from uxarray.formatting_html import array_repr
from uxarray.grid import Grid
from uxarray.grid.dual import construct_dual
from uxarray.grid.validation import _check_duplicate_nodes_indices
from uxarray.io._healpix import get_zoom_from_cells
from uxarray.plot.accessor import UxDataArrayPlotAccessor
from uxarray.remap.accessor import RemapAccessor
from uxarray.subset import DataArraySubsetAccessor

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset


class UxDataArray(xr.DataArray):
    """Grid informed ``xarray.DataArray`` with an attached ``Grid`` accessor
    and grid-specific functionality.

    Parameters
    ----------
    uxgrid : uxarray.Grid, optional
        The `Grid` object that makes this array aware of the unstructured
        grid topology it belongs to.
        If `None`, it needs to be an instance of `uxarray.Grid`.

    Other Parameters
    ----------------
    *args:
        Arguments for the ``xarray.DataArray`` class
    **kwargs:
        Keyword arguments for the ``xarray.DataArray`` class

    Notes
    -----
    See `xarray.DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`__
    for further information about DataArrays.

    Grid-Aware Accessor Methods
    ---------------------------
    The following methods return specialized accessors that preserve grid information:

    - ``groupby``: Groups data by dimension/coordinate
    - ``groupby_bins``: Groups data by bins
    - ``resample``: Resamples timeseries data
    - ``rolling``: Rolling window operations
    - ``coarsen``: Coarsens data by integer factors
    - ``weighted``: Weighted operations
    - ``rolling_exp``: Exponentially weighted rolling (requires numbagg)
    - ``cumulative``: Cumulative operations

    All these methods work identically to xarray but maintain the uxgrid attribute
    throughout operations.
    """

    # expected instance attributes, required for subclassing with xarray (as of v0.13.0)
    __slots__ = ("_uxgrid",)

    def __init__(self, *args, uxgrid: Grid = None, **kwargs):
        self._uxgrid = None

        if uxgrid is not None and not isinstance(uxgrid, Grid):
            raise RuntimeError(
                "uxarray.UxDataArray.__init__: uxgrid can be either None or "
                "an instance of the uxarray.Grid class"
            )
        else:
            self.uxgrid = uxgrid

        super().__init__(*args, **kwargs)

    # declare various accessors
    plot = UncachedAccessor(UxDataArrayPlotAccessor)
    subset = UncachedAccessor(DataArraySubsetAccessor)
    remap = UncachedAccessor(RemapAccessor)
    cross_section = UncachedAccessor(UxDataArrayCrossSectionAccessor)

    def _repr_html_(self) -> str:
        if OPTIONS["display_style"] == "text":
            return f"<pre>{escape(repr(self))}</pre>"
        return array_repr(self)

    @classmethod
    def _construct_direct(cls, *args, **kwargs):
        """Override to make the result a ``uxarray.UxDataArray`` class."""
        return cls(xr.DataArray._construct_direct(*args, **kwargs))

    def _copy(self, **kwargs):
        """Override to make the result a complete instance of
        ``uxarray.UxDataArray``."""
        copied = super()._copy(**kwargs)

        deep = kwargs.get("deep", None)

        if deep:
            # Reinitialize the uxgrid assessor
            copied.uxgrid = self.uxgrid.copy()  # deep copy
        else:
            # Point to the existing uxgrid object
            copied.uxgrid = self.uxgrid

        return copied

    def _replace(self, *args, **kwargs):
        """Override to make the result a complete instance of
        ``uxarray.UxDataArray``."""
        da = super()._replace(*args, **kwargs)

        if isinstance(da, UxDataArray):
            da.uxgrid = self.uxgrid
        else:
            da = UxDataArray(da, uxgrid=self.uxgrid)

        return da

    @property
    def uxgrid(self):
        """Linked ``Grid`` representing to the unstructured grid the data
        resides on."""

        return self._uxgrid

    # a setter function
    @uxgrid.setter
    def uxgrid(self, ugrid_obj):
        self._uxgrid = ugrid_obj

    @property
    def data_mapping(self):
        """Returns which unstructured grid a data variable is mapped to."""
        if self._face_centered():
            return "faces"
        elif self._edge_centered():
            return "edges"
        elif self._node_centered():
            return "nodes"
        else:
            return None

    def to_geodataframe(
        self,
        periodic_elements: str | None = "exclude",
        projection=None,
        cache: bool | None = True,
        override: bool | None = False,
        engine: str | None = "spatialpandas",
        exclude_antimeridian: bool | None = None,
        **kwargs,
    ):
        """Constructs a ``GeoDataFrame`` consisting of polygons representing
        the faces of the current ``Grid`` with a face-centered data variable
        mapped to them.

        Periodic polygons (i.e. those that cross the antimeridian) can be handled using the ``periodic_elements``
        parameter. Setting ``periodic_elements='split'`` will split each periodic polygon along the antimeridian.
        Setting ``periodic_elements='exclude'`` will exclude any periodic polygon from the computed GeoDataFrame.
        Setting ``periodic_elements='ignore'`` will compute the GeoDataFrame assuming no corrections are needed, which
        is best used for grids that do not initially include any periodic polygons.

        Parameters
        ----------
        periodic_elements : str, optional
            Method for handling periodic elements. One of ['exclude', 'split', or 'ignore']:
            - 'exclude': Periodic elements will be identified and excluded from the GeoDataFrame
            - 'split': Periodic elements will be identified and split using the ``antimeridian`` package
            - 'ignore': No processing will be applied to periodic elements.
        projection: ccrs.Projection, optional
            Geographic projection used to transform polygons. Only supported when periodic_elements is set to
            'ignore' or 'exclude'
        cache: bool, optional
            Flag used to select whether to cache the computed GeoDataFrame
        override: bool, optional
            Flag used to select whether to ignore any cached GeoDataFrame
        engine: str, optional
            Selects what library to use for creating a GeoDataFrame. One of ['spatialpandas', 'geopandas']. Defaults
            to spatialpandas
        exclude_antimeridian: bool, optional
            Flag used to select whether to exclude polygons that cross the antimeridian (Will be deprecated)

        Returns
        -------
        gdf : spatialpandas.GeoDataFrame or geopandas.GeoDataFrame
            The output ``GeoDataFrame`` with a filled out "geometry" column of polygons and a data column with the
            same name as the ``UxDataArray`` (or named ``var`` if no name exists)
        """

        if self.values.ndim > 1:
            # data is multidimensional, must be a 1D slice
            raise ValueError(
                f"Data Variable must be 1-dimensional, with shape {self.uxgrid.n_face} "
                f"for face-centered data."
            )

        if self.values.size == self.uxgrid.n_face:
            gdf, non_nan_polygon_indices = self.uxgrid.to_geodataframe(
                periodic_elements=periodic_elements,
                projection=projection,
                project=kwargs.get("project", True),
                cache=cache,
                override=override,
                exclude_antimeridian=exclude_antimeridian,
                return_non_nan_polygon_indices=True,
                engine=engine,
            )

            if exclude_antimeridian is not None:
                if exclude_antimeridian:
                    periodic_elements = "exclude"
                else:
                    periodic_elements = "split"

            # set a default variable name if the data array is not named
            var_name = self.name if self.name is not None else "var"

            if periodic_elements == "exclude":
                # index data to ignore data mapped to periodic elements
                _data = np.delete(
                    self.values,
                    self.uxgrid._gdf_cached_parameters["antimeridian_face_indices"],
                    axis=0,
                )
            else:
                _data = self.values

            if non_nan_polygon_indices is not None:
                # index data to ignore NaN polygons
                _data = _data[non_nan_polygon_indices]

            gdf[var_name] = _data

        elif self.values.size == self.uxgrid.n_node:
            raise ValueError(
                f"Data Variable with size {self.values.size} does not match the number of faces "
                f"({self.uxgrid.n_face}. Current size matches the number of nodes. Consider running "
                f"``UxDataArray.topological_mean(destination='face') to aggregate the data onto the faces."
            )
        elif self.values.size == self.uxgrid.n_edge:
            raise ValueError(
                f"Data Variable with size {self.values.size} does not match the number of faces "
                f"({self.uxgrid.n_face}. Current size matches the number of edges."
            )
        else:
            # data is not mapped to
            raise ValueError(
                f"Data Variable with size {self.values.size} does not match the number of faces "
                f"({self.uxgrid.n_face}."
            )

        return gdf

    def to_polycollection(
        self,
        **kwargs,
    ):
        """Constructs a ``matplotlib.collections.PolyCollection``` consisting
        of polygons representing the faces of the current ``UxDataArray`` with
        a face-centered data variable mapped to them.

        Parameters
        ----------
        **kwargs: dict
            Key word arguments to pass into the construction of a PolyCollection
        """
        # data is multidimensional, must be a 1D slice
        if self.values.ndim > 1:
            raise ValueError(
                f"Data Variable must be 1-dimensional, with shape {self.uxgrid.n_face} "
                f"for face-centered data."
            )

        poly_collection = self.uxgrid.to_polycollection(**kwargs)

        poly_collection.set_array(self.data)

        return poly_collection

    def to_raster(
        self,
        ax: GeoAxes,
        *,
        pixel_ratio: float | None = None,
        pixel_mapping: xr.DataArray | np.ndarray | None = None,
        return_pixel_mapping: bool = False,
    ):
        """
        Rasterizes a data variable stored on the faces of an unstructured grid onto the pixels of the provided Cartopy GeoAxes.

        Parameters
        ----------
        ax : GeoAxes
            A Cartopy :class:`~cartopy.mpl.geoaxes.GeoAxes` onto which the data will be rasterized.
            Each pixel in this axes will be sampled against the unstructured grid's face geometry.
        pixel_ratio : float, default=1.0
            A scaling factor to adjust the resolution of the rasterization.
            A value greater than 1 increases the resolution (sharpens the image),
            while a value less than 1 will result in a coarser rasterization.
            The resolution also depends on what the figure's DPI setting is
            prior to calling :meth:`to_raster`.
            You can control DPI with the ``dpi`` keyword argument when creating the figure,
            or by using :meth:`~matplotlib.figure.Figure.set_dpi` after creation.
        pixel_mapping : xr.DataArray or array-like, optional
            Precomputed mapping from pixels within the Cartopy GeoAxes boundary
            to grid face indices (1-dimensional).
        return_pixel_mapping : bool, default=False
            If ``True``, the pixel mapping will be returned in addition to the raster,
            and then you can pass it via the `pixel_mapping` parameter for future rasterizations
            using the same or equivalent :attr:`uxgrid` and `ax`.
            Note that this is also specific to the pixel ratio setting.

        Returns
        -------
        raster : numpy.ndarray, shape (ny, nx)
            Array of resampled data values corresponding to each pixel.
        pixel_mapping : xr.DataArray, shape (n,)
            If ``return_pixel_mapping=True``, the computed pixel mapping is returned
            so that you can reuse it.
            Axes and pixel ratio info are included as attributes.

        Notes
        -----
        - This method currently employs a nearest-neighbor resampling approach. For every pixel in the GeoAxes,
          it finds the face of the unstructured grid that contains the pixel's geographic coordinate and colors
          that pixel with the face's data value.
        - If a pixel does not intersect any face (i.e., lies outside the grid domain),
          it will be left empty (transparent).

        Examples
        --------
        >>> import cartopy.crs as ccrs
        >>> import matplotlib.pyplot as plt

        Create a :class:`~cartopy.mpl.geoaxes.GeoAxes` with a Robinson projection and global extent

        >>> fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
        >>> ax.set_global()

        Rasterize data onto the GeoAxes

        >>> raster = uxds["psi"].to_raster(ax=ax)

        Use :meth:`~cartopy.mpl.geoaxes.GeoAxes.imshow` to visualize the raster

        >>> ax.imshow(raster, origin="lower", extent=ax.get_xlim() + ax.get_ylim())

        """
        from uxarray.constants import INT_DTYPE
        from uxarray.plot.matplotlib import (
            _ensure_dimensions,
            _nearest_neighbor_resample,
            _RasterAxAttrs,
        )

        data = _ensure_dimensions(self)

        if not isinstance(ax, GeoAxes):
            raise TypeError("`ax` must be an instance of cartopy.mpl.geoaxes.GeoAxes")

        pixel_ratio_set = pixel_ratio is not None
        if not pixel_ratio_set:
            pixel_ratio = 1.0
        if pixel_mapping is not None:
            input_ax_attrs = _RasterAxAttrs.from_ax(ax, pixel_ratio=pixel_ratio)
            if isinstance(pixel_mapping, xr.DataArray):
                pixel_ratio_input = pixel_ratio
                pixel_ratio = pixel_mapping.attrs["pixel_ratio"]
                if pixel_ratio_set and pixel_ratio_input != pixel_ratio:
                    warn(
                        "Pixel ratio mismatch: "
                        f"{pixel_ratio_input} passed but {pixel_ratio} in pixel_mapping. "
                        "Using the pixel_mapping attribute.",
                        stacklevel=2,
                    )
                input_ax_attrs = _RasterAxAttrs.from_ax(ax, pixel_ratio=pixel_ratio)
                pm_ax_attrs = _RasterAxAttrs.from_xr_attrs(pixel_mapping.attrs)
                if input_ax_attrs != pm_ax_attrs:
                    raise ValueError(
                        "Pixel mapping incompatible with ax. "
                        + input_ax_attrs._value_comparison_message(pm_ax_attrs)
                    )
            pixel_mapping = np.asarray(pixel_mapping, dtype=INT_DTYPE)
        else:

            def _is_default_extent() -> bool:
                # Default extents are indicated by xlim/ylim being (0, 1)
                # when autoscale is still on (no extent has been explicitly set)
                if not ax.get_autoscale_on():
                    return False
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                return np.allclose(xlim, (0.0, 1.0)) and np.allclose(ylim, (0.0, 1.0))

            if _is_default_extent():
                try:
                    import cartopy.crs as ccrs

                    lon_min = float(self.uxgrid.node_lon.min(skipna=True).values)
                    lon_max = float(self.uxgrid.node_lon.max(skipna=True).values)
                    lat_min = float(self.uxgrid.node_lat.min(skipna=True).values)
                    lat_max = float(self.uxgrid.node_lat.max(skipna=True).values)
                    ax.set_extent(
                        (lon_min, lon_max, lat_min, lat_max),
                        crs=ccrs.PlateCarree(),
                    )
                    warn(
                        "Axes extent was default; auto-setting from grid lon/lat bounds for rasterization. "
                        "Set the extent explicitly to control this, e.g. via ax.set_global(), "
                        "ax.set_extent(...), or ax.set_xlim(...) + ax.set_ylim(...).",
                        stacklevel=2,
                    )
                except Exception as e:
                    warn(
                        f"Failed to auto-set extent from grid bounds: {e}",
                        stacklevel=2,
                    )
            input_ax_attrs = _RasterAxAttrs.from_ax(ax, pixel_ratio=pixel_ratio)

        raster, pixel_mapping_np = _nearest_neighbor_resample(
            data,
            ax,
            pixel_ratio=pixel_ratio,
            pixel_mapping=pixel_mapping,
        )
        if return_pixel_mapping:
            pixel_mapping_da = xr.DataArray(
                pixel_mapping_np,
                name="pixel_mapping",
                dims=("n_pixel",),
                attrs={
                    "long_name": "pixel_mapping",
                    "description": (
                        "Mapping from raster pixels within a Cartopy GeoAxes "
                        "to nearest grid face index."
                    ),
                    **input_ax_attrs.to_xr_attrs(),
                },
            )
            return raster, pixel_mapping_da
        else:
            return raster

    def to_dataset(
        self,
        dim: Hashable = None,
        *,
        name: Hashable = None,
        promote_attrs: bool = False,
    ) -> UxDataset:
        """Convert a ``UxDataArray`` to a ``UxDataset``.

        Parameters
        ----------
        dim : Hashable, optional
            Name of the dimension on this array along which to split this array
            into separate variables. If not provided, this array is converted
            into a Dataset of one variable.
        name : Hashable, optional
            Name to substitute for this array's name. Only valid if ``dim`` is
            not provided.
        promote_attrs : bool, default: False
            Set to True to shallow copy attrs of UxDataArray to returned UxDataset.

        Returns
        -------
        uxds: UxDataSet
        """
        xrds = super().to_dataset(dim=dim, name=name, promote_attrs=promote_attrs)
        uxds = uxarray.core.dataset.UxDataset(xrds, uxgrid=self.uxgrid)

        return uxds

    def to_xarray(self):
        return xr.DataArray(self)

    def integrate(
        self, quadrature_rule: str | None = "triangular", order: int | None = 4
    ) -> UxDataArray:
        """Computes the integral of a data variable.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        uxda : UxDataArray
            UxDataArray containing the integrated data variable

        Examples
        --------
        Open a Uxarray dataset and compute the integral

        >>> import uxarray as ux
        >>> uxds = ux.open_dataset("grid.ug", "centroid_pressure_data_ug")
        >>> integral = uxds["psi"].integrate()
        """
        if self.values.shape[-1] == self.uxgrid.n_face:
            face_areas = self.uxgrid.face_areas.values

            # perform dot product between face areas and last dimension of data
            integral = np.einsum("i,...i", face_areas, self.values)

        elif self.values.shape[-1] == self.uxgrid.n_node:
            raise ValueError("Integrating data mapped to each node not yet supported.")

        elif self.values.shape[-1] == self.uxgrid.n_edge:
            raise ValueError("Integrating data mapped to each edge not yet supported.")

        else:
            raise ValueError(
                f"The final dimension of the data variable does not match the number of nodes, edges, "
                f"or faces. Expected one of "
                f"{self.uxgrid.n_node}, {self.uxgrid.n_edge}, or {self.uxgrid.n_face}, "
                f"but received {self.values.shape[-1]}"
            )

        # construct a uxda with integrated quantity
        uxda = UxDataArray(
            integral, uxgrid=self.uxgrid, dims=self.dims[:-1], name=self.name
        )

        return uxda

    def zonal_mean(self, lat=(-90, 90, 10), conservative: bool = False, **kwargs):
        """Compute non-conservative or conservative averages of a face-centered variable along lines of constant latitude or latitude bands.

        A zonal mean in UXarray operates differently depending on the ``conservative`` flag:

        - **Non-conservative**: Calculates the mean by sampling face values at specific latitude lines and weighting each contribution by the length of the line where each face intersects that latitude.
        - **Conservative**: Preserves integral quantities by calculating the mean by sampling face values within latitude bands and weighting contributions by their area overlap with latitude bands.

        Parameters
        ----------
        lat : tuple, float, or array-like, default=(-90, 90, 10)
            Latitude specification:
                - tuple (start, end, step): For non-conservative, computes means at intervals of `step`.
                For conservative, creates band edges via np.arange(start, end+step, step).
                - float: Single latitude for non-conservative averaging
                - array-like: For non-conservative, latitudes to sample. For conservative, band edges.
        conservative : bool, default=False
            If True, performs conservative (area-weighted) zonal averaging over latitude bands.
            If False, performs non-conservative (intersection-weighted) averaging at latitude lines.

        Returns
        -------
        UxDataArray
            Contains zonal means with a new 'latitudes' dimension and corresponding coordinates.
            Name will be original_name + '_zonal_mean' or 'zonal_mean' if unnamed.

        Examples
        --------
        # Non-conservative averaging from -90° to 90° at 10° intervals by default
        >>> uxds["var"].zonal_mean()

        # Single latitude (non-conservative) over 30° latitude
        >>> uxds["var"].zonal_mean(lat=30.0)

        # Conservative averaging over latitude bands
        >>> uxds["var"].zonal_mean(lat=(-60, 60, 10), conservative=True)

        # Conservative with explicit band edges
        >>> uxds["var"].zonal_mean(lat=[-90, -30, 0, 30, 90], conservative=True)

        Notes
        -----
        Only supported for face-centered data variables.

        Conservative averaging preserves integral quantities and is recommended for
        physical analysis. Non-conservative averaging samples at latitude lines.
        """
        if not self._face_centered():
            raise ValueError(
                "Zonal mean computations are currently only supported for face-centered data variables."
            )

        face_axis = self.dims.index("n_face")

        if not conservative:
            # Non-conservative (traditional) zonal averaging
            if isinstance(lat, tuple):
                start, end, step = lat
                if step <= 0:
                    raise ValueError("Step size must be positive.")
                if step < 0.1:
                    warnings.warn(
                        f"Very small step size ({step}°) may lead to performance issues...",
                        UserWarning,
                        stacklevel=2,
                    )
                num_points = int(round((end - start) / step)) + 1
                latitudes = np.linspace(start, end, num_points)
                latitudes = np.clip(latitudes, -90, 90)
            elif isinstance(lat, (float, int)):
                latitudes = [lat]
            elif isinstance(lat, (list, np.ndarray)):
                latitudes = np.asarray(lat)
            else:
                raise ValueError(
                    "Invalid value for 'lat' provided. Must be a scalar, tuple (min_lat, max_lat, step), or array-like."
                )

            res = _compute_non_conservative_zonal_mean(
                uxda=self, latitudes=latitudes, **kwargs
            )

            dims = list(self.dims)
            dims[face_axis] = "latitudes"

            return xr.DataArray(
                res,
                dims=dims,
                coords={"latitudes": latitudes},
                name=self.name + "_zonal_mean"
                if self.name is not None
                else "zonal_mean",
                attrs={"zonal_mean": True, "conservative": False},
            )

        else:
            # Conservative zonal averaging
            if isinstance(lat, tuple):
                start, end, step = lat
                if step <= 0:
                    raise ValueError(
                        "Step size must be positive for conservative averaging."
                    )
                if step < 0.1:
                    warnings.warn(
                        f"Very small step size ({step}°) may lead to performance issues...",
                        UserWarning,
                        stacklevel=2,
                    )
                num_points = int(round((end - start) / step)) + 1
                edges = np.linspace(start, end, num_points)
                edges = np.clip(edges, -90, 90)
            elif isinstance(lat, (list, np.ndarray)):
                edges = np.asarray(lat, dtype=float)
            else:
                raise ValueError(
                    "For conservative averaging, 'lat' must be a tuple (start, end, step) or array-like band edges."
                )

            if edges.ndim != 1 or edges.size < 2:
                raise ValueError("Band edges must be 1D with at least two values")

            res = _compute_conservative_zonal_mean_bands(self, edges)

            # Use band centers as coordinate values
            centers = 0.5 * (edges[:-1] + edges[1:])

            dims = list(self.dims)
            dims[face_axis] = "latitudes"

            return xr.DataArray(
                res,
                dims=dims,
                coords={"latitudes": centers},
                name=self.name + "_zonal_mean"
                if self.name is not None
                else "zonal_mean",
                attrs={
                    "zonal_mean": True,
                    "conservative": True,
                    "lat_band_edges": edges,
                },
            )

    def zonal_average(self, lat=(-90, 90, 10), conservative: bool = False, **kwargs):
        """Alias of zonal_mean; prefer `zonal_mean` for primary API."""
        return self.zonal_mean(lat=lat, conservative=conservative, **kwargs)

    def azimuthal_mean(
        self,
        center_coord,
        outer_radius: int | float,
        radius_step: int | float,
        return_hit_counts: bool = False,
    ):
        """Compute averages along circles of constant great-circle distance from a point.

        Parameters
        ----------
        center_coord: tuple, list, ndarray
            Longitude and latitude of the center of the bounding circle
        outer_radius: scalar, int, float
            The maximum radius, in great-circle degrees, at which the azimuthal mean will be computed.
        radius_step: scalar, int, float
            Means will be computed at intervals of `radius_step` on the interval [0, outer_radius]
        return_hit_counts: bool, false
            Indicates whether to return the number of hits at each radius

        Returns
        -------
        azimuthal_mean: xr.DataArray
            Contains a variable with a dimension 'radius' corresponding to the azimuthal average.
        hit_counts: xr.DataArray
            The number of hits at each radius


        Examples
        --------
        # Range from 0° to 5° at 0.5° intervals, around the central point lon,lat=10,50
        >>> az = uxds["var"].azimuthal_mean(
        ...     center_coord=(10, 50), outer_radius=5.0, radius_step=0.5
        ... )
        >>> az.plot(title="Azimuthal Mean")

        Notes
        -----
        Only supported for face-centered data variables. Candidate faces are determined
        using bounding circles - for radii = [r1, r2, r3, ...] faces whose centers lie at distance d,
        r2 < d <= r3 are included in calculations for r3.
        """
        from uxarray.grid.coordinates import _lonlat_rad_to_xyz

        if not self._face_centered():
            raise ValueError(
                "Azimuthal mean computations are currently only supported for face-centered data variables."
            )

        if outer_radius <= 0:
            raise ValueError("Radius must be a positive scalar.")

        kdtree = self.uxgrid._get_scipy_kd_tree()

        lon_deg, lat_deg = map(float, np.asarray(center_coord))
        center_xyz = np.array(
            _lonlat_rad_to_xyz(np.deg2rad(lon_deg), np.deg2rad(lat_deg))
        )

        radii_deg = np.arange(0.0, outer_radius + radius_step, radius_step, dtype=float)
        radii_rad = np.deg2rad(radii_deg)
        chord_radii = 2.0 * np.sin(radii_rad / 2.0)

        faces_processed = np.array([], dtype=np.int_)
        means = np.full(
            (radii_deg.size, *self.to_xarray().isel(drop=True, n_face=0).shape), np.nan
        )
        hit_count = np.zeros_like(radii_deg, dtype=np.int_)

        for ii, r_chord in enumerate(chord_radii):
            # indices of faces within the bounding circle for this radius
            within = np.array(
                kdtree.query_ball_point(center_xyz, r_chord), dtype=np.int_
            )
            if within.size:
                within.sort()

            # include only the new ring: r_(i-1) < d <= r_i
            faces_in_bin = np.setdiff1d(within, faces_processed, assume_unique=True)
            hit_count[ii] = faces_in_bin.size

            if hit_count[ii] == 0:
                continue

            faces_processed = within  # cumulative set for next iteration

            tpose = self.isel(n_face=faces_in_bin).transpose(..., "n_face")
            means[ii, ...] = tpose.weighted_mean().data

        # swap the leading 'radius' axis into the former n_face position
        face_axis = self.dims.index("n_face")
        dims = list(self.dims)
        dims[face_axis] = "radius"
        means = np.moveaxis(means, 0, face_axis)

        hit_count = xr.DataArray(
            data=hit_count, dims="radius", coords={"radius": radii_deg}
        )

        uxda = xr.DataArray(
            means,
            dims=dims,
            coords={"radius": radii_deg},
            name=self.name + "_azimuthal_mean"
            if self.name is not None
            else "azimuthal_mean",
            attrs={
                "azimuthal_mean": True,
                "center_lon": lon_deg,
                "center_lat": lat_deg,
                "radius_units": "degrees",
            },
        )

        if return_hit_counts:
            return uxda, hit_count
        else:
            return uxda

    azimuthal_average = azimuthal_mean

    def weighted_mean(self, weights=None):
        """Computes a weighted mean.

        This function calculates the weighted mean of a variable,
        using the specified `weights`. If no weights are provided, it will automatically select
        appropriate weights based on whether the variable is face-centered or edge-centered. If
        the variable is neither face nor edge-centered a warning is raised, and an unweighted mean is computed instead.

        Parameters
        ----------
        weights : np.ndarray or None, optional
            The weights to use for the weighted mean calculation. If `None`, the function will
            determine weights based on the variable's association:

            - For face-centered variables: uses `self.uxgrid.face_areas.data`
            - For edge-centered variables: uses `self.uxgrid.edge_node_distances.data`

            If the variable is neither face-centered nor edge-centered, a warning is raised, and
            an unweighted mean is computed instead. User-defined weights should match the shape
            of the data variable's last dimension.

        Returns
        -------
        UxDataArray
            A new `UxDataArray` object representing the weighted mean of the input variable. The
            result is attached to the same `uxgrid` attribute as the original variable.

        Example
        -------
        >>> weighted_mean = uxds["t2m"].weighted_mean()


        Raises
        ------
        AssertionError
            If user-defined `weights` are provided and the shape of `weights` does not match
            the shape of the data variable's last dimension.

        Warnings
        --------
        UserWarning
            Raised when attempting to compute a weighted mean on a variable without associated
            weights. An unweighted mean will be computed in this case.

        Notes
        -----
        - The weighted mean is computed along the last dimension of the data variable, which is
          assumed to be the geometry dimension (e.g., faces, edges, or nodes).
        """
        if weights is None:
            if self._face_centered():
                weights = self.uxgrid.face_areas.data
            elif self._edge_centered():
                weights = self.uxgrid.edge_node_distances.data
            else:
                warnings.warn(
                    "Attempting to perform a weighted mean calculation on a variable that does not have"
                    "associated weights. Weighted mean is only supported for face or edge centered "
                    "variables. Performing an unweighted mean."
                )
        else:
            # user-defined weights
            assert weights.shape[-1] == self.shape[-1]

        # compute the total weight
        total_weight = weights.sum()

        # compute the weighted mean, with an assumption on the index of dimension (last one is geometry)
        weighted_mean = (self * weights).sum(axis=-1) / total_weight

        # create a UxDataArray and return it
        return UxDataArray(weighted_mean, uxgrid=self.uxgrid)

    def topological_mean(
        self,
        destination: Literal["node", "edge", "face"],
        **kwargs,
    ):
        """Performs a topological mean aggregation.

        See Also
        --------
        numpy.mean
        dask.array.mean
        xarray.DataArray.mean

        Parameters
        ----------
        destination: str,
            Destination grid dimension for aggregation.

            Node-Centered Variable:
            - ``destination='edge'``: Aggregation is applied on the nodes that saddle each edge, with the result stored
            on each edge
            - ``destination='face'``: Aggregation is applied on the nodes that surround each face, with the result stored
            on each face.

            Edge-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the edges that intersect each node, with the result stored
            on each node.
            - ``Destination='face'``: Aggregation is applied on the edges that surround each face, with the result stored
            on each face.

            Face-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the faces that saddle each node, with the result stored
            on each node.
            - ``Destination='edge'``: Aggregation is applied on the faces that saddle each edge, with the result stored
            on each edge.


        Returns
        -------
        reduced: UxDataArray
            New UxDataArray with ``mean`` applied to its data.
        """
        return _uxda_grid_aggregate(self, destination, "mean", **kwargs)

    def topological_min(
        self,
        destination=None,
        **kwargs,
    ):
        """Performs a topological min aggregation.

        See Also
        --------
        numpy.min
        dask.array.min
        xarray.DataArray.min

        Parameters
        ----------
        destination: str,
            Destination grid dimension for Aggregation.

            Node-Centered Variable:
            - ``destination='edge'``: Aggregation is applied on the nodes that saddle each edge, with the result stored
            on each edge
            - ``destination='face'``: Aggregation is applied on the nodes that surround each face, with the result stored
            on each face.

            Edge-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the edges that intersect each node, with the result stored
            on each node.
            - ``Destination='face'``: Aggregation is applied on the edges that surround each face, with the result stored
            on each face.

            Face-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the faces that saddle each node, with the result stored
            on each node.
            - ``Destination='edge'``: Aggregation is applied on the faces that saddle each edge, with the result stored
            on each edge.


        Returns
        -------
        reduced: UxDataArray
            New UxDataArray with ``min`` applied to its data.
        """
        return _uxda_grid_aggregate(self, destination, "min", **kwargs)

    def topological_max(
        self,
        destination=None,
        **kwargs,
    ):
        """Performs a topological max aggregation.

        See Also
        --------
        numpy.max
        dask.array.max
        xarray.DataArray.max

        Parameters
        ----------
        destination: str,
            Destination grid dimension for Aggregation.

            Node-Centered Variable:
            - ``destination='edge'``: Aggregation is applied on the nodes that saddle each edge, with the result stored
            on each edge
            - ``destination='face'``: Aggregation is applied on the nodes that surround each face, with the result stored
            on each face.

            Edge-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the edges that intersect each node, with the result stored
            on each node.
            - ``Destination='face'``: Aggregation is applied on the edges that surround each face, with the result stored
            on each face.

            Face-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the faces that saddle each node, with the result stored
            on each node.
            - ``Destination='edge'``: Aggregation is applied on the faces that saddle each edge, with the result stored
            on each edge.


        Returns
        -------
        reduced: UxDataArray
            New UxDataArray with ``max`` applied to its data.
        """

        return _uxda_grid_aggregate(self, destination, "max", **kwargs)

    def topological_median(
        self,
        destination=None,
        **kwargs,
    ):
        """Performs a topological median aggregation.

        See Also
        --------
        numpy.median
        dask.array.median
        xarray.DataArray.median

        Parameters
        ----------

        destination: str,
            Destination grid dimension for Aggregation.

            Node-Centered Variable:
            - ``destination='edge'``: Aggregation is applied on the nodes that saddle each edge, with the result stored
            on each edge
            - ``destination='face'``: Aggregation is applied on the nodes that surround each face, with the result stored
            on each face.

            Edge-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the edges that intersect each node, with the result stored
            on each node.
            - ``Destination='face'``: Aggregation is applied on the edges that surround each face, with the result stored
            on each face.

            Face-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the faces that saddle each node, with the result stored
            on each node.
            - ``Destination='edge'``: Aggregation is applied on the faces that saddle each edge, with the result stored
            on each edge.


        Returns
        -------
        reduced: UxDataArray
            New UxDataArray with ``median`` applied to its data.
        """
        return _uxda_grid_aggregate(self, destination, "median", **kwargs)

    def topological_std(
        self,
        destination=None,
        **kwargs,
    ):
        """Performs a topological std aggregation.

        See Also
        --------
        numpy.std
        dask.array.std
        xarray.DataArray.std

        Parameters
        ----------
        destination: str,
            Destination grid dimension for Aggregation.

            Node-Centered Variable:
            - ``destination='edge'``: Aggregation is applied on the nodes that saddle each edge, with the result stored
            on each edge
            - ``destination='face'``: Aggregation is applied on the nodes that surround each face, with the result stored
            on each face.

            Edge-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the edges that intersect each node, with the result stored
            on each node.
            - ``Destination='face'``: Aggregation is applied on the edges that surround each face, with the result stored
            on each face.

            Face-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the faces that saddle each node, with the result stored
            on each node.
            - ``Destination='edge'``: Aggregation is applied on the faces that saddle each edge, with the result stored
            on each edge.


        Returns
        -------
        reduced: UxDataArray
            New UxDataArray with ``std`` applied to its data.
        """
        return _uxda_grid_aggregate(self, destination, "std", **kwargs)

    def topological_var(
        self,
        destination=None,
        **kwargs,
    ):
        """Performs a topological var aggregation.

        See Also
        --------
        numpy.var
        dask.array.var
        xarray.DataArray.var

        Parameters
        ----------

        destination: str,
            Destination grid dimension for Aggregation.

            Node-Centered Variable:
            - ``destination='edge'``: Aggregation is applied on the nodes that saddle each edge, with the result stored
            on each edge
            - ``destination='face'``: Aggregation is applied on the nodes that surround each face, with the result stored
            on each face.

            Edge-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the edges that intersect each node, with the result stored
            on each node.
            - ``Destination='face'``: Aggregation is applied on the edges that surround each face, with the result stored
            on each face.

            Face-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the faces that saddle each node, with the result stored
            on each node.
            - ``Destination='edge'``: Aggregation is applied on the faces that saddle each edge, with the result stored
            on each edge.


        Returns
        -------
        reduced: UxDataArray
            New UxDataArray with ``var`` applied to its data.
        """
        return _uxda_grid_aggregate(self, destination, "var", **kwargs)

    def topological_sum(
        self,
        destination=None,
        **kwargs,
    ):
        """Performs a topological sum aggregation.

        See Also
        --------
        numpy.sum
        dask.array.sum
        xarray.DataArray.sum

        Parameters
        ----------
        destination: str,
            Destination grid dimension for Aggregation.

            Node-Centered Variable:
            - ``destination='edge'``: Aggregation is applied on the nodes that saddle each edge, with the result stored
            on each edge
            - ``destination='face'``: Aggregation is applied on the nodes that surround each face, with the result stored
            on each face.

            Edge-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the edges that intersect each node, with the result stored
            on each node.
            - ``Destination='face'``: Aggregation is applied on the edges that surround each face, with the result stored
            on each face.

            Face-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the faces that saddle each node, with the result stored
            on each node.
            - ``Destination='edge'``: Aggregation is applied on the faces that saddle each edge, with the result stored
            on each edge.


        Returns
        -------
        reduced: UxDataArray
            New UxDataArray with ``sum`` applied to its data.
        """
        return _uxda_grid_aggregate(self, destination, "sum", **kwargs)

    def topological_prod(
        self,
        destination=None,
        **kwargs,
    ):
        """Performs a topological prod aggregation.

        See Also
        --------
        numpy.prod
        dask.array.prod
        xarray.DataArray.prod

        Parameters

        destination: str,
            Destination grid dimension for Aggregation.

            Node-Centered Variable:
            - ``destination='edge'``: Aggregation is applied on the nodes that saddle each edge, with the result stored
            on each edge
            - ``destination='face'``: Aggregation is applied on the nodes that surround each face, with the result stored
            on each face.

            Edge-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the edges that intersect each node, with the result stored
            on each node.
            - ``Destination='face'``: Aggregation is applied on the edges that surround each face, with the result stored
            on each face.

            Face-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the faces that saddle each node, with the result stored
            on each node.
            - ``Destination='edge'``: Aggregation is applied on the faces that saddle each edge, with the result stored
            on each edge.


        Returns
        -------
        reduced: UxDataArray
            New UxDataArray with ``prod`` applied to its data.
        """
        return _uxda_grid_aggregate(self, destination, "prod", **kwargs)

    def topological_all(
        self,
        destination=None,
        **kwargs,
    ):
        """Performs a topological all aggregation.

        See Also
        --------
        numpy.all
        dask.array.all
        xarray.DataArray.all

        Parameters
        ----------
        destination: str,
            Destination grid dimension for Aggregation.

            Node-Centered Variable:
            - ``destination='edge'``: Aggregation is applied on the nodes that saddle each edge, with the result stored
            on each edge
            - ``destination='face'``: Aggregation is applied on the nodes that surround each face, with the result stored
            on each face.

            Edge-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the edges that intersect each node, with the result stored
            on each node.
            - ``Destination='face'``: Aggregation is applied on the edges that surround each face, with the result stored
            on each face.

            Face-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the faces that saddle each node, with the result stored
            on each node.
            - ``Destination='edge'``: Aggregation is applied on the faces that saddle each edge, with the result stored
            on each edge.


        Returns
        -------
        reduced: UxDataArray
            New UxDataArray with ``all`` applied to its data.
        """
        return _uxda_grid_aggregate(self, destination, "all", **kwargs)

    def topological_any(
        self,
        destination=None,
        **kwargs,
    ):
        """Performs a topological any aggregation.

        See Also
        --------
        numpy.any
        dask.array.any
        xarray.DataArray.any

        Parameters
        ----------
        destination: str,
            Destination grid dimension for Aggregation.

            Node-Centered Variable:
            - ``destination='edge'``: Aggregation is applied on the nodes that saddle each edge, with the result stored
            on each edge
            - ``destination='face'``: Aggregation is applied on the nodes that surround each face, with the result stored
            on each face.

            Edge-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the edges that intersect each node, with the result stored
            on each node.
            - ``Destination='face'``: Aggregation is applied on the edges that surround each face, with the result stored
            on each face.

            Face-Centered Variable:
            - ``destination='node'``: Aggregation is applied on the faces that saddle each node, with the result stored
            on each node.
            - ``Destination='edge'``: Aggregation is applied on the faces that saddle each edge, with the result stored
            on each edge.


        Returns
        -------
        reduced: UxDataArray
            New UxDataArray with ``any`` applied to its data.
        """
        return _uxda_grid_aggregate(self, destination, "any", **kwargs)

    def gradient(self, **kwargs) -> UxDataset:
        """
        Computes the gradient of a data variable.

        Returns
        -------
        gradient: UxDataset
            Dataset containing the zonal and merdional components of the gradient.

        Notes
        -----
        The Green-Gauss theorm is utilized, where a closed control volume around each cell
        is formed connecting centroids of the neighboring cells. The surface integral is
        approximated using the trapezoidal rule. The sum of the contributions is then
        normalized by the cell volume.

        Example
        -------
        >>> uxds["var"].gradient()
        """
        from uxarray import UxDataset

        if "use_magnitude" in kwargs or "normalize" in kwargs:
            # Deprecation warning for old gradient implementation
            warn(
                "The `use_magnitude` and `normalize` parameters are deprecated. ",
                DeprecationWarning,
            )

        # Compute the zonal and meridional gradient components of the stored data variable
        grad_zonal_da, grad_meridional_da = _compute_gradient(self)

        # Create a dataset containing both gradient components
        return UxDataset(
            {
                "zonal_gradient": grad_zonal_da,
                "meridional_gradient": grad_meridional_da,
            },
            uxgrid=self.uxgrid,
            attrs={"gradient": True},
            coords=self.coords,
        )

    def curl(self, other: "UxDataArray", **kwargs) -> "UxDataArray":
        """
        Computes the curl of a vector field.

        Parameters
        ----------
        other : UxDataArray
            The second component of the vector field. This UxDataArray should
            represent the meridional (v) component, while self represents the
            zonal (u) component.
        **kwargs : dict
            Additional keyword arguments (currently unused, reserved for future extensions).

        Returns
        -------
        curl : UxDataArray
            The curl of the vector field (u, v), computed as:
            curl = ∂v/∂x - ∂u/∂y

        Notes
        -----
        The curl is computed using the existing gradient infrastructure.
        For a 2D vector field V = (u, v), the curl is a scalar field representing
        the rotation or circulation density at each point.

        The curl is computed by:
        1. Computing the gradient of the u-component: ∇u = (∂u/∂x, ∂u/∂y)
        2. Computing the gradient of the v-component: ∇v = (∂v/∂x, ∂v/∂y)
        3. Extracting the relevant components: ∂v/∂x and ∂u/∂y
        4. Computing: curl = ∂v/∂x - ∂u/∂y

        Requirements:
        - Both components must be UxDataArray objects
        - Both must be defined on the same grid
        - Both must be 1-dimensional (use .isel() for multi-dimensional data)
        - Data must be face-centered

        Example
        -------
        >>> u_component = uxds["u_wind"]
        >>> v_component = uxds["v_wind"]
        >>> curl_field = u_component.curl(v_component)
        """
        # Input validation
        if not isinstance(other, UxDataArray):
            raise TypeError("other must be a UxDataArray")

        if self.uxgrid != other.uxgrid:
            raise ValueError("Both vector components must be on the same grid")

        if self.dims != other.dims:
            raise ValueError("Both vector components must have the same dimensions")

        if len(self.dims) != 1:
            raise ValueError(
                "Curl computation currently only supports 1-dimensional data. "
                "Use .isel() to select a single time slice or level."
            )

        # Compute gradients of both components
        grad_u_zonal, grad_u_meridional = _compute_gradient(self)
        grad_v_zonal, grad_v_meridional = _compute_gradient(other)

        # Compute curl = ∂v/∂x - ∂u/∂y
        curl_values = grad_v_zonal.values - grad_u_meridional.values

        # Create the result UxDataArray
        curl_da = UxDataArray(
            curl_values,
            dims=self.dims,
            attrs={
                "long_name": f"Curl of ({self.name}, {other.name})",
                "units": "1/s"
                if "units" not in self.attrs
                else f"({self.attrs.get('units', '1')})/m",
                "description": "Curl of vector field computed as ∂v/∂x - ∂u/∂y",
            },
            uxgrid=self.uxgrid,
            name=f"curl_{self.name}_{other.name}",
        )

        return curl_da

    def divergence(self, other: "UxDataArray", **kwargs) -> "UxDataArray":
        """
        Computes the divergence of the vector field defined by this UxDataArray and other.

        Parameters
        ----------
        other : UxDataArray
            The second component of the vector field. This UxDataArray represents the first component.
        **kwargs
            Additional keyword arguments (reserved for future use).

        Returns
        -------
        divergence : UxDataArray
            UxDataArray containing the divergence of the vector field.

        Notes
        -----
        The divergence is computed using the finite volume method. For a vector field V = (u, v),
        where u and v are the components represented by this UxDataArray and other respectively,
        the divergence is calculated as div(V) = ∂u/∂x + ∂v/∂y.

        The implementation uses edge-centered gradients and face-centered divergence calculation
        following the discrete divergence theorem.

        Example
        -------
        >>> u_component = uxds["u_wind"]  # First component of vector field
        >>> v_component = uxds["v_wind"]  # Second component of vector field
        >>> div_field = u_component.divergence(v_component)
        """
        if not isinstance(other, UxDataArray):
            raise TypeError("other must be a UxDataArray")

        if self.uxgrid != other.uxgrid:
            raise ValueError("Both UxDataArrays must have the same grid")

        if self.dims != other.dims:
            raise ValueError("Both UxDataArrays must have the same dimensions")

        if self.ndim > 1:
            raise ValueError(
                "Divergence currently requires 1D face-centered data. Consider "
                "reducing the dimension by selecting data across leading dimensions (e.g., `.isel(time=0)`, "
                "`.sel(lev=500)`, or `.mean('time')`)."
            )

        if not (self._face_centered() and other._face_centered()):
            raise ValueError(
                "Computing the divergence is only supported for face-centered data variables."
            )

        # Compute gradients of both components
        u_gradient = self.gradient()
        v_gradient = other.gradient()

        # For divergence: div(V) = ∂u/∂x + ∂v/∂y
        # We use the zonal gradient (∂/∂lon) of u and meridional gradient (∂/∂lat) of v
        u = u_gradient["zonal_gradient"]
        v = v_gradient["meridional_gradient"]

        # Align DataArrays to ensure coords/dims match, then perform xarray-aware addition
        u, v = xr.align(u, v)
        divergence = u + v
        divergence.name = "divergence"
        divergence.attrs.update(
            {
                "divergence": True,
                "units": "1/s" if "units" not in kwargs else kwargs["units"],
            }
        )

        # Wrap result as a UxDataArray while preserving uxgrid and coords
        divergence_da = UxDataArray(divergence, uxgrid=self.uxgrid)

        return divergence_da

    def difference(self, destination: str | None = "edge"):
        """Computes the absolute difference of a data variable.

        The difference for a face-centered data variable can be computed on each edge using the ``edge_face_connectivity``,
        specified by ``destination='edge'``.

        The difference for a node-centered data variable can be computed on each edge using the ``edge_node_connectivity``,
        specified by ``destination='edge'``.

        Computing the difference for an edge-centered data variable is not yet supported.

        Note
        ----
        Not to be confused with the ``.diff()`` method from xarray.
        https://docs.xarray.dev/en/stable/generated/xarray.DataArray.diff.html

        Parameters
        ----------
        destination: {‘node’, ‘edge’, ‘face’}, default='edge''
            The desired destination for computing the difference across and storing on
        """

        if destination not in ["node", "edge", "face"]:
            raise ValueError(
                f"Invalid destination '{destination}'. Must be one of ['node', 'edge', 'face']"
            )

        dims = list(self.dims)
        var_name = str(self.name) + "_" if self.name is not None else " "

        if self._face_centered():
            if destination == "edge":
                _difference = _calculate_edge_face_difference(
                    self.values,
                    self.uxgrid.edge_face_connectivity.values,
                    self.uxgrid.n_edge,
                )
                dims[-1] = "n_edge"
                name = f"{var_name}edge_face_difference"
            elif destination == "face":
                raise ValueError(
                    "Invalid destination 'face' for a face-centered data variable, computing"
                    "the difference and storing it on each face is not possible"
                )
            elif destination == "node":
                raise ValueError(
                    "Support for computing the difference of a face-centered data variable and storing"
                    "the result on each node not yet supported."
                )

        elif self._node_centered():
            if destination == "edge":
                _difference = _calculate_edge_node_difference(
                    self.values, self.uxgrid.edge_node_connectivity.values
                )
                dims[-1] = "n_edge"
                name = f"{var_name}edge_node_difference"
            elif destination == "node":
                raise ValueError(
                    "Invalid destination 'node' for a node-centered data variable, computing"
                    "the difference and storing it on each node is not possible"
                )

            elif destination == "face":
                raise ValueError(
                    "Support for computing the difference of a node-centered data variable and storing"
                    "the result on each face not yet supported."
                )

        elif self._edge_centered():
            raise NotImplementedError(
                "Difference for edge centered data variables not yet implemented"
            )

        else:
            raise ValueError("TODO: ")

        uxda = UxDataArray(
            _difference,
            uxgrid=self.uxgrid,
            name=name,
            dims=dims,
        )

        return uxda

    def _face_centered(self) -> bool:
        """Returns whether the data stored is Face Centered (i.e. contains the
        "n_face" dimension)"""
        return "n_face" in self.dims

    def _node_centered(self) -> bool:
        """Returns whether the data stored is Node Centered (i.e. contains the
        "n_node" dimension)"""
        return "n_node" in self.dims

    def _edge_centered(self) -> bool:
        """Returns whether the data stored is Edge Centered (i.e. contains the
        "n_edge" dimension)"""
        return "n_edge" in self.dims

    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        drop: bool = False,
        missing_dims: str = "raise",
        ignore_grid: bool = False,
        inverse_indices: bool = False,
        **indexers_kwargs,
    ):
        """
        Return a new DataArray whose data is given by selecting indexes along the specified dimension(s).

        Performs xarray-style integer-location indexing along specified dimensions.
        If a single grid dimension ('n_node', 'n_edge', or 'n_face') is provided
        and `ignore_grid=False`, the underlying grid is sliced accordingly,
        and remaining indexers are applied to the resulting DataArray.

        Parameters
        ----------
        indexers : Mapping[Any, Any], optional
            A mapping of dimension names to indexers. Each indexer may be an integer,
            slice, array-like, or DataArray. Mutually exclusive with indexing via kwargs.
        drop : bool, default=False
            If True, drop any coordinate variables indexed by integers instead of
            retaining them as length-1 dimensions.
        missing_dims : {'raise', 'warn', 'ignore'}, default='raise'
            Behavior when indexers reference dimensions not present in the array.
            - 'raise': raise an error
            - 'warn': emit a warning and ignore missing dimensions
            - 'ignore': ignore missing dimensions silently
        ignore_grid : bool, default=False
            If False (default), allow slicing on one grid dimension to automatically
            update the associated UXarray grid. If True, fall back to pure xarray behavior.
        inverse_indices : bool, default=False
            For grid-based slicing, pass this flag to `Grid.isel` to invert indices
            when selecting (useful for staggering or reversing order).
        **indexers_kwargs : dimension=indexer pairs, optional
            Alternative syntax for specifying `indexers` via keyword arguments.

        Returns
        -------
        UxDataArray
            A new UxDataArray indexed according to `indexers` and updated grid if applicable.

        Raises
        ------
        ValueError
            If more than one grid dimension is selected and `ignore_grid=False`.
        """
        from uxarray.core.utils import _validate_indexers

        indexers, grid_dims = _validate_indexers(
            indexers, indexers_kwargs, "isel", ignore_grid
        )

        try:
            # Grid Branch
            if not ignore_grid:
                if len(grid_dims) == 1:
                    # pop off the one grid‐dim indexer
                    grid_dim = grid_dims.pop()
                    grid_indexer = indexers.pop(grid_dim)

                    sliced_grid = self.uxgrid.isel(
                        **{grid_dim: grid_indexer}, inverse_indices=inverse_indices
                    )

                    da = self._slice_from_grid(sliced_grid)

                    # if there are any remaining indexers, apply them
                    if indexers:
                        xarr = super(UxDataArray, da).isel(
                            indexers=indexers, drop=drop, missing_dims=missing_dims
                        )
                        # re‐wrap so the grid sticks around
                        return type(self)(xarr, uxgrid=sliced_grid)

                    # no other dims, return the grid‐sliced da
                    return da
                else:
                    return type(self)(
                        super().isel(
                            indexers=indexers or None,
                            drop=drop,
                            missing_dims=missing_dims,
                        ),
                        uxgrid=self.uxgrid,
                    )

            return super().isel(
                indexers=indexers or None,
                drop=drop,
                missing_dims=missing_dims,
            )
        except ValueError as e:
            if "Dimensions" in str(e) and "do not exist" in str(e):
                # The error message from xarray is quite good, but we can add to it.
                # e.g. "Dimensions {'level'} do not exist. Expected one of ('n_face', 'time', 'lev')"
                # Let's just append the available dimensions.
                original_error_msg = str(e)
                raise ValueError(
                    f"{original_error_msg}. Available dimensions: {self.dims}"
                ) from e
            else:
                # re-raise other ValueErrors
                raise e

    @classmethod
    def from_xarray(cls, da: xr.DataArray, uxgrid: Grid, ugrid_dims: dict = None):
        """
        Converts a ``xarray.DataArray`` into a ``uxarray.UxDataset`` paired with a user-defined ``Grid``

        Parameters
        ----------
        da : xr.DataArray
            An Xarray data array containing data residing on an unstructured grid
        uxgrid : Grid
            ``Grid`` object representing an unstructured grid
        ugrid_dims : dict, optional
            A dictionary mapping data array dimensions to UGRID dimensions.

        Returns
        -------
        cls
            A ``ux.UxDataArray`` with data from the ``xr.DataArray` paired with a ``ux.Grid``
        """
        if ugrid_dims is None:
            ugrid_dims = uxgrid._source_dims_dict

        # map each dimension to its UGRID equivalent
        ds = _map_dims_to_ugrid(da, ugrid_dims, uxgrid)

        return cls(ds, uxgrid=uxgrid)

    @classmethod
    def from_healpix(
        cls,
        da: xr.DataArray,
        pixels_only: bool = True,
        face_dim: str = "cell",
        **kwargs,
    ):
        """
        Loads a data array represented in the HEALPix format into a ``ux.UxDataArray``, paired
        with a ``Grid`` containing information about the HEALPix definition.

        Parameters
        ----------
        da: xr.DataArray
            Reference to a HEALPix DataArray
        pixels_only : bool, optional
            Whether to only compute pixels (`face_lon`, `face_lat`) or to also construct boundaries (`face_node_connectivity`, `node_lon`, `node_lat`)
        face_dim: str, optional
            Data dimension corresponding to the HEALPix face mapping. Typically, is set to "cell", but may differ.

        Returns
        -------
        cls
            A ``ux.UxDataArray`` instance
        """

        if not isinstance(da, xr.DataArray):
            raise ValueError("`da` must be a xr.DataArray")

        if face_dim not in da.dims:
            raise ValueError(
                f"The provided face dimension '{face_dim}' is present in the provided healpix data array."
                f"Please set 'face_dim' to the dimension corresponding to the healpix face dimension."
            )

        # Attach a HEALPix Grid
        uxgrid = Grid.from_healpix(
            zoom=get_zoom_from_cells(da.sizes[face_dim]),
            pixels_only=pixels_only,
            **kwargs,
        )

        return cls.from_xarray(da, uxgrid, {face_dim: "n_face"})

    def _slice_from_grid(self, sliced_grid):
        """Slices a  ``UxDataArray`` from a sliced ``Grid``, using cached
        indices to correctly slice the data variable."""

        if self._face_centered():
            da_sliced = self.isel(
                n_face=sliced_grid._ds["subgrid_face_indices"], ignore_grid=True
            )

        elif self._edge_centered():
            da_sliced = self.isel(
                n_edge=sliced_grid._ds["subgrid_edge_indices"], ignore_grid=True
            )

        elif self._node_centered():
            da_sliced = self.isel(
                n_node=sliced_grid._ds["subgrid_node_indices"], ignore_grid=True
            )

        else:
            raise ValueError(
                "Data variable must be either node, edge, or face centered."
            )

        return UxDataArray(da_sliced, uxgrid=sliced_grid)

    def get_dual(self):
        """Compute the dual mesh for a data array, returns a new data array
        object.

        Returns
        --------
        dual : uxda
            Dual Mesh `uxda` constructed
        """

        if _check_duplicate_nodes_indices(self.uxgrid):
            raise RuntimeError("Duplicate nodes found, cannot construct dual")

        if self.uxgrid.partial_sphere_coverage:
            warn(
                "This mesh is partial, which could cause inconsistent results and data will be lost",
                Warning,
            )

        # Get dual mesh node face connectivity
        dual_node_face_conn = construct_dual(grid=self.uxgrid)

        # Construct dual mesh
        dual = self.uxgrid.from_topology(
            self.uxgrid.face_lon.values,
            self.uxgrid.face_lat.values,
            dual_node_face_conn,
        )

        # Dictionary to swap dimensions
        dim_map = {"n_face": "n_node", "n_node": "n_face"}

        # Get correct dimensions for the dual
        dims = [dim_map.get(dim, dim) for dim in self.dims]

        # Get the values from the data array
        data = np.array(self.values)

        # Construct the new data array
        uxda = uxarray.UxDataArray(uxgrid=dual, data=data, dims=dims, name=self.name)

        return uxda

    def __getattribute__(self, name):
        """Intercept accessor method calls to return Ux-aware accessors."""
        # Lazy import to avoid circular imports
        from uxarray.core.accessors import DATAARRAY_ACCESSOR_METHODS

        if name in DATAARRAY_ACCESSOR_METHODS:
            from uxarray.core import accessors

            # Get the accessor class by name
            accessor_class = getattr(accessors, DATAARRAY_ACCESSOR_METHODS[name])

            # Get the parent method
            parent_method = super().__getattribute__(name)

            # Create a wrapper method
            def method(*args, **kwargs):
                # Call the parent method
                result = parent_method(*args, **kwargs)
                # Wrap the result with our accessor
                return accessor_class(result, self.uxgrid)

            # Copy the docstring from the parent method
            method.__doc__ = parent_method.__doc__
            method.__name__ = name

            return method

        # For all other attributes, use the default behavior
        return super().__getattribute__(name)

    def where(self, cond: Any, other: Any = dtypes.NA, drop: bool = False):
        return UxDataArray(super().where(cond, other, drop), uxgrid=self.uxgrid)

    where.__doc__ = xr.DataArray.where.__doc__

    def fillna(self, value: Any):
        return UxDataArray(super().fillna(value), uxgrid=self.uxgrid)

    fillna.__doc__ = xr.DataArray.fillna.__doc__
