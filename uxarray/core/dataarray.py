from __future__ import annotations

import warnings
from html import escape
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Literal, Mapping, Optional
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core import dtypes
from xarray.core.options import OPTIONS
from xarray.core.utils import UncachedAccessor

import uxarray
from uxarray.constants import GRID_DIMS
from uxarray.core.aggregation import _uxda_grid_aggregate
from uxarray.core.gradient import (
    _calculate_edge_face_difference,
    _calculate_edge_node_difference,
    _compute_gradient,
)
from uxarray.core.utils import (
    _map_dims_to_ugrid,
    _resolve_coordinate_labels_to_indices,
    _validate_indexers,
)
from uxarray.core.zonal import (
    _compute_conservative_zonal_mean_bands,
    _compute_non_conservative_zonal_mean,
    _compute_zonal_anomaly,
)
from uxarray.cross_sections import UxDataArrayCrossSectionAccessor
from uxarray.errors import (
    DataCenteringError,
    DimensionError,
    GridInvalidError,
    GridsMismatchError,
)
from uxarray.formatting_html import array_repr
from uxarray.grid import Grid
from uxarray.grid.dual import construct_dual
from uxarray.grid.neighbors import DataArrayNeighborhood, Neighborhood
from uxarray.grid.validation import _check_duplicate_nodes_indices
from uxarray.io._healpix import get_zoom_from_cells
from uxarray.plot.accessor import UxDataArrayPlotAccessor
from uxarray.remap.accessor import RemapAccessor
from uxarray.subset import DataArraySubsetAccessor
from uxarray.utils.coords import _preserve_valid_coords

if TYPE_CHECKING:
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes

    from uxarray.core.dataset import UxDataset


class UxDataArray(xr.DataArray):
    """Grid informed ``xarray.DataArray`` with an attached ``Grid`` accessor
    and grid-specific functionality.

    Parameters
    ----------
    uxgrid : uxarray.Grid
        The `Grid` object that makes this array aware of the unstructured
        grid topology it belongs to.
        Providing `None` is possible but intended for internal use only;
        if `None`, must set self.uxgrid before using any grid-aware methods.

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

    def __init__(self, *args, uxgrid: Grid | None = None, **kwargs):
        # Note: allowing uxgrid=None is not desirable (see issue #1620)
        #   but it is the default, to simplify subclassing from xarray.
        # E.g., self.isel() uses self._replace(), which goes to xarray.DataArray._replace(),
        #   which returns type(self)(...) with explicit kwargs only (no **kwargs),
        #   making it very challenging to pass uxgrid at time of construction.
        # Workaround here: clarified in docstring, and allow initial uxgrid=None,
        #   but crash with GridInvalidError upon accessing self.uxgrid, if still None.

        # Need self._uxgrid if None; self.uxgrid ensures value is actually a Grid.
        if uxgrid is None:
            self._uxgrid = uxgrid
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
            copied._uxgrid = self._uxgrid.copy()  # deep copy
        else:
            # Point to the existing uxgrid object
            copied._uxgrid = self._uxgrid

        return copied

    def _replace(self, *args, **kwargs):
        """Override to make the result a complete instance of
        ``uxarray.UxDataArray``."""
        da = super()._replace(*args, **kwargs)

        if isinstance(da, UxDataArray):
            da._uxgrid = self._uxgrid
        else:
            da = UxDataArray(da, uxgrid=self._uxgrid)

        return da

    @property
    def uxgrid(self) -> Grid:
        """Linked unstructured grid (``uxarray.Grid``) which the data resides on."""
        # _uxgrid=None should only cause crash during grid-aware operations.
        # So, internally: use self._uxgrid for non-grid-aware operations like _copy() or _replace(),
        # but self.uxgrid for everything else, like integrate().
        if self._uxgrid is None:
            # (comment in self.__init__ describes why this possibility exists.)
            raise GridInvalidError(
                f"Expected a uxarray.Grid; got {type(self).__name__}.uxgrid = None. "
                "Maybe you forgot to provide uxgrid when initializing this UxDataArray?"
            )
        return self._uxgrid

    @uxgrid.setter
    def uxgrid(self, ugrid_obj: Grid):
        if not isinstance(ugrid_obj, Grid):
            raise TypeError(
                f"Expected a uxarray.Grid; got value with type={type(ugrid_obj)} "
                f"(while setting {type(self).__name__}.uxgrid = value)."
            )
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

    @property
    def data_location(self):
        """Returns where on the grid the data variable is stored.

        The location is inferred from the grid dimension present in the data
        variable, using UGRID-style names:

        - ``"face_centered"`` if the data contains the ``n_face`` dimension
        - ``"node_centered"`` if the data contains the ``n_node`` dimension
        - ``"edge_centered"`` if the data contains the ``n_edge`` dimension
        - ``None`` if the data is not mapped to the grid

        Notes
        -----
        Additional locations described in the UGRID/spectral-element ecosystem
        (e.g. ``"face_average"``, ``"edge_orthogonal"``, ``"edge_parallel"``,
        ``"cgll"``, ``"dgll"``) cannot be inferred from a data variable's
        dimensions alone and are not currently distinguished here.

        Returns
        -------
        str or None
            One of ``"face_centered"``, ``"node_centered"``,
            ``"edge_centered"``, or ``None``.
        """
        if self._face_centered():
            return "face_centered"
        elif self._node_centered():
            return "node_centered"
        elif self._edge_centered():
            return "edge_centered"
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

        if self.ndim > 1:
            # data is multidimensional, must be a 1D slice
            raise DimensionError(
                f"Data Variable must be 1-dimensional, with shape {self.uxgrid.n_face} "
                f"for face-centered data."
            )

        if self._face_centered():
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

        else:
            raise DataCenteringError(
                f"to_geodataframe() expects face_centered data; got {self.data_location} data "
                f"(with sizes={dict(**self.sizes)}). Consider running "
                "``UxDataArray.topological_mean(destination='face')`` to aggregate the data onto faces."
            )

        return gdf

    def to_polycollection(
        self,
        periodic_elements: Optional[str] = "exclude",
        projection: Optional[ccrs.Projection] = None,
        return_indices: Optional[bool] = False,
        cache: Optional[bool] = True,
        override: Optional[bool] = False,
        **kwargs,
    ):
        """Constructs a ``matplotlib.collections.PolyCollection``` consisting
        of polygons representing the faces of the current ``UxDataArray`` with
        a face-centered data variable mapped to them.

        Parameters
        ----------
        periodic_elements : str, optional
            Method for handling periodic elements. One of ['exclude', 'split', or 'ignore']:
            - 'exclude': Periodic elements will be identified and excluded from the GeoDataFrame
            - 'split': Periodic elements will be identified and split using the ``antimeridian`` package
            - 'ignore': No processing will be applied to periodic elements.
        projection: ccrs.Projection
            Cartopy geographic projection to use
        return_indices: bool
            Flag to indicate whether to return the indices of corrected polygons, if any exist
        cache: bool
            Flag to indicate whether to cache the computed PolyCollection
        override: bool
            Flag to indicate whether to override a cached PolyCollection, if it exists
        """
        # data is multidimensional, must be a 1D slice
        if self.ndim > 1:
            raise DimensionError(
                f"Data Variable must be 1-dimensional, with shape {self.uxgrid.n_face} "
                f"for face-centered data."
            )

        if self._face_centered():
            poly_collection, corrected_to_original_faces = (
                self.uxgrid.to_polycollection(
                    override=override,
                    cache=cache,
                    periodic_elements=periodic_elements,
                    return_indices=True,
                    projection=projection,
                    **kwargs,
                )
            )

            if periodic_elements == "exclude":
                # index data to ignore data mapped to periodic elements
                _data = np.delete(
                    self.values,
                    self.uxgrid._poly_collection_cached_parameters[
                        "antimeridian_face_indices"
                    ],
                    axis=0,
                )
            elif periodic_elements == "split":
                _data = self.values[corrected_to_original_faces]
            else:
                _data = self.values

            if (
                self.uxgrid._poly_collection_cached_parameters[
                    "non_nan_polygon_indices"
                ]
                is not None
            ):
                # index data to ignore NaN polygons
                _data = _data[
                    self.uxgrid._poly_collection_cached_parameters[
                        "non_nan_polygon_indices"
                    ]
                ]

            poly_collection.set_array(_data)

            if return_indices:
                return poly_collection, corrected_to_original_faces
            else:
                return poly_collection
        else:
            raise DataCenteringError("Data variable must be face centered.")

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
        from cartopy.mpl.geoaxes import GeoAxes

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
                return ax.get_autoscale_on()

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
        uxds = uxarray.core.dataset.UxDataset(xrds, uxgrid=self._uxgrid)

        return uxds

    def to_xarray(self) -> xr.DataArray:
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
        Open a UXarray dataset and compute the integral

        >>> import uxarray as ux
        >>> uxds = ux.open_dataset("grid.ug", "centroid_pressure_data_ug")
        >>> integral = uxds["psi"].integrate()
        """
        # TODO: support integration regardless of n_face dimension position,
        #    and remove the self.dims[-1] == "n_face" check.
        #    (uxarray/xarray features should be agnostic to dimension positions.)
        if self._face_centered() and self.dims[-1] == "n_face":
            # dot product between face areas and the face dimension of the data
            if isinstance(self.data, np.ndarray):
                # eager data: a direct einsum avoids xr.dot's per-call overhead
                integral = np.einsum(
                    "i,...i", self.uxgrid.face_areas.values, self.values
                )
            else:
                # dask-backed data: xr.dot keeps the reduction lazy
                integral = xr.dot(self, self.uxgrid.face_areas, dim="n_face")

        elif not self._face_centered():
            raise DataCenteringError(
                "Integration of non-face_centered data is not yet supported. "
                f"(Got {self.data_location} data with sizes={dict(**self.sizes)})"
            )

        else:
            raise DimensionError(
                "Integration of data with n_face not as the final dimension is not yet supported. "
                f"Got face_centered data, but the final dimension was {self.dims[-1]}, not 'n_face'."
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

        References
        ----------
        Chen, H., Ullrich, P. A., and Panetta, J. (2026). Fast and accurate
        intersections on a sphere. SIAM Journal on Scientific Computing, 48(2),
        B208-B232. https://doi.org/10.1137/25M1737614
        """
        if not self._face_centered():
            raise DataCenteringError(
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

            # Assign coords from `self` to the result except one that corresponds to `dims[face_axis]`
            new_coords = _preserve_valid_coords(self, "n_face")
            # Add latitudes to the resulting coords
            new_coords["latitudes"] = latitudes

            return xr.DataArray(
                res,
                dims=dims,
                coords=new_coords,
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
                raise DimensionError("Band edges must be 1D with at least two values")

            res = _compute_conservative_zonal_mean_bands(self, edges)

            # Use band centers as coordinate values
            centers = 0.5 * (edges[:-1] + edges[1:])

            dims = list(self.dims)
            dims[face_axis] = "latitudes"

            # Assign coords from `self` to the result except one that corresponds to `dims[face_axis]`
            new_coords = _preserve_valid_coords(self, "n_face")
            # Add latitudes to the resulting coords
            new_coords["latitudes"] = centers

            return xr.DataArray(
                res,
                dims=dims,
                coords=new_coords,
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
        """Alias of zonal_mean; prefer `zonal_mean` for primary API.

        See Also
        --------
        zonal_mean : Full docstring, including algorithm references.
        """
        return self.zonal_mean(lat=lat, conservative=conservative, **kwargs)

    def zonal_anomaly(self, lat=(-90, 90, 10), conservative: bool = False):
        """Compute the zonal anomaly: each face value minus the mean of its latitude band.

        Returns a new ``UxDataArray`` with the same dimensions as the input,
        where each face holds its original value minus the zonal mean of the
        latitude band it belongs to.

        Parameters
        ----------
        lat : tuple or array-like, default=(-90, 90, 10)
            Latitude band specification:
                - tuple (start, end, step): band edges via np.linspace(start, end, n)
                - array-like: explicit band edges in degrees
        conservative : bool, default=False
            If True, uses area-weighted band means and blends across bands for
            faces that straddle a band boundary, reusing the face-band weight
            matrix computed for zonal_mean so no geometry is duplicated.
            If False, assigns each face to a band by its centroid latitude.

        Returns
        -------
        UxDataArray
            Same dimensions as input with per-face band mean subtracted.

        Examples
        --------
        >>> uxds["var"].zonal_anomaly()
        >>> uxds["var"].zonal_anomaly(lat=(-60, 60, 5), conservative=True)

        See Also
        --------
        zonal_mean : Underlying zonal averaging algorithm and references.
        """
        if not self._face_centered():
            raise DataCenteringError(
                "Zonal anomaly is only supported for face-centered data variables."
            )

        if isinstance(lat, tuple):
            start, end, step = lat
            if step <= 0:
                raise ValueError("Step size must be positive.")
            num_points = int(round((end - start) / step)) + 1
            edges = np.linspace(start, end, num_points)
            edges = np.clip(edges, -90, 90)
        elif isinstance(lat, (list, np.ndarray)):
            edges = np.asarray(lat, dtype=float)
        else:
            raise TypeError(
                "Invalid value for 'lat'. Must be a tuple (start, end, step) or array-like band edges."
            )

        if edges.ndim != 1 or edges.size < 2:
            raise DimensionError("Band edges must be 1D with at least two values.")

        res = _compute_zonal_anomaly(self, edges, conservative=conservative)

        return UxDataArray(
            res,
            dims=self.dims,
            coords=self.coords,
            name=self.name + "_zonal_anomaly"
            if self.name is not None
            else "zonal_anomaly",
            attrs={"zonal_anomaly": True, "conservative": conservative},
            uxgrid=self.uxgrid,
        )

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
            raise DataCenteringError(
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

        # Assign coords from `self` to the result except one that corresponds to `dims[face_axis]`
        new_coords = _preserve_valid_coords(self, "n_face")
        # Add radii_deg to the resulting coords
        new_coords["radius"] = radii_deg

        uxda = xr.DataArray(
            means,
            dims=dims,
            coords=new_coords,
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

    def gradient(self, scale_by_radius: bool = True, **kwargs) -> UxDataset:
        """
        Computes the gradient of a data variable.

        Parameters
        ----------
        scale_by_radius : bool, default=True
            Divide unit-sphere derivatives by ``uxgrid.sphere_radius`` so the
            result carries physical, per-meter units (``[data units]/m``). When
            ``False`` the result is left on the unit sphere with per-radian units
            (``[data units]/rad``). If ``True`` but the grid has no
            ``sphere_radius`` attribute, the result falls back to unit-sphere
            output and a ``UserWarning`` is emitted.

        Returns
        -------
        gradient: UxDataset
            Dataset containing the zonal and meridional components of the gradient.
            With the default ``scale_by_radius=True`` the components are in
            ``[data units]/m``; with ``scale_by_radius=False`` they are in
            ``[data units]/rad``.

        Notes
        -----
        The Green-Gauss theorem is utilized, where a closed control volume around each cell
        is formed connecting centroids of the neighboring cells. The surface integral is
        approximated using the trapezoidal rule. The sum of the contributions is then
        normalized by the cell volume.

        By default the raw unit-sphere (per-radian) gradient is divided by
        ``uxgrid.sphere_radius`` to yield physical per-meter values. For Earth
        (radius ~6.37e6 m) this scales magnitudes down by ~1/6.37e6 relative to
        the unit-sphere result.

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
        grad_zonal_da, grad_meridional_da = _compute_gradient(
            self, scale_by_radius=scale_by_radius
        )

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

    def curl(
        self, other: "UxDataArray", scale_by_radius: bool = True, **kwargs
    ) -> "UxDataArray":
        """
        Computes the curl of a vector field.

        Parameters
        ----------
        other : UxDataArray
            The second component of the vector field. This UxDataArray should
            represent the meridional (v) component, while self represents the
            zonal (u) component.
        scale_by_radius : bool, default=True
            Divide unit-sphere derivatives by ``uxgrid.sphere_radius`` so the
            result carries physical, per-meter units (e.g. ``1/s`` for a velocity
            field). When ``False`` the result is left on the unit sphere
            (per radian).
        **kwargs : dict
            Additional keyword arguments (currently unused, reserved for future extensions).

        Returns
        -------
        curl : UxDataArray
            The curl of the vector field (u, v), computed as:
            curl = ∂v/∂x - ∂u/∂y. With the default ``scale_by_radius=True`` the
            result is in ``([u units])/m`` (``1/s`` for velocity); with
            ``scale_by_radius=False`` it is in ``([u units])/rad``.

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
            raise GridsMismatchError("Both vector components must be on the same grid")

        if self.dims != other.dims:
            raise DimensionError("Both vector components must have the same dimensions")

        if len(self.dims) != 1:
            raise DimensionError(
                "Curl computation currently only supports 1-dimensional data. "
                "Use .isel() to select a single time slice or level."
            )

        # Compute gradients of both components
        grad_u_zonal, grad_u_meridional = _compute_gradient(
            self, scale_by_radius=scale_by_radius
        )
        grad_v_zonal, grad_v_meridional = _compute_gradient(
            other, scale_by_radius=scale_by_radius
        )

        # Compute curl = ∂v/∂x - ∂u/∂y + u·tan(φ)/a
        #
        # The trailing term is the spherical metric term. Dropping it is only
        # valid on a plane; on the sphere it costs a factor of two on
        # solid-body rotation. When the derivatives have been divided by the
        # radius the term carries the same 1/a factor.
        tan_lat = np.tan(np.deg2rad(self.uxgrid.face_lat.values))
        metric = self.data * tan_lat
        if scale_by_radius and "sphere_radius" in self.uxgrid._ds.attrs:
            metric = metric / self.uxgrid._ds.attrs["sphere_radius"]
        curl_values = grad_v_zonal.data - grad_u_meridional.data + metric

        u_units = self.attrs.get("units", "")
        has_sphere_radius = "sphere_radius" in self.uxgrid._ds.attrs
        if scale_by_radius and has_sphere_radius:
            curl_units = f"({u_units})/m" if u_units else "1/m"
        else:
            curl_units = f"({u_units})/rad" if u_units else "1/rad"

        # Create the result UxDataArray
        curl_da = UxDataArray(
            curl_values,
            dims=self.dims,
            attrs={
                "long_name": f"Curl of ({self.name}, {other.name})",
                "units": curl_units,
                "description": (
                    "Curl of vector field computed as ∂v/∂x - ∂u/∂y + u·tan(φ)/a"
                ),
            },
            uxgrid=self.uxgrid,
            name=f"curl_{self.name}_{other.name}",
        )

        return curl_da

    def divergence(
        self, other: "UxDataArray", scale_by_radius: bool = True, **kwargs
    ) -> "UxDataArray":
        """
        Computes the divergence of the vector field defined by this UxDataArray and other.

        Parameters
        ----------
        other : UxDataArray
            The second (meridional, v) component of the vector field; ``self`` is
            the first (zonal, u) component.
        scale_by_radius : bool, default=True
            Divide unit-sphere derivatives by ``uxgrid.sphere_radius``. When
            ``True`` (and the grid has a ``sphere_radius`` attribute) the result
            carries physical, per-meter units (e.g. ``1/s`` for a velocity
            field); when ``False`` the result is left on the unit sphere
            (per radian).
        **kwargs
            Additional keyword arguments. ``units`` may be passed to override the
            automatically inferred units.

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
        following the discrete divergence theorem. By default the underlying
        gradients are divided by ``uxgrid.sphere_radius``; pass
        ``scale_by_radius=False`` for per-radian (unit-sphere) output.

        Example
        -------
        >>> u_component = uxds["u_wind"]  # First component of vector field
        >>> v_component = uxds["v_wind"]  # Second component of vector field
        >>> div_field = u_component.divergence(v_component)
        """
        if not isinstance(other, UxDataArray):
            raise TypeError("other must be a UxDataArray")

        if self.uxgrid != other.uxgrid:
            raise GridsMismatchError("Both UxDataArrays must have the same grid")

        if self.dims != other.dims:
            raise DimensionError("Both UxDataArrays must have the same dimensions")

        if self.ndim > 1:
            raise DimensionError(
                "Divergence currently requires 1D face-centered data. Consider "
                "reducing the dimension by selecting data across leading dimensions (e.g., `.isel(time=0)`, "
                "`.sel(lev=500)`, or `.mean('time')`)."
            )

        if not (self._face_centered() and other._face_centered()):
            raise DataCenteringError(
                "Computing the divergence is only supported for face-centered data variables."
            )

        # Compute gradients of both components
        u_gradient = self.gradient(scale_by_radius=scale_by_radius)
        v_gradient = other.gradient(scale_by_radius=scale_by_radius)

        # For divergence: div(V) = ∂u/∂x + ∂v/∂y - v·tan(φ)/a
        # We use the zonal gradient (∂/∂lon) of u and meridional gradient (∂/∂lat) of v
        u = u_gradient["zonal_gradient"]
        v = v_gradient["meridional_gradient"]

        # Align DataArrays to ensure coords/dims match, then perform xarray-aware addition
        u, v = xr.align(u, v)
        divergence = u + v

        # Spherical metric term, the companion of the one in curl(). Omitting
        # it is only valid on a plane.
        tan_lat = np.tan(np.deg2rad(self.uxgrid.face_lat.values))
        metric = other.values * tan_lat
        if scale_by_radius and "sphere_radius" in self.uxgrid._ds.attrs:
            metric = metric / self.uxgrid._ds.attrs["sphere_radius"]
        divergence = divergence - metric
        divergence.name = "divergence"

        # Infer units consistently with gradient()/curl(): a divergence is a
        # spatial derivative of the input field, so it carries an extra 1/length
        # factor (per meter when scaled by radius, otherwise per radian).
        if "units" in kwargs:
            div_units = kwargs["units"]
        else:
            u_units = self.attrs.get("units", "")
            has_sphere_radius = "sphere_radius" in self.uxgrid._ds.attrs
            if scale_by_radius and has_sphere_radius:
                div_units = f"({u_units})/m" if u_units else "1/m"
            else:
                div_units = f"({u_units})/rad" if u_units else "1/rad"

        divergence.attrs.update(
            {
                "divergence": True,
                "units": div_units,
            }
        )

        # Wrap result as a UxDataArray while preserving uxgrid and coords
        divergence_da = UxDataArray(divergence, uxgrid=self.uxgrid)

        return divergence_da

    def scalardotgradient(self, v: "UxDataArray", q: "UxDataArray") -> "UxDataArray":
        """
        Compute the dot product between a vector field and the gradient of a scalar field.

        Parameters
        ----------
        v : UxDataArray
            The meridional component of the vector field. ``self`` is treated as
            the zonal component.
        q : UxDataArray
            Scalar field whose gradient is dotted with the vector field.

        Returns
        -------
        scalar_dot_gradient : UxDataArray
            Dot product ``self * dq/dx + v * dq/dy``.
        """
        if not isinstance(v, UxDataArray):
            raise TypeError("v must be a UxDataArray")

        if not isinstance(q, UxDataArray):
            raise TypeError("q must be a UxDataArray")

        if self.uxgrid != v.uxgrid or self.uxgrid != q.uxgrid:
            raise GridsMismatchError("All UxDataArrays must have the same grid")

        if self.dims != v.dims or self.dims != q.dims:
            raise DimensionError("All UxDataArrays must have the same dimensions")

        if self.ndim > 1:
            raise DimensionError(
                "Scalar dot gradient currently requires 1D face-centered data. "
                "Consider selecting a single slice before computing."
            )

        if not (self._face_centered() and v._face_centered() and q._face_centered()):
            raise DataCenteringError(
                "Computing the scalar dot gradient is only supported for face-centered data variables."
            )

        # Validate coordinate alignment up-front so a misaligned input fails
        # before the (potentially expensive) gradient call.
        u_aligned, v_aligned, q_aligned = xr.align(self, v, q, join="exact", copy=False)

        q_gradient = q_aligned.gradient()
        q_zonal = q_gradient["zonal_gradient"]
        q_meridional = q_gradient["meridional_gradient"]

        scalar_dot_gradient = (u_aligned * q_zonal) + (v_aligned * q_meridional)
        scalar_dot_gradient.name = "scalar_dot_gradient"
        scalar_dot_gradient.attrs.update(
            {
                "long_name": "scalar dot gradient",
                "description": "Dot product u * (dq/dx) + v * (dq/dy).",
            }
        )

        return UxDataArray(scalar_dot_gradient, uxgrid=self.uxgrid)

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
                raise DataCenteringError(
                    "Invalid destination 'face' for a face-centered data variable, computing"
                    "the difference and storing it on each face is not possible"
                )
            elif destination == "node":
                raise DataCenteringError(
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
                raise DataCenteringError(
                    "Invalid destination 'node' for a node-centered data variable, computing"
                    "the difference and storing it on each node is not possible"
                )

            elif destination == "face":
                raise DataCenteringError(
                    "Support for computing the difference of a node-centered data variable and storing"
                    "the result on each face not yet supported."
                )

        elif self._edge_centered():
            raise NotImplementedError(
                "Difference for edge centered data variables not yet implemented"
            )

        else:
            raise DataCenteringError("TODO: ")

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
        """Return a new UxDataArray indexed along the specified dimension(s).
        The data is indexed, as well as the underlying grid when applicable.

        Grid dimensions ('n_node', 'n_edge', 'n_face') are treated specially
        when `ignore_grid=False` (this is the default). Any one of them can be indexed,
        regardless of data location, and the result will be sliced to form the minimal grid
        of faces containing all the nodes, edges, or faces specified. For example,
        using n_edge=7 selects just the two faces touching edge 7. For data on 'n_face',
        the result would have 'n_face' with just those two faces. For data on 'n_edge',
        the result would have 'n_edge' with all edges located on either of those two faces.
        Grid dimension indexers cannot have more than 1 dimension (such as a 2D DataArray).

        Parameters
        ----------
        indexers : Mapping[Any, Any], optional
            A dict with keys matching dimensions and values given
            by integers, slice objects or arrays.
            indexer can be a integer, slice, array-like or DataArray.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        drop : bool, default=False
            If ``drop=True``, drop coordinates variables indexed by integers
            instead of making them scalar.
        missing_dims : {'raise', 'warn', 'ignore'}, default='raise'
            What to do if dimensions that should be selected from are not present in the
            UxDataArray:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions
        ignore_grid : bool, default=False
            If False (default), slice the underlying UXarray grid appropriately too,
            ensuring the resulting data actually lies on the result's underlying grid.
            If True, slice the data only; attach self.uxgrid to the result, unchanged.
            CAUTION: using ignore_grid=True will cause the result's data to be
            inconsistent with its underlying grid, if any grid dimensions were sliced.
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
        DimensionError (subclass of ValueError)
            If more than one grid dimension is selected and `ignore_grid=False`.
        ValueError
            If parameters are invalid for xarray's .isel(), such as if
            slicing by a nonexistent dimension, or using invalid indexers.
        """
        indexers, grid_dims = _validate_indexers(
            indexers, indexers_kwargs, "isel", ignore_grid
        )

        if ignore_grid or len(grid_dims) == 0:
            # no grid dims, or ignore_grid=True --> just call xarray's isel
            return type(self)(
                super().isel(
                    indexers=indexers or None,
                    drop=drop,
                    missing_dims=missing_dims,
                ),
                uxgrid=self.uxgrid,
            )
        elif len(grid_dims) == 1:
            # pop off the one grid‐dim indexer
            grid_dim = grid_dims.pop()
            indexers = indexers.copy()  # don't modify the original dict
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
        else:  # len(grid_dims)>1; _validate_indexers should have crashed.
            raise AssertionError("internal implementation error if reached this line")

    def sel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        method: str | None = None,
        tolerance: int | float | Iterable[int | float] | None = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ):
        """Returns a new array indexed by labels, instead of indices, along the specified dimension(s).

        Grid dimensions ('n_node', 'n_edge', 'n_face') are treated specially. Any one of them
        can be indexed, regardless of data location, and the result will be sliced to form the
        minimal grid of faces containing all the nodes, edges, or faces specified. For example,
        using n_edge=7 selects just the two faces touching edge 7. For data on 'n_face',
        the result would have 'n_face' with just those two faces. For data on 'n_edge',
        the result would have 'n_edge' with all edges located on either of those two faces.
        Grid dimension indexers cannot have more than 1 dimension (such as a 2D DataArray).

        By default, grid dims do not have coordinates assigned. But, if they have
        been assigned, `.sel()` respects them in the intuitive way. For example,
        using `.sel(n_face=30)` for data with `n_face` coordinates [0,10,20,30,40]
        would be equivalent to using `.isel(n_face=3)`. Meanwhile, if the data
        does not contain the specified grid dim (as in the n_edge=7 example above),
        it also cannot contain coordinates along that grid dim,
        so in that case `.sel()` performs index-based selection just like `.isel()`.

        Under the hood, this method is powered by using pandas's powerful Index
        objects. This makes label based indexing essentially just as fast as
        using integer indexing.

        It also means this method uses pandas's (well documented) logic for
        indexing. This means you can use string shortcuts for datetime indexes
        (e.g., '2000-01' to select all values in January 2000). It also means
        that slices are treated as inclusive of both the start and stop values,
        unlike normal Python indexing, for any dimensions with coordinate labels.
        (Dimensions without coordinates treat slices normally.)

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by scalars, slices or arrays of tick labels. For dimensions with
            multi-index, the indexer may also be a dict-like object with keys
            matching index level names.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method to use for inexact matches:

            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value

            Can only provide ``method`` if all indexed dims actually have coords,
            else raises ValueError (consistent with xarray sel() behavior).
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Can only provide ``tolerance`` if all indexed dims actually have coords,
            else raises ValueError (consistent with xarray sel() behavior).
        drop : bool, optional
            If ``drop=True``, drop coordinates variables in `indexers` instead
            of making them scalar.
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : UxDataArray
            A new UxDataArray with each dimension is indexed appropriately,
            and the uxgrid indexed appropriately as well, if indexing any grid dim.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached,
            except for indexers along a grid dimension (see issue #1712).
            In general, the result's data will be a view of the data in this array,
            unless indexing along a grid dimension or otherwise
            triggering vectorized indexing by using an array indexer,
            in which case the data will be a copy.
        """
        indexers, grid_dims = _validate_indexers(
            indexers, indexers_kwargs, "sel", ignore_grid=False
        )  # (sel doesn't support ignore_grid=True option)

        if len(grid_dims) == 0:
            # no grid dims --> just call xarray's sel
            return type(self)(
                self.to_xarray().sel(
                    indexers=indexers,
                    method=method,
                    tolerance=tolerance,
                    drop=drop,
                ),
                uxgrid=self.uxgrid,
            )
        elif len(grid_dims) == 1:
            # pop off the one grid‐dim indexer
            grid_dim = list(grid_dims)[0]
            indexers = indexers.copy()  # don't modify the original dict
            grid_indexer = indexers.pop(grid_dim)
            if grid_dim in self.coords:  # label-based indexing
                grid_indices = _resolve_coordinate_labels_to_indices(
                    grid_dim,
                    grid_indexer,
                    self.coords[grid_dim],
                    method=method,
                    tolerance=tolerance,
                )
            else:  # index-based indexing
                # crash if provided `method` or `tolerance`, as promised in docstring;
                if method is not None or tolerance is not None:
                    raise ValueError(
                        f"cannot supply selection options {dict(method=method, tolerance=tolerance)} "
                        f"for dimension {grid_dim!r} that has no associated coordinate or index"
                    )
                grid_indices = grid_indexer

            # offload the grid-indexing work to isel():
            result = self.isel({grid_dim: grid_indices}, drop=drop)

            # index by other dims if any remain:
            ds = result.to_xarray().sel(
                indexers=indexers,  # (grid_dim indexer was popped)
                method=method,
                tolerance=tolerance,
                drop=drop,
            )

            return type(self)(ds, uxgrid=result.uxgrid)
        else:  # len(grid_dims)>1; _validate_indexers should have crashed.
            raise AssertionError("internal implementation error if reached this line")

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
            raise DimensionError(
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
                n_face=sliced_grid._ds["_subgrid_face_indices"], ignore_grid=True
            )

        elif self._edge_centered():
            da_sliced = self.isel(
                n_edge=sliced_grid._ds["_subgrid_edge_indices"], ignore_grid=True
            )

        elif self._node_centered():
            da_sliced = self.isel(
                n_node=sliced_grid._ds["_subgrid_node_indices"], ignore_grid=True
            )

        else:
            raise DataCenteringError(
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
            raise GridInvalidError("Duplicate nodes found, cannot construct dual")

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

        # Construct the new data array
        uxda = uxarray.UxDataArray(
            uxgrid=dual, data=self.data, dims=dims, name=self.name
        )

        return uxda

    def _neighborhood_location(self, caller: str) -> str:
        """Grid location this data is mapped to, in ``Neighborhood`` terms."""
        if self._face_centered():
            return "face centers"
        if self._node_centered():
            return "nodes"
        if self._edge_centered():
            return "edge centers"
        raise DataCenteringError(
            f"`{caller}()` requires data mapped to nodes, edges, or faces, "
            f"but the dimensions {self.dims!r} do not match any grid dimension "
            f"{GRID_DIMS}."
        )

    def neighborhood(self, r: float = 1.0) -> DataArrayNeighborhood:
        """Groups this data by the elements within ``r`` degrees of each grid
        element, to be reduced over by a method of the returned
        :class:`DataArrayNeighborhood`.

        Each reduction replaces the value at every grid element with a
        reduction of all elements within a circular neighborhood of radius
        ``r``, as in a smoothing filter.

        Parameters
        ----------
        r : float, default=1.
            Radius of the neighborhood, in degrees.

        Returns
        -------
        DataArrayNeighborhood
            Bound to this data, so its reduction methods take only the
            parameters of the reduction: ``mean()``, ``sum()``, ``min()``,
            ``max()``, ``median()``, ``ptp()``, ``std(ddof)``, ``var(ddof)``,
            ``quantile(q)``, ``percentile(q)``, or ``reduce(func)`` for
            anything else. Each returns a ``UxDataArray`` of float64.

        Raises
        ------
        DataCenteringError (subclass of ValueError)
            If the data is not mapped to nodes, edges, or faces.

        Notes
        -----
        ``r`` is a great-circle distance in degrees. An element's neighborhood
        overlaps those of the elements around it, and every element is its own
        neighbor at distance 0, so ``r = 0`` returns the data unchanged and the
        result never contains spurious ``NaN``.

        Building this queries the grid for neighbors, which usually costs more
        than the reduction itself. That query is what the returned object holds
        on to, so several reductions at one radius should share one call rather
        than repeat it. To share it across variables too, build the
        neighborhood from the grid instead, with :meth:`Grid.neighborhood`.

        A neighborhood may span the whole grid, so the grid dimension cannot be
        chunked; it is collapsed to a single chunk (with a warning) for
        dask-backed data. The remaining dimensions stay chunked and lazy, so
        chunk along ``time`` rather than the grid dimension.

        Examples
        --------
        Apply a mean filter with a 5-degree radius:

        >>> import uxarray as ux
        >>> uxds = ux.tutorial.open_dataset("outCSne30-vortex")
        >>> uxda = uxds["psi"]
        >>> smoothed = uxda.neighborhood(r=5.0).mean()

        Reductions taking a parameter receive it as a keyword argument:

        >>> p90 = uxda.neighborhood(r=5.0).percentile(90)
        >>> spread = uxda.neighborhood(r=5.0).std(ddof=1)

        Several reductions at one radius share the neighbor query:

        >>> nb = uxda.neighborhood(r=5.0)
        >>> smoothed, spread = nb.mean(), nb.std()

        See Also
        --------
        DataArrayNeighborhood : The reductions available on the returned object.
        Grid.neighborhood : Neighborhood shared across several variables.
        UxDataArray.topological_mean : Aggregate values across neighboring grid element types.
        UxDataArray.zonal_mean : Average over latitude bands.
        UxDataArray.azimuthal_mean : Average over rings of constant great-circle distance.
        """
        neighborhood = Neighborhood(
            self.uxgrid,
            r=r,
            on=self._neighborhood_location("neighborhood"),
        )
        return DataArrayNeighborhood(neighborhood, self)

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
                return accessor_class(result, self._uxgrid)

            # Copy the docstring from the parent method
            method.__doc__ = parent_method.__doc__
            method.__name__ = name

            return method

        # For all other attributes, use the default behavior
        return super().__getattribute__(name)

    def where(self, cond: Any, other: Any = dtypes.NA, drop: bool = False):
        return UxDataArray(super().where(cond, other, drop), uxgrid=self._uxgrid)

    where.__doc__ = xr.DataArray.where.__doc__

    def fillna(self, value: Any):
        return UxDataArray(super().fillna(value), uxgrid=self._uxgrid)

    fillna.__doc__ = xr.DataArray.fillna.__doc__
