from __future__ import annotations

import os
import sys
from html import escape
from typing import IO, Any, Mapping
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core import dtypes
from xarray.core.options import OPTIONS
from xarray.core.utils import UncachedAccessor

import uxarray
from uxarray.core.dataarray import UxDataArray
from uxarray.core.utils import _map_dims_to_ugrid, _open_dataset_with_fallback
from uxarray.formatting_html import dataset_repr
from uxarray.grid import Grid
from uxarray.grid.dual import construct_dual
from uxarray.grid.validation import _check_duplicate_nodes_indices
from uxarray.io._healpix import get_zoom_from_cells
from uxarray.plot.accessor import UxDatasetPlotAccessor
from uxarray.remap.accessor import RemapAccessor


class UxDataset(xr.Dataset):
    """Grid informed ``xarray.Dataset`` with an attached ``Grid`` accessor and
    grid-specific functionality.

    Parameters
    ----------
    uxgrid : uxarray.Grid, optional
        The ``Grid`` object that makes this array aware of the unstructured
        grid topology it belongs to.

        If ``None``, it needs to be an instance of ``uxarray.Grid``.

    Other Parameters
    ----------------
    *args:
        Arguments for the ``xarray.Dataset`` class
    **kwargs:
        Keyword arguments for the ``xarray.Dataset`` class

    Notes
    -----
    See `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`__
    for further information about Datasets.

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
    __slots__ = (
        "_uxgrid",
        "_source_datasets",
    )

    def __init__(
        self,
        *args,
        uxgrid: Grid = None,
        source_datasets: str | None = None,
        **kwargs,
    ):
        self._uxgrid = None
        self._source_datasets = source_datasets
        # setattr(self, 'source_datasets', source_datasets)

        if uxgrid is not None and not isinstance(uxgrid, Grid):
            raise RuntimeError(
                "uxarray.UxDataset.__init__: uxgrid can be either None or "
                "an instance of the `uxarray.Grid` class"
            )
        else:
            self._uxgrid = uxgrid

        super().__init__(*args, **kwargs)

    # declare plotting accessor
    plot = UncachedAccessor(UxDatasetPlotAccessor)
    remap = UncachedAccessor(RemapAccessor)

    def _repr_html_(self) -> str:
        if OPTIONS["display_style"] == "text":
            return f"<pre>{escape(repr(self))}</pre>"
        return dataset_repr(self)

    def __getitem__(self, key):
        """Override to make sure the result is an instance of
        ``uxarray.UxDataArray`` or ``uxarray.UxDataset``."""

        value = super().__getitem__(key)

        if isinstance(value, xr.DataArray):
            value = UxDataArray(value, uxgrid=self.uxgrid)
        elif isinstance(value, xr.Dataset):
            value = UxDataset(
                value, uxgrid=self.uxgrid, source_datasets=self.source_datasets
            )

        return value

    # def __setitem__(self, key, value):
    #     """Override to make sure the `value` is an instance of
    #     ``uxarray.UxDataArray``."""
    #     if isinstance(value, xr.DataArray):
    #         value = UxDataArray(value, uxgrid=self.uxgrid)
    #
    #     if isinstance(value, UxDataArray):
    #         value = value.to_dataarray()
    #
    #     super().__setitem__(key, value)

    @property
    def source_datasets(self):
        """Property to keep track of the source data sets used to instantiate
        this ``uxarray.UxDataset``.

        Can be used as metadata for diagnosis purposes.

        Examples
        --------
        uxds = ux.open_dataset(grid_path, data_path)
        uxds.source_datasets
        """
        return self._source_datasets

    # a setter function
    @source_datasets.setter
    def source_datasets(self, source_datasets_input):
        self._source_datasets = source_datasets_input

    @property
    def uxgrid(self):
        """Linked ``Grid`` representing to the unstructured grid the data
        resides on."""
        return self._uxgrid

    # a setter function
    @uxgrid.setter
    def uxgrid(self, ugrid_obj):
        self._uxgrid = ugrid_obj

    def _calculate_binary_op(self, *args, **kwargs):
        """Override to make the result a complete instance of
        ``uxarray.UxDataset``."""
        ds = super()._calculate_binary_op(*args, **kwargs)

        if isinstance(ds, UxDataset):
            ds.uxgrid = self.uxgrid
            ds.source_datasets = self.source_datasets
        else:
            ds = UxDataset(ds, uxgrid=self.uxgrid, source_datasets=self.source_datasets)

        return ds

    def _construct_dataarray(self, name) -> UxDataArray:
        """Override to make the result an instance of
        ``uxarray.UxDataArray``."""
        xarr = super()._construct_dataarray(name)
        return UxDataArray(xarr, uxgrid=self.uxgrid)

    @classmethod
    def _construct_direct(cls, *args, **kwargs):
        """Override to make the result an ``uxarray.UxDataset`` class."""

        return cls(xr.Dataset._construct_direct(*args, **kwargs))

    def _copy(self, **kwargs):
        """Override to make the result a complete instance of
        ``uxarray.UxDataset``."""
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
        ``uxarray.UxDataset``."""
        ds = super()._replace(*args, **kwargs)

        if isinstance(ds, UxDataset):
            ds.uxgrid = self.uxgrid
            ds.source_datasets = self.source_datasets
        else:
            ds = UxDataset(ds, uxgrid=self.uxgrid, source_datasets=self.source_datasets)

        return ds

    @classmethod
    def from_dataframe(cls, dataframe):
        """Override to make the result a ``uxarray.UxDataset`` class."""

        return cls(
            {col: ("index", dataframe[col].values) for col in dataframe.columns},
            coords={"index": dataframe.index},
        )

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Override to make the result a ``uxarray.UxDataset`` class."""

        return cls(
            {key: ("index", val) for key, val in data.items()},
            coords={"index": range(len(next(iter(data.values()))))},
            **kwargs,
        )

    @classmethod
    def from_structured(cls, ds: xr.Dataset):
        """Converts a structured ``xarray.Dataset`` into an unstructured ``uxarray.UxDataset``

        Parameters
        ----------
        ds : xr.Dataset
            The structured `xarray.Dataset` to convert. Must contain longitude and latitude variables consistent
            with the CF-conventions

        tol : float, optional
            Tolerance for considering nodes as identical when constructing the grid from longitude and latitude.
            Default is `1e-10`.

        Returns
        -------
        UxDataset
            An instance of `uxarray.UxDataset`
        """
        from uxarray import Grid

        uxgrid = Grid.from_dataset(ds)

        ds = _map_dims_to_ugrid(ds, uxgrid._source_dims_dict, uxgrid)

        # Drop spatial coordinates
        coords_to_drop = [
            coord for coord, da_coord in ds.coords.items() if "n_face" in da_coord.dims
        ]
        ds = ds.drop_vars(coords_to_drop)

        return cls(ds, uxgrid=uxgrid)

    @classmethod
    def from_xarray(cls, ds: xr.Dataset, uxgrid: Grid = None, ugrid_dims: dict = None):
        """
        Converts a ``xarray.Dataset`` into a ``uxarray.UxDataset``, paired with either a user-defined or
        parsed ``Grid``

        Parameters
        ----------
        ds: xr.Dataset
            An Xarray dataset containing data residing on an unstructured grid
        uxgrid: Grid, optional
            ``Grid`` object representing an unstructured grid. If a grid is not provided, the source ds will be
            parsed to see if a ``Grid`` can be constructed.
        ugrid_dims: dict, optional
            A dictionary mapping dataset dimensions to UGRID dimensions.

        Returns
        -------
        cls
            A ``ux.UxDataset`` with data from the ``xr.Dataset` paired with a ``ux.Grid``
        """
        if uxgrid is not None:
            if ugrid_dims is None and uxgrid._source_dims_dict is not None:
                ugrid_dims = uxgrid._source_dims_dict
            # Grid is provided,
        else:
            # parse
            uxgrid = Grid.from_dataset(ds)
            ugrid_dims = uxgrid._source_dims_dict

        # map each dimension to its UGRID equivalent
        ds = _map_dims_to_ugrid(ds, ugrid_dims, uxgrid)

        return cls(ds, uxgrid=uxgrid)

    @classmethod
    def from_healpix(
        cls,
        ds: str | os.PathLike | xr.Dataset,
        pixels_only: bool = True,
        face_dim: str = "cell",
        **kwargs,
    ):
        """
        Loads a dataset represented in the HEALPix format into a ``ux.UxDataSet``, paired
        with a ``Grid`` containing information about the HEALPix definition.

        Parameters
        ----------
        ds: str, os.PathLike, xr.Dataset
            Reference to a HEALPix Dataset
        pixels_only : bool, optional
            Whether to only compute pixels (`face_lon`, `face_lat`) or to also construct boundaries (`face_node_connectivity`, `node_lon`, `node_lat`)
        face_dim: str, optional
            Data dimension corresponding to the HEALPix face mapping. Typically, is set to "cell", but may differ.

        Returns
        -------
        cls
            A ``ux.UxDataset`` instance
        """

        if not isinstance(ds, xr.Dataset):
            ds = _open_dataset_with_fallback(ds, **kwargs)

        if face_dim not in ds.dims:
            raise ValueError(
                f"The provided face dimension '{face_dim}' is not present in the provided healpix dataset."
                f"Please set 'face_dim' to the dimension corresponding to the healpix face dimension."
            )

        # Attach a HEALPix Grid
        uxgrid = Grid.from_healpix(
            zoom=get_zoom_from_cells(ds.sizes[face_dim]),
            pixels_only=pixels_only,
            **kwargs,
        )

        return cls.from_xarray(ds, uxgrid, {face_dim: "n_face"})

    def _slice_dataset_from_grid(self, sliced_grid, grid_dim: str, grid_indexer):
        data_vars = {}
        for name, da in self.data_vars.items():
            if grid_dim in da.dims:
                if hasattr(da, "_slice_from_grid"):
                    data_vars[name] = da._slice_from_grid(sliced_grid)
                else:
                    data_vars[name] = da.isel({grid_dim: grid_indexer})
            else:
                data_vars[name] = da

        coords = {}
        for cname, cda in self.coords.items():
            if grid_dim in cda.dims:
                # Prefer authoritative coords from the sliced grid if available
                replacement = getattr(sliced_grid, cname, None)
                coords[cname] = (
                    replacement
                    if replacement is not None
                    else cda.isel({grid_dim: grid_indexer})
                )
            else:
                coords[cname] = cda

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self.attrs)

        return ds

    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        drop: bool = False,
        missing_dims: str = "raise",
        ignore_grid: bool = False,
        inverse_indices: bool = False,
        **indexers_kwargs,
    ):
        """Returns a new dataset with each array indexed along the specified
        dimension(s).

        Performs xarray-style integer-location indexing along specified dimensions.
        If a single grid dimension ('n_node', 'n_edge', or 'n_face') is provided
        and `ignore_grid=False`, the underlying grid is sliced accordingly,
        and remaining indexers are applied to the resulting Dataset.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by integers, slice objects or arrays.
            indexer can be a integer, slice, array-like or DataArray.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        drop : bool, default: False
            If ``drop=True``, drop coordinates variables indexed by integers
            instead of making them scalar.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            Dataset:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions
        ignore_grid : bool, default=False
            If False (default), allow slicing on one grid dimension to automatically
            update the associated UXarray grid. If True, fall back to pure xarray behavior.
        inverse_indices : bool, default=False
            For grid-based slicing, pass this flag to `Grid.isel` to invert indices
            when selecting (useful for staggering or reversing order).
        **indexers_kwargs : dimension=indexer pairs, optional

        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

                Returns
        -------
        UxDataset
            A new UxDataset indexed according to `indexers` and updated grid if applicable.
        """
        from uxarray.core.utils import _validate_indexers

        indexers, grid_dims = _validate_indexers(
            indexers, indexers_kwargs, "isel", ignore_grid
        )

        if not ignore_grid:
            if len(grid_dims) == 1:
                grid_dim = grid_dims.pop()
                grid_indexer = indexers.pop(grid_dim)

                # slice the grid
                sliced_grid = self.uxgrid.isel(
                    **{grid_dim: grid_indexer}, inverse_indices=inverse_indices
                )

                ds = self._slice_dataset_from_grid(
                    sliced_grid=sliced_grid,
                    grid_dim=grid_dim,
                    grid_indexer=grid_indexer,
                )

                if indexers:
                    ds = xr.Dataset.isel(
                        ds, indexers=indexers, drop=drop, missing_dims=missing_dims
                    )

                return type(self)(ds, uxgrid=sliced_grid)
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

    def __getattribute__(self, name):
        """Intercept accessor method calls to return Ux-aware accessors."""
        # Lazy import to avoid circular imports
        from uxarray.core.accessors import DATASET_ACCESSOR_METHODS

        if name in DATASET_ACCESSOR_METHODS:
            from uxarray.core import accessors

            # Get the accessor class by name
            accessor_class = getattr(accessors, DATASET_ACCESSOR_METHODS[name])

            # Get the parent method
            parent_method = super().__getattribute__(name)

            # Create a wrapper method
            def method(*args, **kwargs):
                # Call the parent method
                result = parent_method(*args, **kwargs)
                # Wrap the result with our accessor
                return accessor_class(result, self.uxgrid, self.source_datasets)

            # Copy the docstring from the parent method
            method.__doc__ = parent_method.__doc__
            method.__name__ = name

            return method

        # For all other attributes, use the default behavior
        return super().__getattribute__(name)

    def info(self, buf: IO = None, show_attrs=False) -> None:
        """Concise summary of Dataset variables and attributes including grid
        topology information stored in the ``uxgrid`` property.

        Parameters
        ----------
        buf : file-like, default: sys.stdout
            writable buffer
        show_attrs : bool
            Flag to select whether to show attributes

        See Also
        --------
        pandas.DataFrame.assign
        ncdump : netCDF's ncdump
        """
        if buf is None:  # pragma: no cover
            buf = sys.stdout

        lines = []
        lines.append("uxarray.Dataset {")

        lines.append("grid topology dimensions:")
        for name, size in self.uxgrid._ds.sizes.items():
            lines.append(f"\t{name} = {size}")

        lines.append("\ngrid topology variables:")
        for name, da in self.uxgrid._ds.variables.items():
            dims = ", ".join(map(str, da.dims))
            lines.append(f"\t{da.dtype} {name}({dims})")
            if show_attrs:
                for k, v in da.attrs.items():
                    lines.append(f"\t\t{name}:{k} = {v}")

        lines.append("\ndata dimensions:")
        for name, size in self.sizes.items():
            lines.append(f"\t{name} = {size}")

        lines.append("\ndata variables:")
        for name, da in self.variables.items():
            dims = ", ".join(map(str, da.dims))
            lines.append(f"\t{da.dtype} {name}({dims})")
            if show_attrs:
                for k, v in da.attrs.items():
                    lines.append(f"\t\t{name}:{k} = {v}")

        if show_attrs:
            lines.append("\nglobal attributes:")
            for k, v in self.attrs.items():
                lines.append(f"\t:{k} = {v}")

        lines.append("}")
        buf.write("\n".join(lines))

    def integrate(self, quadrature_rule="triangular", order=4):
        """Integrates over all the faces of the givfen mesh.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        Calculated integral : float

        Examples
        --------
        Open a Uxarray dataset

        >>> import uxarray as ux
        >>> uxds = ux.open_dataset("grid.ug", "centroid_pressure_data_ug")

        # Compute the integral
        >>> integral = uxds.integrate()
        """

        # TODO: Deprecation Warning
        warn(
            "This method currently only works when there is a single DataArray in this Dataset. For integration of a "
            "single data variable, use the UxDataArray.integrate() method instead. This function will be deprecated and "
            "replaced with one that can perform a Dataset-wide integration in a future release.",
            DeprecationWarning,
        )

        integral = 0.0

        face_areas = self.uxgrid.face_areas.values

        # TODO: Should we fix this requirement? Shouldn't it be applicable to
        # TODO: all variables of dataset or a dataarray instead?
        var_key = list(self.keys())
        if len(var_key) > 1:
            # warning: print message
            print(
                "WARNING: The dataset has more than one variable, using the first variable for integration"
            )

        var_key = var_key[0]
        face_vals = self[var_key].to_numpy()
        integral = np.dot(face_areas, face_vals)

        return integral

    def to_array(self) -> UxDataArray:
        """Override to make the result an instance of
        ``uxarray.UxDataArray``."""

        xarr = super().to_array()
        return UxDataArray(xarr, uxgrid=self.uxgrid)

    def to_xarray(self, grid_format: str = "UGRID") -> xr.Dataset:
        """
        Converts a ``ux.UXDataset`` to a ``xr.Dataset``.

        Parameters
        ----------
        grid_format : str, default="UGRID"
            The format in which to convert the grid. Supported values are "UGRID" and "HEALPix". The dimensions will
            match the selected grid format.

        Returns
        -------
        xr.Dataset
            The ``ux.UXDataset`` represented as a ``xr.Dataset``
        """
        if grid_format == "HEALPix":
            ds = self.rename_dims({"n_face": "cell"})
            return xr.Dataset(ds)

        return xr.Dataset(self)

    def get_dual(self):
        """Compute the dual mesh for a dataset, returns a new dataset object.

        Returns
        --------
        dual : uxds
            Dual Mesh `uxds` constructed
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

        # Initialize new dataset
        dataset = uxarray.UxDataset(uxgrid=dual)

        # Dictionary to swap dimensions
        dim_map = {"n_face": "n_node", "n_node": "n_face"}

        # For each data array in the dataset, reconstruct the data array with the dual mesh
        for var in self.data_vars:
            # Get correct dimensions for the dual
            dims = [dim_map.get(dim, dim) for dim in self[var].dims]

            # Get the values from the data array
            data = np.array(self[var].values)

            # Construct the new data array
            uxda = uxarray.UxDataArray(uxgrid=dual, data=data, dims=dims, name=var)

            # Add data array to dataset
            dataset[var] = uxda

        return dataset

    def where(self, cond: Any, other: Any = dtypes.NA, drop: bool = False):
        return UxDataset(self.to_xarray().where(cond, other, drop), uxgrid=self.uxgrid)

    where.__doc__ = xr.Dataset.where.__doc__

    def sel(
        self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs
    ):
        return UxDataset(
            self.to_xarray().sel(
                indexers=indexers,
                method=method,
                tolerance=tolerance,
                drop=drop,
                **indexers_kwargs,
            ),
            uxgrid=self.uxgrid,
        )

    sel.__doc__ = xr.Dataset.sel.__doc__

    def fillna(self, value: Any):
        return UxDataset(super().fillna(value), uxgrid=self.uxgrid)

    fillna.__doc__ = xr.Dataset.fillna.__doc__
