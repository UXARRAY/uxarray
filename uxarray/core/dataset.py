from __future__ import annotations

import numpy as np
import xarray as xr

import sys

from typing import Optional, IO, Union

from uxarray.grid import Grid
from uxarray.core.dataarray import UxDataArray

from uxarray.plot.accessor import UxDatasetPlotAccessor

from xarray.core.utils import UncachedAccessor

from uxarray.remap import UxDatasetRemapAccessor

from warnings import warn


class UxDataset(xr.Dataset):
    """A ``xarray.Dataset``-like, multi-dimensional, in memory, array database.
    Inherits from ``xarray.Dataset`` and has its own unstructured grid-aware
    dataset operators and attributes through the ``uxgrid`` accessor.

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
        source_datasets: Optional[str] = None,
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
            self.uxgrid = uxgrid

        super().__init__(*args, **kwargs)

    # declare plotting accessor
    plot = UncachedAccessor(UxDatasetPlotAccessor)
    remap = UncachedAccessor(UxDatasetRemapAccessor)

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
        """``uxarray.Grid`` property for ``uxarray.UxDataset`` to make it
        unstructured grid-aware.

        Examples
        --------
        uxds = ux.open_dataset(grid_path, data_path)
        uxds.uxgrid
        """
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

        # call function to get area of all the faces as a np array
        face_areas, face_jacobian = self.uxgrid.compute_face_areas(
            quadrature_rule, order
        )

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

    def nearest_neighbor_remap(
        self,
        destination_obj: Union[Grid, UxDataArray, UxDataset],
        remap_to: str = "nodes",
        coord_type: str = "spherical",
    ):
        """Nearest Neighbor Remapping between a source (``UxDataset``) and
        destination.`.

        Parameters
        ---------
        destination_obj : Grid, UxDataArray, UxDataset
            Destination for remapping
        remap_to : str, default="nodes"
            Location of where to map data, either "nodes", "edge centers", or "face centers"
        coord_type : str, default="spherical"
            Indicates whether to remap using on spherical or cartesian coordinates
        """
        warn(
            "This usage of remapping will be deprecated in a future release. It is advised to use uxds.remap.nearest_neighbor() instead.",
            DeprecationWarning,
        )

        return self.remap.nearest_neighbor(destination_obj, remap_to, coord_type)

    def inverse_distance_weighted_remap(
        self,
        destination_obj: Union[Grid, UxDataArray, UxDataset],
        remap_to: str = "nodes",
        coord_type: str = "spherical",
        power=2,
        k=8,
    ):
        """Inverse Distance Weighted Remapping between a source (``UxDataset``)
        and destination.`.

        Parameters
        ---------
        destination_obj : Grid, UxDataArray, UxDataset
            Destination for remapping
        remap_to : str, default="nodes"
            Location of where to map data, either "nodes", "edge centers", or "face centers"
        coord_type : str, default="spherical"
            Indicates whether to remap using on spherical or cartesian coordinates
        power : int, default=2
            Power parameter for inverse distance weighting. This controls how local or global the remapping is, a higher
            power causes points that are further away to have less influence
        k : int, default=8
            Number of nearest neighbors to consider in the weighted calculation.
        """
        warn(
            "This usage of remapping will be deprecated in a future release. It is advised to use uxds.remap.inverse_distance_weighted() instead.",
            DeprecationWarning,
        )

        return self.remap.inverse_distance_weighted(
            destination_obj, remap_to, coord_type, power, k
        )
