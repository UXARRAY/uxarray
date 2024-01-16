from __future__ import annotations

import xarray as xr
import numpy as np

from typing import TYPE_CHECKING, Optional, Union

from uxarray.grid import Grid
import uxarray.core.dataset

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset

from xarray.core.utils import UncachedAccessor

from uxarray.remap.nearest_neighbor import _nearest_neighbor_uxda

from uxarray.plot.accessor import UxDataArrayPlotAccessor
from uxarray.subset import DataArraySubsetAccessor


class UxDataArray(xr.DataArray):
    """N-dimensional ``xarray.DataArray``-like array. Inherits from
    ``xarray.DataArray`` and has its own unstructured grid-aware array
    operators and attributes through the ``uxgrid`` accessor.

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
        """``uxarray.Grid`` property for ``uxarray.UxDataArray`` to make it
        unstructured grid-aware.

        Examples
        --------
        uxds = ux.open_dataset(grid_path, data_path)
        uxds.<variable_name>.uxgrid
        """
        return self._uxgrid

    # a setter function
    @uxgrid.setter
    def uxgrid(self, ugrid_obj):
        self._uxgrid = ugrid_obj

    def to_geodataframe(self, override=False, cache=True, exclude_antimeridian=False):
        """Constructs a ``spatialpandas.GeoDataFrame`` with a "geometry"
        column, containing a collection of Shapely Polygons or MultiPolygons
        representing the geometry of the unstructured grid, and a data column
        representing a 1D slice of data mapped to each Polygon.

        Parameters
        override: bool
            Flag to recompute the ``GeoDataFrame`` stored under the ``uxgrid`` if one is already cached
        cache: bool
            Flag to indicate if the computed ``GeoDataFrame`` stored under the ``uxgrid`` accessor should be cached
        exclude_antimeridian: bool, Optional
            Selects whether to exclude any face that contains an edge that crosses the antimeridian

        Returns
        -------
        gdf : spatialpandas.GeoDataFrame
            The output `GeoDataFrame` with a filled out "geometry" and 1D data column representing the geometry of the unstructured grid
        """

        if self.values.ndim > 1:
            # data is multidimensional, must be a 1D slice
            raise ValueError(
                f"Data Variable must be 1-dimensional, with shape {self.uxgrid.n_face} "
                f"for face-centered data."
            )

        if self.values.size == self.uxgrid.n_face:
            gdf = self.uxgrid.to_geodataframe(
                override=override,
                cache=cache,
                exclude_antimeridian=exclude_antimeridian,
            )

            if exclude_antimeridian:
                gdf[self.name] = np.delete(
                    self.values, self.uxgrid.antimeridian_face_indices, axis=0
                )
            else:
                gdf[self.name] = self.values
            return gdf

        elif self.values.size == self.uxgrid.n_node:
            raise ValueError(
                f"Data Variable with size {self.values.size} does not match the number of faces "
                f"({self.uxgrid.n_face}. Current size matches the number of nodes."
            )

        # data not mapped to faces or nodes
        else:
            raise ValueError(
                f"Data Variable with size {self.values.size} does not match the number of faces "
                f"({self.uxgrid.n_face}."
            )

    def to_polycollection(
        self, override=False, cache=True, correct_antimeridian_polygons=True
    ):
        """Constructs a ``matplotlib.collections.PolyCollection`` object with
        polygons representing the geometry of the unstructured grid, with
        polygons that cross the antimeridian split across the antimeridian.

        Parameters
        ----------
        override : bool
            Flag to recompute the ``PolyCollection`` stored under the ``uxgrid`` if one is already cached
        cache : bool
            Flag to indicate if the computed ``PolyCollection`` stored under the ``uxgrid`` accessor should be cached
        correct_antimeridian_polygons: bool, Optional
            Parameter to select whether to correct and split antimeridian polygons

        Returns
        -------
        poly_collection : matplotlib.collections.PolyCollection
            The output `PolyCollection` of polygons representing the geometry of the unstructured grid paired with
            a data variable.
        """

        # data is multidimensional, must be a 1D slice
        if self.values.ndim > 1:
            raise ValueError(
                f"Data Variable must be 1-dimensional, with shape {self.uxgrid.n_face} "
                f"for face-centered data."
            )

        # face-centered data
        if self.values.size == self.uxgrid.n_face:
            (
                poly_collection,
                corrected_to_original_faces,
            ) = self.uxgrid.to_polycollection(
                override=override,
                cache=cache,
                correct_antimeridian_polygons=correct_antimeridian_polygons,
            )

            # map data with antimeridian polygons
            if len(corrected_to_original_faces) > 0:
                data = self.values[corrected_to_original_faces]

            # no antimeridian polygons
            else:
                data = self.values

            poly_collection.set_array(data)
            return poly_collection, corrected_to_original_faces

        # node-centered data
        elif self.values.size == self.uxgrid.n_node:
            raise ValueError(
                f"Data Variable with size {self.values.size} mapped on the nodes of each polygon"
                f"not supported yet."
            )

        # data not mapped to faces or nodes
        else:
            raise ValueError(
                f"Data Variable with size {self.values.size} does not match the number of faces "
                f"({self.uxgrid.n_face}."
            )

    def to_dataset(self) -> UxDataset:
        """Converts a ``UxDataArray`` into a ``UxDataset`` with a single data
        variable."""
        xrds = super().to_dataset()
        return uxarray.core.dataset.UxDataset(xrds, uxgrid=self.uxgrid)

    def nearest_neighbor_remap(
        self,
        destination_obj: Union[Grid, UxDataArray, UxDataset],
        remap_to: str = "nodes",
        coord_type: str = "spherical",
    ):
        """Nearest Neighbor Remapping between a source (``UxDataArray``) and
        destination.`.

        Parameters
        ---------
        destination_obj : Grid, UxDataArray, UxDataset
            Destination for remapping
        remap_to : str, default="nodes"
            Location of where to map data, either "nodes" or "face centers"
        coord_type : str, default="spherical"
            Indicates whether to remap using on spherical or cartesian coordinates
        """

        return _nearest_neighbor_uxda(self, destination_obj, remap_to, coord_type)

    def integrate(
        self, quadrature_rule: Optional[str] = "triangular", order: Optional[int] = 4
    ) -> UxDataArray:
        """Computes the integral of a data variable residing on an unstructured
        grid.

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
        >>> import uxarray as ux
        >>> uxds = ux.open_dataset("grid.ug", "centroid_pressure_data_ug")

        # Compute the integral
        >>> integral = uxds['psi'].integrate()
        """
        if self.values.shape[-1] == self.uxgrid.n_face:
            face_areas, face_jacobian = self.uxgrid.compute_face_areas(
                quadrature_rule, order
            )

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

    def nodal_average(self):
        """Computes the Nodal Average of a Data Variable, which is the mean of
        the nodes that surround each face.

        Can be used for remapping node-centered data to each face.
        """

        if not self._node_centered():
            # nodal average expects node-centered data
            raise ValueError(
                f"Data Variable must be mapped to the corner nodes of each face, with dimension "
                f"{self.uxgrid.n_face}."
            )

        data = self.values
        face_nodes = self.uxgrid.face_node_connectivity.values
        n_nodes_per_face = self.uxgrid.n_nodes_per_face.values

        # compute the nodal average while preserving original dimensions
        data_nodal_average = np.array(
            [
                np.mean(data[..., cur_face[0:n_max_nodes]], axis=-1)
                for cur_face, n_max_nodes in zip(face_nodes, n_nodes_per_face)
            ]
        )

        # set `n_nodes` as final dimension
        data_nodal_average = np.moveaxis(data_nodal_average, 0, -1)

        return UxDataArray(
            uxgrid=self.uxgrid,
            data=data_nodal_average,
            dims=self.dims,
            name=self.name + "_nodal_average" if self.name is not None else None,
        ).rename({"n_node": "n_face"})

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

    def isel(self, ignore_grid=False, *args, **kwargs):
        """Grid-informed implementation of xarray's ``isel`` method, which
        enables indexing across grid dimensions.

        Subsetting across grid dimensions ('n_node', 'n_edge', or 'n_face') returns will return a new UxDataArray with
        a newly initialized Grid only containing those elements.

        Currently only supports inclusive selection, meaning that for cases where node or edge indices are provided,
        any face that contains that element is included in the resulting subset. This means that additional elements
        beyond those that were initially provided in the indices will be included. Support for more methods, such as
        exclusive and clipped indexing is in the works.

        Parameters
        **kwargs: kwargs
            Dimension to index, one of ['n_node', 'n_edge', 'n_face'] for grid-indexing, or any other dimension for
            regular xarray indexing

        Example
        -------
        > uxda.subset(n_node=[1, 2, 3])
        """

        from uxarray.constants import GRID_DIMS

        if any(grid_dim in kwargs for grid_dim in GRID_DIMS) and not ignore_grid:
            # slicing a grid-dimension through Grid object

            dim_mask = [grid_dim in kwargs for grid_dim in GRID_DIMS]
            dim_count = np.count_nonzero(dim_mask)

            if dim_count > 1:
                raise ValueError("Only one grid dimension can be sliced at a time")

            if "n_node" in kwargs:
                sliced_grid = self.uxgrid.isel(n_node=kwargs["n_node"])
            elif "n_edge" in kwargs:
                sliced_grid = self.uxgrid.isel(n_edge=kwargs["n_edge"])
            else:
                sliced_grid = self.uxgrid.isel(n_face=kwargs["n_face"])

            return self._slice_from_grid(sliced_grid)

        else:
            # original xarray implementation for non-grid dimensions
            return super().isel(*args, **kwargs)

    def _slice_from_grid(self, sliced_grid):
        """Slices a  ``UxDataArray`` from a sliced ``Grid``, using cached
        indices to correctly slice the data variable."""

        from uxarray.core.dataarray import UxDataArray

        if self._face_centered():
            d_var = self.isel(
                n_face=sliced_grid._ds["subgrid_face_indices"], ignore_grid=True
            ).values

        elif self._edge_centered():
            d_var = self.isel(
                n_edge=sliced_grid._ds["subgrid_edge_indices"], ignore_grid=True
            ).values

        elif self._node_centered():
            d_var = (
                self.isel(
                    n_node=sliced_grid._ds["subgrid_node_indices"], ignore_grid=True
                ).values,
            )

        else:
            raise ValueError(
                "Data variable must be either node, edge, or face centered."
            )

        return UxDataArray(
            uxgrid=sliced_grid,
            data=d_var,
            name=self.name,
            dims=self.dims,
            attrs=self.attrs,
        )
