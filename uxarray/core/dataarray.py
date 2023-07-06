import xarray as xr

from uxarray.grid.grid import Grid


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
                "an instance of the uxarray.Grid class")
        else:
            self.uxgrid = uxgrid

        super().__init__(*args, **kwargs)

    @classmethod
    def _construct_direct(cls, *args, **kwargs):
        """Override to make the result a ``uxarray.UxDataArray`` class."""
        return cls(xr.DataArray._construct_direct(*args, **kwargs))

    def _copy(self, **kwargs):
        """Override to make the result a complete instance of
        ``uxarray.UxDataArray``."""
        copied = super()._copy(**kwargs)

        deep = kwargs.get('deep', None)

        if deep == True:
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

    def to_geodataframe(self, override_grid=False, cache_grid=True):
        """Constructs a ``spatialpandas.GeoDataFrame`` with a "geometry"
        column, containing a collection of Shapely Polygons or MultiPolygons
        representing the geometry of the unstructured grid, and a data column
        representing a 1D slice of data mapped to each Polygon.

        Parameters
        Returns
        -------
        gdf : spatialpandas.GeoDataFrame
            The output `GeoDataFrame` with a filled out "geometry" and 1D data column
        """

        # data is multidimensional, must be a 1D slice
        if self.data.ndim > 1:
            raise ValueError(
                f"Data Variable must be 1-dimensional, with shape {self.uxgrid.nMesh2_face} "
                f"for face data or {self.uxgrid.nMesh2_face} for vertex data.")

        # data mapped to faces
        if self.data.size == self.uxgrid.nMesh2_face:
            gdf = self.uxgrid.to_geodataframe(override=override_grid,
                                              cache=cache_grid)
            gdf[self.name] = self.data
            return gdf

        # data mapped to nodes
        elif self.data.size == self.uxgrid.nMesh2_node:
            gdf = self.uxgrid.to_geodataframe(override=override_grid,
                                              cache=cache_grid)
            gdf[self.name] = self.data
            # TODO: implement method for getting data to be mapped to faces (mean, other interpolation?)
            return gdf

        # data not mapped to faces or nodes
        else:
            raise ValueError(
                f"Data Variable with size {self.data.size} does not match the number of faces "
                f"({self.uxgrid.nMesh2_face} or nodes ({self.uxgrid.nMesh2_node}."
            )

    def to_polycollection(self, override_grid=False, cache_grid=True):
        """Constructs a ``matplotlib.collections.PolyCollection`` object with
        polygons representing the geometry of the unstructured grid, with
        polygons that cross the antimeridian split across the antimeridian.

        Parameters
        ----------
        override_grid : bool
            Flag to recompute the ``PolyCollection`` if one is already cached
        cache_grid : bool
            Flag to indicate if the computed ``PolyCollection`` should be cached

        Returns
        -------
        gdf : spatialpandas.GeoDataFrame
            The output `GeoDataFrame` with a filled out "geometry" collumn
        """

        # data is multidimensional, must be a 1D slice
        if self.data.ndim > 1:
            raise ValueError(
                f"Data Variable must be 1-dimensional, with shape {self.uxgrid.nMesh2_face} "
                f"for face data or {self.uxgrid.nMesh2_face} for vertex data.")

        # data mapped to faces
        if self.data.size == self.uxgrid.nMesh2_face:
            poly_collection = self.uxgrid.to_polycollection(
                override=override_grid, cache=cache_grid)
            poly_collection.set_array(self.data)
            return poly_collection

        # data mapped to nodes
        elif self.data.size == self.uxgrid.nMesh2_node:
            poly_collection = self.uxgrid.to_polycollection(
                override=override_grid, cache=cache_grid)
            # TODO: implement method for getting data to be mapped to faces (mean, other interpolation?)
            return poly_collection

        # data not mapped to faces or nodes
        else:
            raise ValueError(
                f"Data Variable with size {self.data.size} does not match the number of faces "
                f"({self.uxgrid.nMesh2_face} or nodes ({self.uxgrid.nMesh2_node}."
            )
