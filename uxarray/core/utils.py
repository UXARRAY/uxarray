import warnings

import numpy as np
import xarray as xr
from xarray.core.utils import either_dict_or_kwargs

from uxarray.constants import GRID_DIMS
from uxarray.errors import DataCenteringError, DimensionError, GridsMismatchError
from uxarray.io.utils import _get_source_dims_dict, _parse_grid_type


def _open_dataset_with_fallback(filename_or_obj, chunks=None, **kwargs):
    """Internal utility function to open datasets with fallback to netcdf4 engine.

    Attempts to use Xarray's default read engine first, which may be "h5netcdf"
    or "scipy" after v2025.09.0. If that fails (typically for h5-incompatible files),
    falls back to using the "netcdf4" engine.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported).
    chunks : int, dict, 'auto' or None, optional
        If chunks is provided, it is used to load the new dataset into dask
        arrays.
    **kwargs
        Additional keyword arguments passed to xr.open_dataset

    Returns
    -------
    xr.Dataset
        The opened dataset
    """
    try:
        # Try opening with xarray's default read engine
        return xr.open_dataset(filename_or_obj, chunks=chunks, **kwargs)
    except Exception as default_engine_error:
        # If it fails, use the "netcdf4" engine as backup
        # Extract engine from kwargs to prevent duplicate parameter error
        engine = kwargs.pop("engine", "netcdf4")
        try:
            return xr.open_dataset(
                filename_or_obj, engine=engine, chunks=chunks, **kwargs
            )
        except Exception as fallback_error:
            # Chain the fallback onto the original so both engines' reasons are
            # visible; otherwise the default engine's error is lost entirely.
            raise fallback_error from default_engine_error


def _map_dims_to_ugrid(
    ds,
    _source_dims_dict,
    grid,
):
    """Given a dataset containing variables residing on an unstructured grid,
    remaps the original dimension name to match the UGRID conventions (i.e.
    "nCell": "n_face")"""

    if grid.source_grid_spec == "Structured":
        # Case for structured grids, flatten bottom two sptial dimensions

        lon_name, lat_name = _source_dims_dict["n_face"]

        for var_name in ds.data_vars:
            if lon_name in ds[var_name].dims and lat_name in ds[var_name].dims:
                if ds[var_name].dims[-1] == lon_name:
                    dim_ordered = [lat_name, lon_name]
                else:
                    dim_ordered = [lon_name, lat_name]

        ds = ds.stack(n_face=dim_ordered)

    elif grid.source_grid_spec == "GEOS-CS":
        # stack dimensions to flatten them to map to nodes or faces
        for var_name in list(ds.coords) + list(ds.data_vars):
            if all(key in ds[var_name].sizes for key in ["nf", "Ydim", "Xdim"]):
                ds[var_name] = ds[var_name].stack(n_face=["nf", "Ydim", "Xdim"])
            if all(key in ds[var_name].sizes for key in ["nf", "YCdim", "XCdim"]):
                ds[var_name] = ds[var_name].stack(n_node=["nf", "YCdim", "XCdim"])
    else:
        keys_to_drop = []
        for key in _source_dims_dict.keys():
            # obtain all dimensions not present in the original dataset
            if key not in ds.dims:
                keys_to_drop.append(key)

        for key in keys_to_drop:
            # drop dimensions not present in the original dataset
            _source_dims_dict.pop(key)

        # build a reverse map
        size_to_name = {
            grid._ds.sizes[name]: name
            for name in ("n_face", "n_node", "n_edge")
            if name in grid._ds.dims
        }

        for dim in set(ds.dims) - _source_dims_dict.keys():
            name = size_to_name.get(ds.sizes[dim])
            if name:
                _source_dims_dict[dim] = name

        # rename dimensions to follow the UGRID conventions
        ds = ds.swap_dims(_source_dims_dict)

    return ds


def match_chunks_to_ugrid(grid_filename_or_obj, chunks):
    """Matches chunks to of the original dimensions to the UGRID conventions."""

    if not isinstance(chunks, dict):
        # No need to rename
        return chunks

    if isinstance(grid_filename_or_obj, xr.Dataset):
        ds = grid_filename_or_obj
    else:
        ds = _open_dataset_with_fallback(grid_filename_or_obj, chunks=chunks)

    grid_spec, _, _ = _parse_grid_type(ds)

    source_dims_dict = _get_source_dims_dict(ds, grid_spec)

    # correctly chunk standardized ugrid dimension names
    for original_grid_dim, ugrid_grid_dim in source_dims_dict.items():
        if ugrid_grid_dim in chunks:
            chunks[original_grid_dim] = chunks[ugrid_grid_dim]

    return chunks


def _validate_indexers(indexers, indexers_kwargs, func_name, ignore_grid):
    """returns (dict of indexers, set of grid_dim strs).

    Parameters
    ----------
    indexers: dict
        indexers originally provided as dict. E.g., uxarr.isel({'n_face': 0}).
        Provide indexers or indexers_kwargs but not both.
    indexers_kwargs: dict
        indexers originally provided as kwargs. E.g. uxarr.isel(n_face=0).
        Provide indexers or indexers_kwargs but not both.
    func_name: str
        name of the function calling _validate_indexers. E.g. "isel".
        Included in error message if provided both indexers and indexers_kwargs.
    ignore_grid: bool
        whether ignore_grid=True flag was set in the indexing operation.
        If False, ensure len(grid_dims) <= 1 else raise DimensionError.

    Returns
    -------
    indexers: dict
        validated dict of indexers, including grid dims indexers if present.
    grid_dims: set
        set of grid dimension names (from ``GRID_DIMS``) present as keys in indexers;
        values from {"n_face", "n_node", "n_edge"} (at most 1 value if ignore_grid=False).
    """

    # Used to filter out slices containing all Nones (causes subscription errors, i.e., var[0])
    _is_full_none_slice = lambda v: (
        isinstance(v, slice) and v.start is None and v.stop is None and v.step is None
    )

    indexers = either_dict_or_kwargs(indexers, indexers_kwargs, func_name)

    # Only count a grid dim if its indexer is NOT a no-op full slice
    grid_dims = {
        dim
        for dim in GRID_DIMS
        if dim in indexers and not _is_full_none_slice(indexers[dim])
    }

    if not ignore_grid and len(grid_dims) > 1:
        raise DimensionError(
            f"Only one grid dimension can be sliced at a time; got {sorted(grid_dims)}."
        )

    return indexers, grid_dims


def _resolve_coordinate_labels_to_indices(
    dim, labels_to_sel, coord_array, *, method=None, tolerance=None
):
    """returns indices which would be selected by coord_array.sel({dim: labels_to_sel}, ...)
    coord_array.isel({dim: result}) should be equivalent to coord_array.sel({dim: labels_to_sel}, ...).
    If labels_to_sel is an xr.DataArray, its coords/dims will also be attached to the result.

    dim: str
        dimension name to select along
    labels_to_sel: any valid indexer which can be passed to .sel()
        values to select along dim
    coord_array: xr.DataArray or UxDataArray
        coordinate array to select from.
    method, tolerance: passed directly to .sel().
    """
    # just using xarray's .sel() on a simple np.arange(), to ensure exactly consistent behavior with sel().
    # (Maybe a more efficient implementation exists, but this is simple and gives correct results.)
    indices = xr.DataArray(np.arange(coord_array.sizes[dim]), dims=dim)
    _indices_coord_name = f"__{dim}_indices__"  # just needs to be any unused name.
    if _indices_coord_name in coord_array.coords:
        warnings.warn(
            f"Coordinate {_indices_coord_name!r} already exists in coord_array.coords "
            "and will be overwritten, which may cause errors or subtly incorrect results..."
        )
    if hasattr(coord_array, "to_xarray"):  # convert to xarray to avoid recursive sel()
        coord_array = coord_array.to_xarray()
    coord_with_indices = coord_array.assign_coords({_indices_coord_name: indices})
    selected = coord_with_indices.sel(
        {dim: labels_to_sel}, method=method, tolerance=tolerance
    )
    result = selected[_indices_coord_name]
    if isinstance(labels_to_sel, xr.DataArray):
        # handle coords appropriately
        result = result.drop_vars((dim, _indices_coord_name))
        # (drop grid dim coords because the caller is expected to handle those directly;
        # here the goal is just to properly propagate any other coords from labels_to_sel.)
        result = result.rename(None)  # no reason to keep the _indices_coord_name
        # (and keeping it for longer could maybe cause confusing error later?)
    else:
        # drop all coords/name info which was added internally during this method.
        result = result.values
    return result


def _apply_1dfunc_with_grid_core_dim(
    f,
    grid_dim,
    *uxarrays,
    other_args=None,
    kwargs=None,
    n_outputs=1,
    vectorize=True,
    **kw_apply_ufunc,
):
    """Return the results of applying f across uxarrays, like apply_ufunc
    but treating the grid_dim as the core dimension.

    Parameters
    ----------
    f : callable
        function which takes as inputs one or more 1D numpy arrays
        with the grid_dim as the only dimension, and returns one or more
        1D numpy arrays of the same size.
    grid_dim : str
        name of the grid dimension to treat as the core dimension.
    uxarrays : one or more UxDataArray objects
        input uxarray objects to apply f to.
        May have multiple dimensions (must be compatible under broadcasting),
        but must all have the same grid_dim with compatible sizes and coords.
    other_args : iterable or None
        additional positional arguments to pass to f after the uxarrays,
        but which should not be considered for broadcasting as args to a ufunc.
    kwargs : dict or None
        additional keyword arguments to pass to f.
    n_outputs : int
        number of outputs from f, and from this function.
        If 1, this function returns a single UxDataArray; if >1, returns a tuple of UxDataArrays.
    vectorize : bool
        must be True. Included here to emphasize this is like using
        apply_ufunc(vectorize=True), because f is expected to operate on 1D arrays.
        If f is already vectorized to handle more dimensions, use xr.apply_ufunc
        or some other solution; using vectorize=True is slow, in general.
    additional kwargs are passed to xr.apply_ufunc.
    """
    # misc. checks
    if len(uxarrays) == 0:
        raise ValueError("Expected at least one object, got len(uxarrays)==0.")
    if not isinstance(grid_dim, str):
        raise TypeError(f"Expected grid_dim to be a str, got {type(grid_dim)}.")
    if not all(grid_dim in arr.dims for arr in uxarrays):
        raise DataCenteringError(
            f"Expected all input uxarray objects to have grid_dim dimension {grid_dim!r}, "
            f"but got uxarray objects' dims: {[arr.dims for arr in uxarrays]}."
        )
    for i, obj in enumerate(uxarrays):
        if obj.uxgrid != uxarrays[0].uxgrid:
            raise GridsMismatchError(
                f"Expected all input uxarray objects to have the same uxgrid, "
                f"but uxarrays[0].uxgrid != uxarrays[{i}].uxgrid."
            )
    if not vectorize:
        raise ValueError(
            "Expected vectorize=True, because f is expected to operate on 1D arrays."
        )
    # misc. bookkeeping
    if other_args is None:
        other_args = ()
    if kwargs is None:
        kwargs = {}
    if len(other_args) == 0 and len(kwargs) == 0:
        f_wrapped = f
    else:

        def f_wrapped(*numpy_arrays):
            return f(*numpy_arrays, *other_args, **kwargs)

    input_core_dims = [[grid_dim] for _ in uxarrays]
    output_core_dims = [[grid_dim] for _ in range(n_outputs)]
    result_cls = type(uxarrays[0])
    result_grid = uxarrays[0].uxgrid
    result_dims = uxarrays[0].dims
    xarrays = [arr.to_xarray() for arr in uxarrays]
    # actually do the calculations:
    result = xr.apply_ufunc(
        f_wrapped,
        *xarrays,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        vectorize=True,
        **kw_apply_ufunc,
    )
    # convert back to uxarray objects, and (style) restore original dims order.
    if n_outputs == 1:
        result = (result,)  # briefly write as tuple for consistency below
    result = tuple(
        res.transpose(*result_dims, ..., missing_dims="ignore") for res in result
    )
    result = tuple(result_cls(res, uxgrid=result_grid) for res in result)
    if n_outputs == 1:
        result = result[0]  # unpack single result from tuple
    return result
