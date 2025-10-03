import xarray as xr

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
    except Exception:
        # If it fails, use the "netcdf4" engine as backup
        # Extract engine from kwargs to prevent duplicate parameter error
        engine = kwargs.pop("engine", "netcdf4")
        return xr.open_dataset(filename_or_obj, engine=engine, chunks=chunks, **kwargs)


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

    ds = _open_dataset_with_fallback(grid_filename_or_obj, chunks=chunks)
    grid_spec, _, _ = _parse_grid_type(ds)

    source_dims_dict = _get_source_dims_dict(ds, grid_spec)

    # correctly chunk standardized ugrid dimension names
    for original_grid_dim, ugrid_grid_dim in source_dims_dict.items():
        if ugrid_grid_dim in chunks:
            chunks[original_grid_dim] = chunks[ugrid_grid_dim]

    return chunks


def _validate_indexers(indexers, indexers_kwargs, func_name, ignore_grid):
    from xarray.core.utils import either_dict_or_kwargs

    from uxarray.constants import GRID_DIMS

    # Used to filter out slices containing all Nones (causes subscription errors, i.e., var[0])
    _is_full_none_slice = (
        lambda v: isinstance(v, slice)
        and v.start is None
        and v.stop is None
        and v.step is None
    )

    indexers = either_dict_or_kwargs(indexers, indexers_kwargs, func_name)

    # Only count a grid dim if its indexer is NOT a no-op full slice
    grid_dims = {
        dim
        for dim in GRID_DIMS
        if dim in indexers and not _is_full_none_slice(indexers[dim])
    }

    if not ignore_grid and len(grid_dims) > 1:
        raise ValueError(
            f"Only one grid dimension can be sliced at a time; got {sorted(grid_dims)}."
        )

    return indexers, grid_dims
