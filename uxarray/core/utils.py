import xarray as xr

from uxarray.io.utils import _get_source_dims_dict, _parse_grid_type


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

    ds = xr.open_dataset(grid_filename_or_obj, chunks=chunks)
    grid_spec, _, _ = _parse_grid_type(ds)

    source_dims_dict = _get_source_dims_dict(ds, grid_spec)

    # correctly chunk standardized ugrid dimension names
    for original_grid_dim, ugrid_grid_dim in source_dims_dict.items():
        if ugrid_grid_dim in chunks:
            chunks[original_grid_dim] = chunks[ugrid_grid_dim]

    return chunks
