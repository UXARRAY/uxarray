import xarray as xr
from typing import Optional, List, Dict, Union, Sequence


def _map_dims_to_ugrid(
    ds,
    _source_dims_dict,
    grid,
):
    """Given a dataset containing variables residing on an unstructured grid,
    remaps the original dimension name to match the UGRID conventions (i.e.
    "nCell": "n_face")"""

    if grid.source_grid_spec == "GEOS-CS":
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

        for dim in set(ds.dims) ^ _source_dims_dict.keys():
            # obtain dimensions that were not parsed source_dims_dict and attempt to match to a grid element
            if ds.sizes[dim] == grid.n_face:
                _source_dims_dict[dim] = "n_face"
            elif ds.sizes[dim] == grid.n_node:
                _source_dims_dict[dim] = "n_node"
            elif ds.sizes[dim] == grid.n_edge:
                _source_dims_dict[dim] = "n_edge"

        # Possible Issue: https://github.com/UXARRAY/uxarray/issues/610

        # rename dimensions to follow the UGRID conventions
        ds = ds.swap_dims(_source_dims_dict)

    return ds


def _preserve_coordinates(
    coords: xr.core.coordinates.Coordinates,
    dims_to_drop: Optional[List[str]] = None,
    dims_to_isel: Optional[Dict[str, Union[int, slice, Sequence[int]]]] = None,
) -> xr.core.coordinates.Coordinates:
    """
    Preserve coordinates that do not depend on any of the dimensions to drop.
    Optionally, slice preserved coordinates based on provided dimension indices.

    Parameters:
    -----------
    coords : xr.core.coordinates.Coordinates
        The coordinates from an xarray Dataset or DataArray.
    dims_to_drop : list of str, optional
        The list of dimension names to drop. If None, no dimensions are dropped.
    dims_to_isel : dict, optional
        Dictionary mapping dimension names to indices, slices, or sequences for slicing.
        Example: {'time': 0, 'location': slice(None, 10), 'depth': [0, 2, 4]}

    Returns:
    --------
    xr.core.coordinates.Coordinates
        A new Coordinates object with specified coordinates preserved and sliced.

    Raises:
    -------
    TypeError:
        If `dims_to_drop` is not a list or `dims_to_isel` is not a dictionary.
    ValueError:
        If `dims_to_isel` contains dimensions not present in a coordinate.
    """
    # Validate `dims_to_drop`
    if dims_to_drop is not None and not isinstance(dims_to_drop, list):
        raise TypeError("`dims_to_drop` must be a list of dimension names or None.")

    # If dims_to_drop is None, treat it as an empty list (no dimensions to drop)
    if dims_to_drop is None:
        dims_to_drop = []

    # Validate `dims_to_isel`
    if dims_to_isel is not None and not isinstance(dims_to_isel, dict):
        raise TypeError(
            "`dims_to_isel` must be a dictionary mapping dimension names to indices or slices."
        )

    # If dims_to_isel is None, treat it as an empty dict (no slicing to perform)
    if dims_to_isel is None:
        dims_to_isel = {}

    # Initialize an empty dictionary to store preserved coordinates
    preserved_coords = {}

    # Iterate over each coordinate
    for coord_name, coord in coords.items():
        # Check if any of the coordinate's dimensions are in dims_to_drop
        if not any(dim in dims_to_drop for dim in coord.dims):
            # Prepare the indexing dictionary for slicing
            indexer = {
                dim: dims_to_isel[dim] for dim in coord.dims if dim in dims_to_isel
            }

            # Validate that dims_to_isel keys are valid dimensions for this coordinate
            invalid_dims = [dim for dim in indexer.keys() if dim not in coord.dims]
            if invalid_dims:
                raise ValueError(
                    f"Dimensions {invalid_dims} in `dims_to_isel` are not present in the coordinate '{coord_name}'."
                )

            if indexer:
                try:
                    # Use the underlying Variable's isel to avoid triggering `_preserve_coordinates` again
                    sliced_var = coord.variable.isel(**indexer)
                    # Wrap the sliced Variable back into a DataArray without coords to prevent recursion
                    sliced_coord = xr.DataArray(
                        sliced_var,
                        dims=sliced_var.dims,
                        attrs=coord.attrs,
                        name=coord_name,  # Preserve the original coordinate name
                    )
                    preserved_coords[coord_name] = sliced_coord
                except IndexError as e:
                    raise IndexError(
                        f"Indexing error for coordinate '{coord_name}': {e}"
                    ) from e
            else:
                # No slicing needed; preserve the coordinate as is
                preserved_coords[coord_name] = coord

    temp_ds = xr.Dataset(coords=preserved_coords)
    return temp_ds.coords
