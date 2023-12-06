def _map_dims_to_ugrid(
    ds,
    _source_dims_dict,
    grid,
):
    """Given a dataset containing variables residing on an unstructured grid,
    remaps the original dimension name to match the UGRID conventions (i.e.
    "nCell": "n_face")"""

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
        if ds.dims[dim] == grid.n_face:
            _source_dims_dict[dim] = 'n_face'
        elif ds.dims[dim] == grid.n_node:
            _source_dims_dict[dim] = 'n_node'
        elif ds.dims[dim] == grid.n_edge:
            _source_dims_dict[dim] = 'n_edge'

    # Possible Issue: https://github.com/UXARRAY/uxarray/issues/610

    # rename dimensions to follow the UGRID conventions
    ds = ds.rename_dims(_source_dims_dict)

    return ds
