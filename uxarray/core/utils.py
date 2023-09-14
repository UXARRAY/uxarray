def _map_dims_to_ugrid(ds, _source_dims_dict):
    """Given a dataset containing variables residing on an unstructured grid, remaps the original dimension name
    to match the UGRID conventions (i.e. "nCell" : "nMesh2_Face")"""

    keys_to_drop = []
    # obtain all dimensions not present in the original dataset
    for key in _source_dims_dict.keys():
        if key not in ds.dims:
            keys_to_drop.append(key)

    # drop dimensions not present in the original dataset
    for key in keys_to_drop:
        _source_dims_dict.pop(key)

    # rename dimensions to follow the UGRID conventions
    ds = ds.rename_dims(_source_dims_dict)

    return ds
