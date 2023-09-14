def _map_dims_to_ugrid(ds, _source_dims_dict):

    keys_to_drop = []
    for key in _source_dims_dict.keys():
        if key not in ds.dims:
            keys_to_drop.append(key)

    for key in keys_to_drop:
        _source_dims_dict.pop(key)

    ds = ds.rename_dims(_source_dims_dict)

    return ds
