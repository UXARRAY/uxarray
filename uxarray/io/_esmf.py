import xarray as xr


from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


def _read_esmf(ext_ds):
    """TODO:"""

    out_ds = xr.Dataset()

    source_dims_dict = _to_ugrid(ext_ds, out_ds)

    return out_ds, source_dims_dict


def _to_ugrid(in_ds, out_ds):
    """TODO:"""
    # dimensions
    source_dims_dict = {
        "n_node": "nodeCount",
        "n_face": "elementCount",
        "n_max_face_nodes": "maxNodePElement",
    }

    if in_ds["nodeCoords"].units == "degrees":
        # Spherical Coordinates (in degrees)
        node_lon = in_ds["nodeCoords"].isel(coordDim=0)
        node_lat = in_ds["nodeCoords"].isel(coordDim=1)

        face_lon = in_ds["centerCoords"].isel(coordDim=0)
        face_lat = in_ds["centerCoords"].isel(coordDim=1)

        out_ds["node_lon"] = xr.DataArray(
            data=node_lon.values,
            dims=[
                "n_node",
            ],
        )

        out_ds["node_lat"] = xr.DataArray(
            data=node_lat.values,
            dims=[
                "n_node",
            ],
        )

        out_ds["face_lon"] = xr.DataArray(data=face_lon.values, dims=["n_face"])

        out_ds["face_lat"] = xr.DataArray(
            data=face_lat.values,
            dims=[
                "n_face",
            ],
        )

    else:
        # TODO: Cartesian support
        pass

    n_nodes_per_face = in_ds["numElementConn"].values

    if "start_index" in in_ds["elementConn"]:
        start_index = in_ds["elementConn"].start_index
    else:
        # assume start index is 1 (TODO)
        start_index = 1

    face_node_connectivity = in_ds["elementConn"].values.astype(INT_DTYPE)
    for i, max_nodes in enumerate(n_nodes_per_face):
        # convert to zero index and standardize fill values
        face_node_connectivity[i, 0:max_nodes] -= start_index
        face_node_connectivity[i, max_nodes:] = INT_FILL_VALUE

    out_ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_connectivity,
        dims=["n_face", "n_nodes_per_face"],
        attrs={"_FillValue": INT_FILL_VALUE},
    )

    return source_dims_dict
