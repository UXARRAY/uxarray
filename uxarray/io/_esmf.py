import xarray as xr


from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


def _read_esmf(in_ds):
    """Reads in an Xarray dataset containing an ESMF formatted Grid dataset and
    encodes it in the UGRID conventions.

    Adheres to the ESMF Unstructrued Grid Format (ESMFMESH) outlined in the ESMF documentation:
    https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/node3.html#SECTION03028200000000000000

    Conversion
    ----------
    Coordinates stored in "nodeCoords" and "centerCoords" are split and stored in (``node_lon``, ``node_lat``) and
    (``face_lon``, ``face_lat``) respectively.

    Node connectivity stored in "elementConn" is converted to zero-index, with fill values standardized to
    INT_FILL_VALUE and stored in ``face_node_connectivity``.

    The Number of nodes per element ``numElementConn`` is stored in ``n_nodes_per_face``.

    Parameters
    ----------
    in_ds: xr.Dataset
        ESMF Grid Dataset

    Returns
    -------
    out_ds: xr.Dataset
        ESMF Grid encoder in the UGRID conventions
    source_dims_dict: dict
        Mapping of ESMF dimensions to UGRID dimensions
    """

    out_ds = xr.Dataset()

    source_dims_dict = {
        "nodeCount": "n_node",
        "elementCount": "n_face",
        "maxNodePElement": "n_max_face_nodes",
    }

    if in_ds["nodeCoords"].units == "degrees":
        # Spherical Coordinates (in degrees)
        node_lon = in_ds["nodeCoords"].isel(coordDim=0).values
        node_lat = in_ds["nodeCoords"].isel(coordDim=1).values

        out_ds["node_lon"] = xr.DataArray(
            node_lon,
            dims=["n_node"],
            attrs={
                "standard_name": "longitude",
                "long_name": "longitude of mesh nodes",
                "units": "degrees_east",
            },
        )

        out_ds["node_lat"] = xr.DataArray(
            node_lat,
            dims=["n_node"],
            attrs={
                "standard_name": "latitude",
                "long_name": "latitude of mesh nodes",
                "units": "degrees_north",
            },
        )

        if "centerCoords" in in_ds:
            # parse center coords (face centers) if avaliable

            face_lon = in_ds["centerCoords"].isel(coordDim=0).values
            face_lat = in_ds["centerCoords"].isel(coordDim=1).values

            out_ds["face_lon"] = xr.DataArray(
                face_lon,
                dims=["n_face"],
                attrs={
                    "standard_name": "longitude",
                    "long_name": "longitude of center nodes",
                    "units": "degrees_east",
                },
            )

            out_ds["face_lat"] = xr.DataArray(
                face_lat,
                dims=["n_face"],
                attrs={
                    "standard_name": "latitude",
                    "long_name": "latitude of center nodes",
                    "units": "degrees_north",
                },
            )

    else:
        raise ValueError(
            "Reading in ESMF grids with Cartesian coordinates not yet supported"
        )

    n_nodes_per_face = in_ds["numElementConn"].values.astype(INT_DTYPE)

    out_ds["n_nodes_per_face"] = xr.DataArray(
        data=n_nodes_per_face,
        dims=["n_face"],
        attrs={"long_name": "number of non-fill value nodes for each face"},
    )

    if "start_index" in in_ds["elementConn"]:
        start_index = in_ds["elementConn"].start_index
    else:
        # assume start index is 1 if one is not provided
        start_index = 1

    face_node_connectivity = in_ds["elementConn"].values.astype(INT_DTYPE)

    for i, max_nodes in enumerate(n_nodes_per_face):
        # convert to zero index and standardize fill values
        face_node_connectivity[i, 0:max_nodes] -= start_index
        face_node_connectivity[i, max_nodes:] = INT_FILL_VALUE

    out_ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_connectivity,
        dims=["n_face", "n_max_face_nodes"],
        attrs={
            "cf_role": "face_node_connectivity",
            "_FillValue": INT_FILL_VALUE,
            "start_index": INT_DTYPE(0),
        },
    )

    return out_ds, source_dims_dict
