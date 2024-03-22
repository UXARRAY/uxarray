import xarray as xr


from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

from uxarray.conventions import ugrid


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
        "nodeCount": ugrid.NODE_DIM,
        "elementCount": ugrid.FACE_DIM,
        "maxNodePElement": ugrid.N_MAX_FACE_NODES_DIM,
    }

    if in_ds["nodeCoords"].units == "degrees":
        # Spherical Coordinates (in degrees)
        node_lon = in_ds["nodeCoords"].isel(coordDim=0).values
        out_ds[ugrid.NODE_COORDINATES[0]] = xr.DataArray(
            node_lon, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LON_ATTRS
        )

        node_lat = in_ds["nodeCoords"].isel(coordDim=1).values
        out_ds[ugrid.NODE_COORDINATES[1]] = xr.DataArray(
            node_lat, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LAT_ATTRS
        )

        if "centerCoords" in in_ds:
            # parse center coords (face centers) if available
            face_lon = in_ds["centerCoords"].isel(coordDim=0).values
            out_ds[ugrid.FACE_COORDINATES[0]] = xr.DataArray(
                face_lon, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_LON_ATTRS
            )

            face_lat = in_ds["centerCoords"].isel(coordDim=1).values
            out_ds[ugrid.FACE_COORDINATES[1]] = xr.DataArray(
                face_lat, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_LAT_ATTRS
            )

    else:
        raise ValueError(
            "Reading in ESMF grids with Cartesian coordinates not yet supported"
        )

    n_nodes_per_face = in_ds["numElementConn"].values.astype(INT_DTYPE)
    out_ds["n_nodes_per_face"] = xr.DataArray(
        data=n_nodes_per_face,
        dims=ugrid.N_NODES_PER_FACE_DIMS,
        attrs=ugrid.N_NODES_PER_FACE_ATTRS,
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
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    return out_ds, source_dims_dict
