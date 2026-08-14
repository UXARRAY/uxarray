from datetime import datetime

import numpy as np
import xarray as xr

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.conventions import ugrid
from uxarray.errors import GridInvalidError


def _esmf_to_ugrid_dims(in_ds):
    source_dims_dict = {
        "nodeCount": ugrid.NODE_DIM,
        "elementCount": ugrid.FACE_DIM,
        "maxNodePElement": ugrid.N_MAX_FACE_NODES_DIM,
    }
    return source_dims_dict


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

    source_dims_dict = _esmf_to_ugrid_dims(in_ds)

    if in_ds["nodeCoords"].units == "degrees":
        # Spherical Coordinates (in degrees)
        node_lon = in_ds["nodeCoords"].isel(coordDim=0)
        out_ds[ugrid.NODE_COORDINATES[0]] = xr.DataArray(
            node_lon, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LON_ATTRS
        )

        node_lat = in_ds["nodeCoords"].isel(coordDim=1)
        out_ds[ugrid.NODE_COORDINATES[1]] = xr.DataArray(
            node_lat, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LAT_ATTRS
        )

        if "centerCoords" in in_ds:
            # parse center coords (face centers) if available
            face_lon = in_ds["centerCoords"].isel(coordDim=0)
            out_ds[ugrid.FACE_COORDINATES[0]] = xr.DataArray(
                face_lon, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_LON_ATTRS
            )

            face_lat = in_ds["centerCoords"].isel(coordDim=1)
            out_ds[ugrid.FACE_COORDINATES[1]] = xr.DataArray(
                face_lat, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_LAT_ATTRS
            )

    else:
        raise NotImplementedError(
            "Reading in ESMF grids with Cartesian coordinates not yet supported"
        )

    n_nodes_per_face = in_ds["numElementConn"].astype(INT_DTYPE)
    out_ds["n_nodes_per_face"] = xr.DataArray(
        data=n_nodes_per_face,
        dims=ugrid.N_NODES_PER_FACE_DIMS,
        attrs=ugrid.N_NODES_PER_FACE_ATTRS,
    )

    if "start_index" in in_ds["elementConn"].attrs:
        start_index = in_ds["elementConn"].start_index
    else:
        # assume start index is 1 if one is not provided
        start_index = 1

    element_conn = in_ds["elementConn"]
    face_dim, node_dim = element_conn.dims

    # "numElementConn" is the format's authoritative face size, so the padding can
    # be located positionally. Matching the fill value instead means guessing at
    # it: CF decoding turns the ESMF sentinel into NaN, an undecoded read leaves
    # the raw value in place, and the integer cast below preserves neither form
    # (NaN casts to a platform-dependent value, not to INT_FILL_VALUE).
    positions = xr.DataArray(
        np.arange(element_conn.sizes[node_dim], dtype=INT_DTYPE), dims=node_dim
    )
    fill_mask = (positions >= n_nodes_per_face).transpose(face_dim, node_dim)

    # NaN is never a usable index, whatever "numElementConn" claims
    fill_mask = fill_mask | element_conn.isnull()

    face_node_connectivity = element_conn.fillna(0).astype(INT_DTYPE) - start_index
    face_node_connectivity = xr.where(fill_mask, INT_FILL_VALUE, face_node_connectivity)

    out_ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_connectivity,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    return out_ds, source_dims_dict


def _encode_esmf(ds: xr.Dataset) -> xr.Dataset:
    """Encodes a UGRID-compliant xarray.Dataset into ESMF Unstructured Grid
    Format.

    Parameters
    ----------
    ds : xr.Dataset
        A UGRID-compliant xarray.Dataset.

    Returns
    -------
    xr.Dataset
        An xarray.Dataset formatted according to ESMF Unstructured Grid conventions.
    """
    out_ds = xr.Dataset()

    # Node Coordinates (nodeCoords)
    if "node_lon" in ds and "node_lat" in ds:
        out_ds["nodeCoords"] = xr.concat(
            [ds["node_lon"], ds["node_lat"]],
            dim=xr.DataArray([0, 1], dims="coordDim"),
        )
        out_ds["nodeCoords"] = out_ds["nodeCoords"].rename({"n_node": "nodeCount"})
        out_ds["nodeCoords"] = out_ds["nodeCoords"].transpose("nodeCount", "coordDim")
        out_ds["nodeCoords"] = out_ds["nodeCoords"].assign_attrs(units="degrees")
        # Clean up unwanted attributes
        for attr in ["standard_name", "long_name"]:
            if attr in out_ds["nodeCoords"].attrs:
                del out_ds["nodeCoords"].attrs[attr]
    else:
        raise GridInvalidError("Input dataset must contain 'node_lon' and 'node_lat'.")

    # Face Node Connectivity (elementConn)
    if "face_node_connectivity" in ds:
        # ESMF elementConn is 1-based, with -1 for unused; UGRID is 0-based
        face_node_conn = ds["face_node_connectivity"]

        # Only offset the valid indices. Applying the offset to INT_FILL_VALUE and
        # letting it fall through to the int32 encoding below truncates it into a
        # small, valid node index, silently turning padding into real vertices.
        element_conn = xr.where(
            face_node_conn == INT_FILL_VALUE, -1, face_node_conn + 1
        )

        out_ds["elementConn"] = xr.DataArray(
            element_conn,
            dims=("elementCount", "maxNodePElement"),
            attrs={
                "long_name": "Node Indices that define the element connectivity",
                "_FillValue": -1,
            },
        )
        out_ds["elementConn"].encoding = {"dtype": np.int32}
    else:
        raise GridInvalidError("Input dataset must contain 'face_node_connectivity'.")

    # Number of nodes per face (numElementConn)
    if "n_nodes_per_face" in ds:
        out_ds["numElementConn"] = xr.DataArray(
            ds["n_nodes_per_face"],
            dims="elementCount",
            attrs={"long_name": "Number of nodes in each element"},
        )
        out_ds["numElementConn"].encoding = {"dtype": np.byte}
    else:
        # Fallback: derive from elementConn if not explicitly provided
        num_nodes = (out_ds["elementConn"] != -1).sum(dim="maxNodePElement")
        out_ds["numElementConn"] = xr.DataArray(
            num_nodes,
            dims="elementCount",
            attrs={"long_name": "Number of nodes in each element"},
        )
        out_ds["numElementConn"].encoding = {"dtype": np.byte}

    # Optional face coordinates (centerCoords)
    if "face_lon" in ds and "face_lat" in ds:
        out_ds["centerCoords"] = xr.concat(
            [ds["face_lon"], ds["face_lat"]],
            dim=xr.DataArray([0, 1], dims="coordDim"),
        )
        out_ds["centerCoords"] = out_ds["centerCoords"].rename(
            {"n_face": "elementCount"}
        )
        out_ds["centerCoords"] = out_ds["centerCoords"].transpose(
            "elementCount", "coordDim"
        )
        out_ds["centerCoords"] = out_ds["centerCoords"].assign_attrs(units="degrees")
        # Clean up unwanted attributes
        for attr in ["standard_name", "long_name"]:
            if attr in out_ds["centerCoords"].attrs:
                del out_ds["centerCoords"].attrs[attr]

    # Force no '_FillValue' in encoding if not explicitly set
    for v in out_ds.variables:
        if "_FillValue" not in out_ds[v].encoding:
            out_ds[v].encoding["_FillValue"] = None

    # Add global attributes
    out_ds.attrs = {
        "title": "ESMF Unstructured Grid from uxarray",
        "source": "Converted from UGRID conventions by uxarray",
        "date_created": datetime.now().isoformat(),
    }

    return out_ds
