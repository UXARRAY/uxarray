import numpy as np
import xarray as xr

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.conventions import ugrid
from uxarray.grid.connectivity import _replace_fill_values
from uxarray.grid.coordinates import _xyz_to_lonlat_deg


# Exodus Number is one-based.
def _read_exodus(ext_ds):
    """Exodus file reader.

    Parameters: xarray.Dataset, required
    Returns: ugrid aware xarray.Dataset
    """

    # TODO: UGRID Variable Mapping
    source_dims_dict = {}

    # Not loading specific variables.
    # as there is no way to know number of face types etc. without loading
    # connect1, connect2, connect3, etc..
    ds = xr.Dataset()

    # Collect all connectivity arrays
    connect_arrays = [
        var for name, var in ext_ds.variables.items() if "connect" in name
    ]

    if not connect_arrays:
        raise RuntimeError("No connectivity variables found in Exodus file.")

    # find max face nodes
    max_face_nodes = max(arr.shape[1] for arr in connect_arrays)

    padded_arrays = []
    for arr in connect_arrays:
        if arr.shape[1] < max_face_nodes:
            pad_width = max_face_nodes - arr.shape[1]
            padding = np.zeros((arr.shape[0], pad_width), dtype=arr.dtype)
            padded_arr = np.hstack([arr.values, padding])
            padded_arrays.append(padded_arr)
        else:
            padded_arrays.append(arr.values)

    # Concatenate all face node arrays
    face_nodes_data = np.vstack(padded_arrays)

    face_nodes = xr.DataArray(face_nodes_data)

    for key, value in ext_ds.variables.items():
        if key == "qa_records":
            # TODO: Use the data here for Mesh2 construct, if required.
            pass
        # Node Coordinates
        elif key == "coord":
            ds["node_x"] = xr.DataArray(
                data=ext_ds.coord[0], dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_X_ATTRS
            )
            ds["node_y"] = xr.DataArray(
                data=ext_ds.coord[1], dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Y_ATTRS
            )
            if ext_ds.sizes["num_dim"] > 2:
                ds["node_z"] = xr.DataArray(
                    data=ext_ds.coord[2],
                    dims=[ugrid.NODE_DIM],
                    attrs=ugrid.NODE_Z_ATTRS,
                )
        elif key == "coordx":
            ds["node_x"] = xr.DataArray(
                data=ext_ds.coordx, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_X_ATTRS
            )
        elif key == "coordy":
            ds["node_y"] = xr.DataArray(
                data=ext_ds.coordy, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Y_ATTRS
            )
        elif key == "coordz":
            if ext_ds.sizes["num_dim"] > 2:
                ds["node_z"] = xr.DataArray(
                    data=ext_ds.coordz, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Z_ATTRS
                )

    # outside the k,v for loop
    # set the face nodes data compiled in "connect" section

    # standardize fill values and data type face nodes
    face_nodes = _replace_fill_values(
        grid_var=face_nodes,
        original_fill=0,
        new_fill=INT_FILL_VALUE,
        new_dtype=INT_DTYPE,
    )
    face_nodes = xr.where(face_nodes != INT_FILL_VALUE, face_nodes - 1, face_nodes)

    ds["face_node_connectivity"] = xr.DataArray(
        data=face_nodes,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    # populate lon/lat coordinates
    lon, lat = _xyz_to_lonlat_deg(
        ds["node_x"].values, ds["node_y"].values, ds["node_z"].values
    )

    # populate dataset
    ds["node_lon"] = xr.DataArray(
        data=lon, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LON_ATTRS
    )
    ds["node_lat"] = xr.DataArray(
        data=lat, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LAT_ATTRS
    )

    # set lon/lat coordinates
    ds = ds.set_coords(["node_lon", "node_lat"])

    return ds, source_dims_dict


def _get_element_type(num_nodes):
    """Helper function to get exodus element type from number of nodes."""
    ELEMENT_TYPE_DICT = {
        2: "BEAM",
        3: "TRI",
        4: "SHELL4",
        5: "SHELL5",
        6: "TRI6",
        7: "TRI7",
        8: "SHELL8",
    }
    element_type = ELEMENT_TYPE_DICT[num_nodes]
    return element_type
