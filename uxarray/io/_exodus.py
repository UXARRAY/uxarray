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

    # find max face nodes
    max_face_nodes = 0
    for dim in ext_ds.dims:
        if "num_nod_per_el" in dim:
            if ext_ds.sizes[dim] > max_face_nodes:
                max_face_nodes = ext_ds.sizes[dim]

    for key, value in ext_ds.variables.items():
        if key == "qa_records":
            # TODO: Use the data here for Mesh2 construct, if required.
            pass
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
                data=ext_ds.coordx, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Y_ATTRS
            )
        elif key == "coordz":
            if ext_ds.sizes["num_dim"] > 2:
                ds["node_z"] = xr.DataArray(
                    data=ext_ds.coordx, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Z_ATTRS
                )
        elif "connect" in key:
            # check if num face nodes is less than max.
            if value.data.shape[1] <= max_face_nodes:
                face_nodes = value
            else:
                raise RuntimeError("found face_nodes_dim greater than n_max_face_nodes")

    # outside the k,v for loop
    # set the face nodes data compiled in "connect" section

    # standardize fill values and data type face nodes
    face_nodes = _replace_fill_values(
        grid_var=face_nodes[:] - 1,
        original_fill=-1,
        new_fill=INT_FILL_VALUE,
        new_dtype=INT_DTYPE,
    )

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
