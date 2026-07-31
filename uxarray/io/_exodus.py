from datetime import datetime
from pathlib import PurePath

import numpy as np
import xarray as xr

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.conventions import ugrid
from uxarray.grid.connectivity import _replace_fill_values
from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _xyz_to_lonlat_deg


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
                data=ext_ds.coordy, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Y_ATTRS
            )
        elif key == "coordz":
            if ext_ds.sizes["num_dim"] > 2:
                ds["node_z"] = xr.DataArray(
                    data=ext_ds.coordz, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Z_ATTRS
                )
        elif "connect" in key:
            # This variable will be populated in the next step
            pass

    # outside the k,v for loop
    # set the face nodes data compiled in "connect" section
    connect_list = []
    for key, value in ext_ds.variables.items():
        if "connect" in key:
            connect_list.append(value.data)

    padded_blocks = []
    for block in connect_list:
        num_nodes = block.shape[1]
        pad_width = max_face_nodes - num_nodes

        # Pad with 0, as Exodus uses 0 for non-existent nodes
        padded_block = np.pad(
            block, ((0, 0), (0, pad_width)), "constant", constant_values=0
        )
        padded_blocks.append(padded_block)

    # Prevent error on empty grids
    if not padded_blocks:
        face_nodes = np.empty((0, max_face_nodes), dtype=INT_DTYPE)
    else:
        face_nodes = np.vstack(padded_blocks)

    # standardize fill values and data type face nodes
    face_nodes = _replace_fill_values(
        grid_var=xr.DataArray(face_nodes - 1),  # Wrap numpy array in a DataArray
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


def _encode_exodus(ds, outfile=None):
    """Encodes an Exodus file.

    Parameters
    ----------

    ds : xarray.Dataset, required
        Dataset to be encoded to exodus file.

    outfile : string, required
       Name of output file to be added as metadata into the output
       dataset

    Returns
    -------
    exo_ds : xarray.Dataset
        Dataset encoded as exodus file.
    """
    # Note this is 1-based unlike native Mesh2 construct

    exo_ds = xr.Dataset()

    # --- Globals and Attributes ---
    now = datetime.now()
    date = now.strftime("%Y:%m:%d")
    time = now.strftime("%H:%M:%S")
    fp_word = INT_DTYPE(8)
    exo_version = np.float32(5.0)
    api_version = np.float32(5.0)

    exo_ds.attrs = {
        "api_version": api_version,
        "version": exo_version,
        "floating_point_word_size": fp_word,
        "file_size": 0,
    }

    if outfile:
        path = PurePath(outfile)
        out_filename = path.name
        title = f"uxarray({out_filename}){date}: {time}"
        exo_ds.attrs["title"] = title

    exo_ds["time_whole"] = xr.DataArray(data=[], dims=["time_step"])

    # --- Add num_elem dimension ---
    exo_ds.attrs["num_elem"] = ds.sizes["n_face"]

    # --- QA Records ---
    ux_exodus_version = "1.0"
    qa_records = [["uxarray"], [ux_exodus_version], [date], [time]]
    exo_ds["qa_records"] = xr.DataArray(
        data=np.array(qa_records, dtype="S"),
        dims=["num_qa_rec", "four"],
    ).transpose()

    # --- Node Coordinates ---
    if "node_x" not in ds:
        node_lon_rad = np.deg2rad(ds["node_lon"].values)
        node_lat_rad = np.deg2rad(ds["node_lat"].values)
        x, y, z = _lonlat_rad_to_xyz(node_lon_rad, node_lat_rad)
        c_data = np.array([x, y, z])
    else:
        c_data = np.array(
            [ds["node_x"].values, ds["node_y"].values, ds["node_z"].values]
        )

    exo_ds["coord"] = xr.DataArray(data=c_data, dims=["num_dim", "num_nodes"])

    # --- Element Blocks and Connectivity ---
    # Use .sizes to robustly get the dimension size
    n_max_nodes_per_face = ds.sizes["n_max_face_nodes"]
    num_el_all_blks = np.zeros(n_max_nodes_per_face, dtype=np.int64)
    conn_nofill = []

    for row in ds["face_node_connectivity"].values:
        # Find the index of the first fill value (-1)
        fill_val_idx = np.where(row == -1)[0]

        if fill_val_idx.size > 0:
            num_nodes = fill_val_idx[0]
            num_el_all_blks[num_nodes - 1] += 1
            conn_nofill.append(row[:num_nodes].astype(int).tolist())
        else:  # No fill value found, face uses all nodes
            num_nodes = n_max_nodes_per_face
            num_el_all_blks[num_nodes - 1] += 1
            conn_nofill.append(row.astype(int).tolist())

    num_blks = np.count_nonzero(num_el_all_blks)
    conn_nofill.sort(key=len)
    nonzero_el_index_blks = np.nonzero(num_el_all_blks)[0]

    start = 0
    for blk_idx, face_node_idx in enumerate(nonzero_el_index_blks):
        blkID = blk_idx + 1
        num_nodes_per_elem = face_node_idx + 1
        num_elem_in_blk = num_el_all_blks[face_node_idx]
        element_type = _get_element_type(num_nodes_per_elem)

        # Define variable names for this block
        conn_name = f"connect{blkID}"
        el_in_blk_dim = f"num_el_in_blk{blkID}"
        nod_per_el_dim = f"num_nod_per_el{blkID}"

        # Slice the connectivity list for the current block
        conn_blk = conn_nofill[start : start + num_elem_in_blk]
        conn_np = np.array(conn_blk, dtype=np.int64)

        # Add connectivity data for the block (1-based)
        exo_ds[conn_name] = xr.DataArray(
            data=conn_np + 1,
            dims=[el_in_blk_dim, nod_per_el_dim],
            attrs={"elem_type": element_type},
        )

        # Correctly increment the start index for the next block
        start += num_elem_in_blk

    # --- Element Block Properties ---
    prop1_vals = np.arange(1, num_blks + 1, 1, dtype=np.int32)
    exo_ds["eb_prop1"] = xr.DataArray(
        data=prop1_vals, dims=["num_el_blk"], attrs={"name": "ID"}
    )
    exo_ds["eb_status"] = xr.DataArray(
        data=np.ones(num_blks, dtype=np.int32), dims=["num_el_blk"]
    )
    exo_ds["eb_names"] = xr.DataArray(
        data=np.empty(num_blks, dtype="S"), dims=["num_el_blk"]
    )

    # --- Coordinate and Name Placeholders ---
    cnames = np.array(["x", "y", "z"], dtype="S")
    exo_ds["coor_names"] = xr.DataArray(data=cnames, dims=["num_dim"])

    return exo_ds


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
