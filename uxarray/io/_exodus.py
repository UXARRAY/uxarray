import xarray as xr
import numpy as np
from pathlib import PurePath
from datetime import datetime

from uxarray.grid.connectivity import _replace_fill_values
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.grid.coordinates import _get_lonlat_from_xyz, _get_xyz_from_lonlat


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

    # populate ds
    # self.__init_mesh2__()
    ds["Mesh2"] = xr.DataArray(
        attrs={
            "cf_role": "mesh_topology",
            "long_name": "Topology data of unstructured mesh",
            "topology_dimension": -1,
            "node_coordinates": "Mesh2_node_x Mesh2_node_y Mesh2_node_z",
            "node_dimension": "nMesh2_node",
            "face_node_connectivity": "Mesh2_face_nodes",
            "face_dimension": "nMesh2_face"
        })

    # find max face nodes
    max_face_nodes = 0
    for dim in ext_ds.dims:
        if "num_nod_per_el" in dim:
            if ext_ds.dims[dim] > max_face_nodes:
                max_face_nodes = ext_ds.dims[dim]

    # create an empty conn array for storing all blk face_nodes_data
    conn = np.empty((0, max_face_nodes))

    for key, value in ext_ds.variables.items():
        if key == "qa_records":
            # TODO: Use the data here for Mesh2 construct, if required.
            pass
        elif key == "coord":
            ds.Mesh2.attrs['topology_dimension'] = INT_DTYPE(
                ext_ds.dims['num_dim'])
            ds["Mesh2_node_cart_x"] = xr.DataArray(
                data=ext_ds.coord[0],
                dims=["nMesh2_node"],
                attrs={
                    "standard_name": "x",
                    "long_name": "cartesian x",
                    "units": "m",
                })
            ds["Mesh2_node_cart_y"] = xr.DataArray(
                data=ext_ds.coord[1],
                dims=["nMesh2_node"],
                attrs={
                    "standard_name": "y",
                    "long_name": "cartesian y",
                    "units": "m",
                })
            if ext_ds.dims['num_dim'] > 2:
                ds["Mesh2_node_cart_z"] = xr.DataArray(
                    data=ext_ds.coord[2],
                    dims=["nMesh2_node"],
                    attrs={
                        "standard_name": "z",
                        "long_name": "cartesian z",
                        "units": "m",
                    })
        elif key == "coordx":
            ds["Mesh2_node_cart_x"] = xr.DataArray(
                data=ext_ds.coordx,
                dims=["nMesh2_node"],
                attrs={
                    "standard_name": "x",
                    "long_name": "cartesian x",
                    "units": "m",
                })
        elif key == "coordy":
            ds["Mesh2_node_cart_y"] = xr.DataArray(
                data=ext_ds.coordx,
                dims=["nMesh2_node"],
                attrs={
                    "standard_name": "y",
                    "long_name": "cartesian y",
                    "units": "m",
                })
        elif key == "coordz":
            if ext_ds.dims['num_dim'] > 2:
                ds["Mesh2_node_cart_z"] = xr.DataArray(
                    data=ext_ds.coordx,
                    dims=["nMesh2_node"],
                    attrs={
                        "standard_name": "z",
                        "long_name": "cartesian z",
                        "units": "m",
                    })
        elif "connect" in key:
            # check if num face nodes is less than max.
            if value.data.shape[1] <= max_face_nodes:
                conn = np.full((value.data.shape[1], max_face_nodes),
                               0,
                               dtype=conn.dtype)
                conn = value.data
            else:
                raise RuntimeError(
                    "found face_nodes_dim greater than nMaxMesh2_face_nodes")

            # find the elem_type as etype for this element
            for k, v in value.attrs.items():
                if k == "elem_type":
                    # TODO: etype if not used now, remove if it'll never be required
                    etype = v

    # outside the k,v for loop
    # set the face nodes data compiled in "connect" section

    # standardize fill values and data type face nodes
    face_nodes = _replace_fill_values(grid_var=conn[:] - 1,
                                      original_fill=-1,
                                      new_fill=INT_FILL_VALUE,
                                      new_dtype=INT_DTYPE)

    ds["Mesh2_face_nodes"] = xr.DataArray(
        data=face_nodes,
        dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
        attrs={
            "cf_role":
                "face_node_connectivity",
            "_FillValue":
                INT_FILL_VALUE,
            "start_index":
                INT_DTYPE(
                    0)  # NOTE: This might cause an error if numbering has holes
        })

    # populate lon/lat coordinates
    lon, lat = _get_lonlat_from_xyz(ds["Mesh2_node_cart_x"].values,
                                    ds["Mesh2_node_cart_y"].values,
                                    ds["Mesh2_node_cart_z"].values)

    # populate dataset
    ds["Mesh2_node_x"] = xr.DataArray(
        data=lon,
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "longitude",
            "long_name": "longitude of mesh nodes",
            "units": "degrees_east",
        })
    ds["Mesh2_node_y"] = xr.DataArray(data=lat,
                                      dims=["nMesh2_node"],
                                      attrs={
                                          "standard_name": "latitude",
                                          "long_name": "latitude of mesh nodes",
                                          "units": "degrees_north",
                                      })

    # set lon/lat coordinates
    ds = ds.set_coords(["Mesh2_node_x", "Mesh2_node_y"])

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
        "file_size": 0
    }

    if outfile:
        path = PurePath(outfile)
        out_filename = path.name
        title = f"uxarray(" + str(out_filename) + ")" + date + ": " + time

        exo_ds.attrs = {
            "api_version": api_version,
            "version": exo_version,
            "floating_point_word_size": fp_word,
            "file_size": 0,
            "title": title
        }

    exo_ds["time_whole"] = xr.DataArray(data=[], dims=["time_step"])

    # qa_records
    # version identifier of the application code: https://gsjaardema.github.io/seacas-docs/exodusII-new.pdf page: 12
    ux_exodus_version = 1.0
    qa_records = [["uxarray"], [ux_exodus_version], [date], [time]]
    exo_ds["qa_records"] = xr.DataArray(data=xr.DataArray(
        np.array(qa_records, dtype='str')),
                                        dims=["four", "num_qa_rec"])

    # Set the dim to 3 as we will always have x/y/z for cartesian grid
    # Note: Don't get orig dimension from Mesh2 attribute topology dimension
    if "Mesh2_node_cart_x" not in ds:
        x, y, z = _get_xyz_from_lonlat(ds["Mesh2_node_x"].values,
                                       ds["Mesh2_node_y"].values)
        c_data = xr.DataArray([x, y, z])
    else:
        c_data = xr.DataArray([
            ds["Mesh2_node_cart_x"].data.tolist(),
            ds["Mesh2_node_cart_y"].data.tolist(),
            ds["Mesh2_node_cart_z"].data.tolist()
        ])

    exo_ds["coord"] = xr.DataArray(data=c_data, dims=["num_dim", "num_nodes"])

    # process face nodes, this array holds num faces at corresponding location
    # eg num_el_all_blks = [0, 0, 6, 12] signifies 6 TRI and 12 SHELL elements
    num_el_all_blks = np.zeros(ds["nMaxMesh2_face_nodes"].size, "i8")
    # this list stores connectivity without filling
    conn_nofill = []

    # store the number of faces in an array
    for row in ds["Mesh2_face_nodes"].astype(INT_DTYPE).data:

        # find out -1 in each row, this indicates lower than max face nodes
        arr = np.where(row == -1)
        # arr[0].size returns the location of first -1 in the conn list
        # if > 0, arr[0][0] is the num_nodes forming the face
        if arr[0].size > 0:
            # increment the number of faces at the corresponding location
            num_el_all_blks[arr[0][0] - 1] += 1
            # append without -1s eg. [1, 2, 3, -1] to [1, 2, 3]
            # convert to list (for sorting later)
            row = row[:(arr[0][0])].tolist()
            list_node = list(map(int, row))
            conn_nofill.append(list_node)
        elif arr[0].size == 0:
            # increment the number of faces for this "nMaxMesh2_face_nodes" face
            num_el_all_blks[ds["nMaxMesh2_face_nodes"].size - 1] += 1
            # get integer list nodes
            list_node = list(map(int, row.tolist()))
            conn_nofill.append(list_node)
        else:
            raise RuntimeError(
                "num nodes in conn array is greater than nMaxMesh2_face_nodes. Abort!"
            )
    # get number of blks found
    num_blks = np.count_nonzero(num_el_all_blks)

    # sort connectivity by size, lower dim faces first
    conn_nofill.sort(key=len)

    # get index of blocks found
    nonzero_el_index_blks = np.nonzero(num_el_all_blks)

    # break Mesh2_face_nodes into blks
    start = 0
    for blk in range(num_blks):
        blkID = blk + 1
        str_el_in_blk = "num_el_in_blk" + str(blkID)
        str_nod_per_el = "num_nod_per_el" + str(blkID)
        str_att_in_blk = "num_att_in_blk" + str(blkID)
        str_global_id = "global_id" + str(blkID)
        str_edge_type = "edge_type" + str(blkID)
        str_attrib = "attrib" + str(blkID)
        str_connect = "connect" + str(blkID)

        # get element type
        num_nodes = len(conn_nofill[start])
        element_type = _get_element_type(num_nodes)

        # get number of faces for this block
        num_faces = num_el_all_blks[nonzero_el_index_blks[0][blk]]
        # assign Data variables
        # convert list to np.array, sorted list guarantees we have the correct info
        conn_blk = conn_nofill[start:start + num_faces]
        conn_np = np.array([np.array(xi, dtype=INT_DTYPE) for xi in conn_blk])
        exo_ds[str_connect] = xr.DataArray(data=xr.DataArray((conn_np[:] + 1)),
                                           dims=[str_el_in_blk, str_nod_per_el],
                                           attrs={"elem_type": element_type})

        # edge type
        exo_ds[str_edge_type] = xr.DataArray(
            data=xr.DataArray(np.zeros((num_faces, num_nodes),
                                       dtype=INT_DTYPE)),
            dims=[str_el_in_blk, str_nod_per_el])

        # global id
        gid = np.arange(start + 1, start + num_faces + 1, 1)
        exo_ds[str_global_id] = xr.DataArray(data=(gid), dims=[str_el_in_blk])

        # attrib
        # TODO: fix num attr
        num_attr = 1
        exo_ds[str_attrib] = xr.DataArray(data=xr.DataArray(
            np.zeros((num_faces, num_attr), float)),
                                          dims=[str_el_in_blk, str_att_in_blk])

        start = num_faces

    # blk for loop ends

    # eb_prop1
    prop1_vals = np.arange(1, num_blks + 1, 1)
    exo_ds["eb_prop1"] = xr.DataArray(data=prop1_vals,
                                      dims=["num_el_blk"],
                                      attrs={"name": "ID"})
    # eb_status
    exo_ds["eb_status"] = xr.DataArray(data=xr.DataArray(
        np.ones([num_blks], dtype=INT_DTYPE)),
                                       dims=["num_el_blk"])

    # eb_names
    eb_names = np.empty(num_blks, dtype='str')
    exo_ds["eb_names"] = xr.DataArray(data=xr.DataArray(eb_names),
                                      dims=["num_el_blk"])
    cnames = ["x", "y", "z"]

    exo_ds["coor_names"] = xr.DataArray(data=xr.DataArray(
        np.array(cnames, dtype='str')),
                                        dims=["num_dim"])

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
