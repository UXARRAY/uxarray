import xarray as xr


def _read_ugrid(xr_ds, var_names_dict):
    """UGRID file reader.

    Parameters: xarray.Dataset, required
    Returns: ugrid aware xarray.Dataset
    """

    # TODO: obtain and change to Mesh2 construct, see Issue #27

    # get the data variable name that has attribute "cf_role" set to "mesh_topology"
    # this is the base xarray.DataArray name
    base_xarray_var = list(
        xr_ds.filter_by_attrs(cf_role="mesh_topology").keys())[0]

    var_names_dict["Mesh2"] = base_xarray_var

    coord_names = xr_ds[base_xarray_var].node_coordinates.split()

    if len(coord_names) == 1:
        var_names_dict["Mesh2_node_x"] = coord_names[0]
    elif len(coord_names) == 2:
        var_names_dict["Mesh2_node_x"] = coord_names[0]
        var_names_dict["Mesh2_node_y"] = coord_names[1]
    elif len(coord_names) == 3:
        var_names_dict["Mesh2_node_x"] = coord_names[0]
        var_names_dict["Mesh2_node_y"] = coord_names[1]
        var_names_dict["Mesh2_node_z"] = coord_names[2]

    # set #nodes use x coordinates, y or z will be the same and can also be used
    coord_dim_name = xr_ds[var_names_dict["Mesh2_node_x"]].dims
    var_names_dict["nMesh2_node"] = coord_dim_name[0]

    face_node_names = xr_ds[base_xarray_var].face_node_connectivity.split()

    face_node_name = face_node_names[0]
    var_names_dict["Mesh2_face_nodes"] = xr_ds[face_node_name].name
    var_names_dict["nMesh2_face"] = xr_ds[face_node_name].dims[0]
    var_names_dict["nMaxMesh2_face_nodes"] = xr_ds[face_node_name].dims[1]

    if len(coord_names) == 2:
        # set coordinates
        xr_ds = xr_ds.set_coords(
            [var_names_dict["Mesh2_node_x"], var_names_dict["Mesh2_node_y"]])
    else:
        xr_ds = xr_ds.set_coords([
            var_names_dict["Mesh2_node_x"], var_names_dict["Mesh2_node_y"],
            var_names_dict["Mesh2_node_z"]
        ])
    return xr_ds, var_names_dict


def _encode_ugrid(ds):
    """Encodes UGRID file.
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be encoded to file

    Uses to_netcdf from xarray object.
    """
    return ds
