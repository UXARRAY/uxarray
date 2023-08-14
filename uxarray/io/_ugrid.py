import numpy as np

from uxarray.grid.connectivity import _replace_fill_values
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


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

    # standardize fill values and data type for face nodes
    xr_ds = _standardize_fill_values(xr_ds, var_names_dict)

    return xr_ds, var_names_dict


def _encode_ugrid(ds):
    """Encodes UGRID file .
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be encoded to file

    Uses to_netcdf from xarray object.
    """
    return ds


def _standardize_fill_values(ds, var_names_dict):
    """Standardizes the fill values and data type of index variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Input Dataset
    ds_vars_dict : dict
        UGRID Variable Name Dictionary

    Returns
    ----------
    ds : xarray.Dataset
        Input Dataset with correct index variables
    """

    # original face nodes
    face_nodes = ds[var_names_dict['Mesh2_face_nodes']].values

    # original fill value, if one exists
    if "_FillValue" in ds[var_names_dict['Mesh2_face_nodes']].attrs:
        original_fv = ds[var_names_dict['Mesh2_face_nodes']]._FillValue
    elif np.isnan(ds[var_names_dict['Mesh2_face_nodes']].values).any():
        original_fv = np.nan
    else:
        original_fv = None

    # if current dtype and fill value are not standardized
    if face_nodes.dtype != INT_DTYPE or original_fv != INT_FILL_VALUE:
        # replace fill values and set correct dtype
        new_face_nodes = _replace_fill_values(grid_var=face_nodes,
                                              original_fill=original_fv,
                                              new_fill=INT_FILL_VALUE,
                                              new_dtype=INT_DTYPE)
        # reassign data to use updated face nodes
        ds[var_names_dict['Mesh2_face_nodes']].data = new_face_nodes

        # use new fill value
        ds[var_names_dict['Mesh2_face_nodes']].attrs[
            '_FillValue'] = INT_FILL_VALUE

    return ds


def _is_ugrid(ds):
    """Check mesh topology and dimension."""
    standard_name = lambda v: v is not None
    # getkeys_filter_by_attribute(filepath, attr_name, attr_val)
    # return type KeysView
    node_coords_dv = ds.filter_by_attrs(node_coordinates=standard_name)
    face_conn_dv = ds.filter_by_attrs(face_node_connectivity=standard_name)
    topo_dim_dv = ds.filter_by_attrs(topology_dimension=standard_name)
    mesh_topo_dv = ds.filter_by_attrs(cf_role="mesh_topology")
    if len(mesh_topo_dv) != 0 and len(topo_dim_dv) != 0 and len(
            face_conn_dv) != 0 and len(node_coords_dv) != 0:
        return True
    else:
        return False
