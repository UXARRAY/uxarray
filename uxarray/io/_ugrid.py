import numpy as np

from uxarray.grid.connectivity import _replace_fill_values
from uxarray.grid.utils import _get_ugrid_var_vame_dict
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


def _read_ugrid(xr_ds):
    """UGRID file reader.

    Parameters: xarray.Dataset, required
    Returns: ugrid aware xarray.Dataset
    """

    var_names_dict = _get_ugrid_var_vame_dict()

    # TODO: Standardized UGRID Variable Names (Mesh2_node_x, etc)

    # TODO: obtain and change to Mesh2 construct, see Issue #27

    # get the data variable name that has attribute "cf_role" set to "mesh_topology"
    # this is the base xarray.DataArray name
    base_xarray_var = list(
        xr_ds.filter_by_attrs(cf_role="mesh_topology").keys())[0]
    # TODO: Dataset includes more than just coordinates and face nodes, handle

    xr_ds = xr_ds.rename({base_xarray_var: "Mesh2"})

    # map and rename coordinates
    coord_names = xr_ds["Mesh2"].node_coordinates.split()
    if len(coord_names) == 1:
        var_names_dict["Mesh2_node_x"] = coord_names[0]
        xr_ds = xr_ds.rename({coord_names[0]: "Mesh2_node_x"})
    elif len(coord_names) == 2:
        var_names_dict["Mesh2_node_x"] = coord_names[0]
        var_names_dict["Mesh2_node_y"] = coord_names[1]
        xr_ds = xr_ds.rename({
            coord_names[0]: "Mesh2_node_x",
            coord_names[1]: "Mesh2_node_y"
        })
    elif len(coord_names) == 3:
        var_names_dict["Mesh2_node_x"] = coord_names[0]
        var_names_dict["Mesh2_node_y"] = coord_names[1]
        var_names_dict["Mesh2_node_z"] = coord_names[
            2]  # TODO: remove Mesh2_node_z

    # map and rename dimensions
    coord_dim_name = xr_ds["Mesh2_node_x"].dims
    var_names_dict["nMesh2_node"] = coord_dim_name[0]
    xr_ds = xr_ds.rename({coord_dim_name[0]: "nMesh2_node"})

    face_node_names = xr_ds["Mesh2"].face_node_connectivity.split()

    face_node_name = face_node_names[0]
    var_names_dict["Mesh2_face_nodes"] = xr_ds[face_node_name].name
    xr_ds = xr_ds.rename({xr_ds[face_node_name].name: "Mesh2_face_nodes"})

    var_names_dict["nMesh2_face"] = xr_ds["Mesh2_face_nodes"].dims[0]
    var_names_dict["nMaxMesh2_face_nodes"] = xr_ds["Mesh2_face_nodes"].dims[1]

    xr_ds = xr_ds.rename({
        xr_ds["Mesh2_face_nodes"].dims[0]: "nMesh2_face",
        xr_ds["Mesh2_face_nodes"].dims[1]: "nMaxMesh2_face_nodes"
    })

    if len(coord_names) == 2:
        # set coordinates
        xr_ds = xr_ds.set_coords(["Mesh2_node_x", "Mesh2_node_y"])
    else:
        xr_ds = xr_ds.set_coords(
            [["Mesh2_node_x", "Mesh2_node_y",
              "Mesh2_node_z"]  # TODO: remove Mesh2_node_z
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
    face_nodes = ds["Mesh2_face_nodes"].values

    # original fill value, if one exists
    if "_FillValue" in ds["Mesh2_face_nodes"].attrs:
        original_fv = ds["Mesh2_face_nodes"]._FillValue
    elif np.isnan(ds["Mesh2_face_nodes"].values).any():
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
        ds["Mesh2_face_nodes"].data = new_face_nodes

        # use new fill value
        ds["Mesh2_face_nodes"].attrs['_FillValue'] = INT_FILL_VALUE

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
