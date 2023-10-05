import numpy as np

from uxarray.grid.connectivity import _replace_fill_values
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


def _read_ugrid(xr_ds):
    """UGRID file reader.

    Parameters: xarray.Dataset, required
    Returns: ugrid aware xarray.Dataset
    """

    source_dims_dict = {}
    # TODO: obtain and change to Mesh2 construct, see Issue #27

    # get the data variable name that has attribute "cf_role" set to "mesh_topology"
    # this is the base xarray.DataArray name
    base_xarray_var = list(
        xr_ds.filter_by_attrs(cf_role="mesh_topology").keys())[0]
    # TODO: Allow for parsing datasets with more than just coordinates and face nodes

    xr_ds = xr_ds.rename({base_xarray_var: "Mesh2"})

    # map and rename coordinates
    coord_names = xr_ds["Mesh2"].node_coordinates.split()
    if len(coord_names) == 1:
        xr_ds = xr_ds.rename({coord_names[0]: "Mesh2_node_x"})
    elif len(coord_names) == 2:
        xr_ds = xr_ds.rename({
            coord_names[0]: "Mesh2_node_x",
            coord_names[1]: "Mesh2_node_y"
        })
    # map and rename dimensions
    coord_dim_name = xr_ds["Mesh2_node_x"].dims

    xr_ds = xr_ds.rename({coord_dim_name[0]: "nMesh2_node"})

    face_node_names = xr_ds["Mesh2"].face_node_connectivity.split()

    face_node_name = face_node_names[0]
    xr_ds = xr_ds.rename({xr_ds[face_node_name].name: "Mesh2_face_nodes"})

    xr_ds = xr_ds.rename({
        xr_ds["Mesh2_face_nodes"].dims[0]: "nMesh2_face",
        xr_ds["Mesh2_face_nodes"].dims[1]: "nMaxMesh2_face_nodes"
    })

    xr_ds = xr_ds.set_coords(["Mesh2_node_x", "Mesh2_node_y"])

    # standardize fill values and data type for face nodes
    xr_ds = _standardize_fill_values(xr_ds)

    # populate source dimensions
    source_dims_dict[coord_dim_name[0]] = "nMesh2_node"
    source_dims_dict[xr_ds["Mesh2_face_nodes"].dims[0]] = "nMesh2_face"
    source_dims_dict[xr_ds["Mesh2_face_nodes"].dims[1]] = "nMaxMesh2_face_nodes"

    return xr_ds, source_dims_dict


def _encode_ugrid(ds):
    """Encodes UGRID file .
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be encoded to file

    Uses to_netcdf from xarray object.
    """
    return ds


def _standardize_fill_values(ds):
    """Standardizes the fill values and data type of index variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Input Dataset

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


def _validate_minimum_ugrid(grid_ds):
    """Checks whether a given ``grid_ds`` meets the requirements for a minimum
    unstructured grid encoded in the UGRID conventions, containing a set of (x,
    y) latlon coordinates and face node connectivity."""
    return ("Mesh2_node_x" in grid_ds and "Mesh2_node_y" in grid_ds) or (
        "Mesh2_node_cart_x" in grid_ds and "Mesh2_node_cart_y" in grid_ds and
        "Mesh2_node_cart_z" in grid_ds) and "Mesh2_face_nodes" in grid_ds
