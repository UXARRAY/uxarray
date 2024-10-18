import numpy as np
import xarray as xr
from uxarray.grid.connectivity import _replace_fill_values
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


import uxarray.conventions.ugrid as ugrid


def _read_ugrid(ds):
    """Parses an unstructured grid dataset and encodes it in the UGRID
    conventions."""

    # Extract grid topology attributes
    grid_topology_name = list(ds.filter_by_attrs(cf_role="mesh_topology").keys())[0]
    ds = ds.rename({grid_topology_name: "grid_topology"})

    # get the names of node_lon and node_lat
    node_lon_name, node_lat_name = ds["grid_topology"].node_coordinates.split()
    coord_dict = {
        node_lon_name: ugrid.NODE_COORDINATES[0],
        node_lat_name: ugrid.NODE_COORDINATES[1],
    }

    if "edge_coordinates" in ds["grid_topology"].attrs:
        # get the names of edge_lon and edge_lat, if they exist
        edge_lon_name, edge_lat_name = ds["grid_topology"].edge_coordinates.split()
        coord_dict[edge_lon_name] = ugrid.EDGE_COORDINATES[0]
        coord_dict[edge_lat_name] = ugrid.EDGE_COORDINATES[1]

    if "face_coordinates" in ds["grid_topology"].attrs:
        # get the names of face_lon and face_lat, if they exist
        face_lon_name, face_lat_name = ds["grid_topology"].face_coordinates.split()
        coord_dict[face_lon_name] = ugrid.FACE_COORDINATES[0]
        coord_dict[face_lat_name] = ugrid.FACE_COORDINATES[1]

    ds = ds.rename(coord_dict)

    conn_dict = {}
    for conn_name in ugrid.CONNECTIVITY_NAMES:
        if conn_name in ds.grid_topology.attrs:
            orig_conn_name = ds.grid_topology.attrs[conn_name]
            conn_dict[orig_conn_name] = conn_name
        elif len(ds.filter_by_attrs(cf_role=conn_name).keys()):
            orig_conn_name = list(ds.filter_by_attrs(cf_role=conn_name).keys())[0]
            conn_dict[orig_conn_name] = conn_name

    ds = ds.rename(conn_dict)

    for conn_name in conn_dict.values():
        ds = _standardize_connectivity(ds, conn_name)

    dim_dict = {}

    # Rename Core Dims (node, edge, face)
    if "node_dimension" in ds["grid_topology"].attrs:
        dim_dict[ds["grid_topology"].node_dimension] = ugrid.NODE_DIM
    else:
        dim_dict[ds["node_lon"].dims[0]] = ugrid.NODE_DIM

    if "face_dimension" in ds["grid_topology"].attrs:
        dim_dict[ds["grid_topology"].face_dimension] = ugrid.FACE_DIM
    else:
        dim_dict[ds["face_node_connectivity"].dims[0]] = ugrid.FACE_DIM

    if "edge_dimension" in ds["grid_topology"].attrs:
        # edge dimension is not always provided
        dim_dict[ds["grid_topology"].edge_dimension] = ugrid.EDGE_DIM
    else:
        if "edge_lon" in ds:
            dim_dict[ds["edge_lon"].dims[0]] = ugrid.EDGE_DIM

    for conn_name in conn_dict.values():
        # Ensure grid dimension (i.e. 'n_face') is always the first dimension
        da = ds[conn_name]
        dims = da.dims

        for grid_dim in dim_dict.keys():
            if dims[1] == grid_dim:
                ds[conn_name] = da.T

    dim_dict[ds["face_node_connectivity"].dims[1]] = ugrid.N_MAX_FACE_NODES_DIM

    for dim in ds.dims:
        if ds.sizes[dim] == 2:
            dim_dict[dim] = "two"

    ds = ds.swap_dims(dim_dict)

    return ds, dim_dict


def _encode_ugrid(ds):
    """Encodes an unstructured grid represented under a ``Grid`` object as a
    ``xr.Dataset`` with an updated grid topology variable."""

    if "grid_topology" in ds:
        ds = ds.drop_vars(["grid_topology"])

    grid_topology = ugrid.BASE_GRID_TOPOLOGY_ATTRS

    if "n_edge" in ds.dims:
        grid_topology["edge_dimension"] = "n_edge"

    if "face_lon" in ds:
        grid_topology["face_coordinates"] = "face_lon face_lat"
    if "edge_lon" in ds:
        grid_topology["edge_coordinates"] = "edge_lon edge_lat"

    # TODO: Encode spherical (i.e. node_x) coordinates eventually (need to extend ugrid conventions)

    for conn_name in ugrid.CONNECTIVITY_NAMES:
        if conn_name in ds:
            grid_topology[conn_name] = conn_name

    grid_topology_da = xr.DataArray(data=-1, attrs=grid_topology)

    ds["grid_topology"] = grid_topology_da

    return ds


def _standardize_connectivity(ds, conn_name):
    """Standardizes the fill values and data type for a given connectivity
    variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Input Dataset

    Returns
    ----------
    ds : xarray.Dataset
        Input Dataset with correct index variables
    """

    # original connectivity
    conn = ds[conn_name].values

    # original fill value, if one exists
    if "_FillValue" in ds[conn_name].attrs:
        original_fv = ds[conn_name]._FillValue
    elif np.isnan(ds[conn_name].values).any():
        original_fv = np.nan
    else:
        original_fv = None

    # if current dtype and fill value are not standardized
    if conn.dtype != INT_DTYPE or original_fv != INT_FILL_VALUE:
        # replace fill values and set correct dtype
        new_conn = _replace_fill_values(
            grid_var=conn,
            original_fill=original_fv,
            new_fill=INT_FILL_VALUE,
            new_dtype=INT_DTYPE,
        )

        if "start_index" in ds[conn_name].attrs:
            new_conn[new_conn != INT_FILL_VALUE] -= INT_DTYPE(ds[conn_name].start_index)
        else:
            fill_value_indices = new_conn != INT_FILL_VALUE
            start_index = new_conn[fill_value_indices].min()
            new_conn[fill_value_indices] -= INT_DTYPE(start_index)

        # reassign data to use updated connectivity
        ds[conn_name].data = new_conn

        # use new fill value
        ds[conn_name].attrs["_FillValue"] = INT_FILL_VALUE

    return ds


def _is_ugrid(ds):
    """Check mesh topology and dimension."""
    # getkeys_filter_by_attribute(filepath, attr_name, attr_val)
    # return type KeysView
    node_coords_dv = ds.filter_by_attrs(node_coordinates=lambda v: v is not None)
    face_conn_dv = ds.filter_by_attrs(face_node_connectivity=lambda v: v is not None)
    topo_dim_dv = ds.filter_by_attrs(topology_dimension=lambda v: v is not None)
    mesh_topo_dv = ds.filter_by_attrs(cf_role="mesh_topology")
    if (
        len(mesh_topo_dv) != 0
        and len(topo_dim_dv) != 0
        and len(face_conn_dv) != 0
        and len(node_coords_dv) != 0
    ):
        return True
    else:
        return False


def _validate_minimum_ugrid(grid_ds):
    """Checks whether a given ``grid_ds`` meets the requirements for a minimum
    unstructured grid encoded in the UGRID conventions, containing a set of (x,
    y) latlon coordinates and face node connectivity."""
    return (
        ("node_lon" in grid_ds and "node_lat" in grid_ds)
        or ("node_x" in grid_ds and "node_y" in grid_ds and "node_z" in grid_ds)
        and "face_node_connectivity" in grid_ds
    )
