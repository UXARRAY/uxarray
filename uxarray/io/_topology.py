import xarray as xr

import uxarray.conventions.ugrid as ugrid
from uxarray.grid.connectivity import _replace_fill_values
from uxarray.constants import INT_FILL_VALUE, INT_DTYPE


def _read_topology(
    node_lon, node_lat, face_node_connectivity, fill_value, start_index, **kwargs
):
    ds = xr.Dataset()

    for coord in ugrid.SPHERICAL_COORD_NAMES:
        # parse spherical coordinates
        if coord in ["node_lon", "node_lat"] or coord in kwargs:
            if coord == "node_lon":
                coord_arr = node_lon
            elif coord == "node_lat":
                coord_arr = node_lat
            else:
                coord_arr = kwargs[coord]

            ds[coord] = xr.DataArray(
                data=coord_arr,
                dims=ugrid.SPHERICAL_COORDS[coord]["dims"],
                attrs=ugrid.SPHERICAL_COORDS[coord]["attrs"],
            )

    for coord in ugrid.CARTESIAN_COORD_NAMES:
        # parse cartesian coords
        if coord in kwargs:
            coord_arr = kwargs[coord]

            ds[coord] = xr.DataArray(
                data=coord_arr,
                dims=ugrid.CARTESIAN_COORDS[coord]["dims"],
                attrs=ugrid.CARTESIAN_COORDS[coord]["attrs"],
            )

    for conn in ugrid.CONNECTIVITY_NAMES:
        # parse connectivity
        if conn == "face_node_connectivity" or conn in kwargs:
            if conn == "face_node_connectivity":
                conn_arr = face_node_connectivity
            else:
                conn_arr = kwargs[conn]

            ds[conn] = xr.DataArray(
                data=_process_connectivity(conn_arr, fill_value, start_index),
                dims=ugrid.CONNECTIVITY[conn]["dims"],
                attrs=ugrid.CONNECTIVITY[conn]["attrs"],
            )

    return ds


def _process_connectivity(conn, orig_fv, start_index):
    """Internal helper for processing connectivity variables, standardizing
    fill values and converting to zero-index."""

    if orig_fv is None:
        # no fill values, adjust start index
        return conn - start_index

    if orig_fv != INT_FILL_VALUE:
        conn = _replace_fill_values(conn, orig_fv, INT_FILL_VALUE, INT_DTYPE)

    conn[conn != INT_FILL_VALUE] -= start_index

    return conn
