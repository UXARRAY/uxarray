from uxarray.conventions import ugrid
import xarray as xr
import numpy as np


def _primal_to_ugrid(in_ds, out_ds):
    """Encodes the Primal Mesh of an ICON Grid into the UGRID conventions."""
    source_dims_dict = {"vertex": "n_node", "edge": "n_edge", "cell": "n_face"}

    # rename dimensions to match ugrid conventions
    in_ds = in_ds.rename_dims(source_dims_dict)

    # node coordinates
    node_lon = np.rad2deg(in_ds["vlon"])
    node_lat = np.rad2deg(in_ds["vlat"])

    out_ds["node_lon"] = xr.DataArray(
        data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )
    out_ds["node_lat"] = xr.DataArray(
        data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    # edge coordinates
    edge_lon = np.rad2deg(in_ds["elon"])
    edge_lat = np.rad2deg(in_ds["elat"])

    out_ds["edge_lon"] = xr.DataArray(
        data=edge_lon, dims=ugrid.EDGE_DIM, attrs=ugrid.EDGE_LON_ATTRS
    )
    out_ds["edge_lat"] = xr.DataArray(
        data=edge_lat, dims=ugrid.EDGE_DIM, attrs=ugrid.EDGE_LAT_ATTRS
    )

    # face coordinates
    face_lon = np.rad2deg(in_ds["clon"])
    face_lat = np.rad2deg(in_ds["clat"])

    out_ds["face_lon"] = xr.DataArray(
        data=face_lon, dims=ugrid.FACE_DIM, attrs=ugrid.FACE_LON_ATTRS
    )
    out_ds["face_lat"] = xr.DataArray(
        data=face_lat, dims=ugrid.FACE_DIM, attrs=ugrid.FACE_LAT_ATTRS
    )

    face_node_connectivity = in_ds["vertex_of_cell"].T - 1

    out_ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_connectivity,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    face_edge_connectivity = in_ds["edge_of_cell"].T - 1

    out_ds["face_edge_connectivity"] = xr.DataArray(
        data=face_edge_connectivity,
        dims=ugrid.FACE_EDGE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_EDGE_CONNECTIVITY_ATTRS,
    )

    face_face_connectivity = in_ds["neighbor_cell_index"].T - 1

    out_ds["face_face_connectivity"] = xr.DataArray(
        data=face_face_connectivity,
        dims=ugrid.FACE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_FACE_CONNECTIVITY_ATTRS,
    )

    edge_face_connectivity = in_ds["adjacent_cell_of_edge"].T - 1

    out_ds["edge_face_connectivity"] = xr.DataArray(
        data=edge_face_connectivity,
        dims=ugrid.EDGE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.EDGE_FACE_CONNECTIVITY_ATTRS,
    )

    edge_node_connectivity = in_ds["edge_vertices"].T - 1
    out_ds["edge_node_connectivity"] = xr.DataArray(
        data=edge_node_connectivity,
        dims=ugrid.EDGE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.EDGE_NODE_CONNECTIVITY_ATTRS,
    )

    return out_ds, source_dims_dict


def _read_icon(ext_ds, use_dual=False):
    """Reads and encodes an ICON mesh into the UGRID conventions."""
    out_ds = xr.Dataset()

    if not use_dual:
        return _primal_to_ugrid(ext_ds, out_ds)
    else:
        raise ValueError("Conversion of the ICON Dual mesh is not yet supported.")
