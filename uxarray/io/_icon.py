import numpy as np
import xarray as xr

from uxarray.conventions import ugrid


def _icon_to_ugrid_dims(in_ds):
    """Parse ICON dimension names and map them to UGRID conventions."""
    source_dims_dict = {}

    # Coordinate-driven mappings
    if "vlat" in in_ds:
        source_dims_dict[in_ds["vlat"].dims[0]] = ugrid.NODE_DIM
    if "vlon" in in_ds:
        source_dims_dict[in_ds["vlon"].dims[0]] = ugrid.NODE_DIM

    if "elat" in in_ds:
        source_dims_dict[in_ds["elat"].dims[0]] = ugrid.EDGE_DIM
    if "elon" in in_ds:
        source_dims_dict[in_ds["elon"].dims[0]] = ugrid.EDGE_DIM

    if "clat" in in_ds:
        source_dims_dict[in_ds["clat"].dims[0]] = ugrid.FACE_DIM
    if "clon" in in_ds:
        source_dims_dict[in_ds["clon"].dims[0]] = ugrid.FACE_DIM

    # Connectivity-driven mappings
    if "vertex_of_cell" in in_ds:
        n_max_face_nodes_dim, face_dim = in_ds["vertex_of_cell"].dims
        source_dims_dict.setdefault(face_dim, ugrid.FACE_DIM)
        source_dims_dict.setdefault(n_max_face_nodes_dim, ugrid.N_MAX_FACE_NODES_DIM)

    if "edge_of_cell" in in_ds:
        n_max_face_edges_dim, face_dim = in_ds["edge_of_cell"].dims
        source_dims_dict.setdefault(face_dim, ugrid.FACE_DIM)
        source_dims_dict.setdefault(
            n_max_face_edges_dim, ugrid.FACE_EDGE_CONNECTIVITY_DIMS[1]
        )

    if "neighbor_cell_index" in in_ds:
        n_max_face_faces_dim, face_dim = in_ds["neighbor_cell_index"].dims
        source_dims_dict.setdefault(face_dim, ugrid.FACE_DIM)
        source_dims_dict.setdefault(
            n_max_face_faces_dim, ugrid.FACE_FACE_CONNECTIVITY_DIMS[1]
        )

    if "adjacent_cell_of_edge" in in_ds:
        two_dim, edge_dim = in_ds["adjacent_cell_of_edge"].dims
        source_dims_dict.setdefault(edge_dim, ugrid.EDGE_DIM)
        source_dims_dict.setdefault(two_dim, ugrid.EDGE_FACE_CONNECTIVITY_DIMS[1])

    if "edge_vertices" in in_ds:
        two_dim, edge_dim = in_ds["edge_vertices"].dims
        source_dims_dict.setdefault(edge_dim, ugrid.EDGE_DIM)
        source_dims_dict.setdefault(two_dim, ugrid.EDGE_NODE_CONNECTIVITY_DIMS[1])

    # Fall back to common ICON dimension names if they were not detected above
    for dim, ugrid_dim in {
        "vertex": ugrid.NODE_DIM,
        "edge": ugrid.EDGE_DIM,
        "cell": ugrid.FACE_DIM,
    }.items():
        if dim in in_ds.dims:
            source_dims_dict.setdefault(dim, ugrid_dim)

    # Keep only dims that actually exist on the dataset
    return {dim: name for dim, name in source_dims_dict.items() if dim in in_ds.dims}


def _primal_to_ugrid(in_ds, out_ds):
    """Encodes the Primal Mesh of an ICON Grid into the UGRID conventions."""
    source_dims_dict = _icon_to_ugrid_dims(in_ds)

    # rename dimensions to match ugrid conventions
    in_ds = in_ds.rename_dims(source_dims_dict)

    # node coordinates
    node_lon = 180.0 * in_ds["vlon"] / np.pi
    node_lat = 180.0 * in_ds["vlat"] / np.pi

    out_ds["node_lon"] = xr.DataArray(
        data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )
    out_ds["node_lat"] = xr.DataArray(
        data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    # edge coordinates
    edge_lon = 180.0 * in_ds["elon"] / np.pi
    edge_lat = 180.0 * in_ds["elat"] / np.pi

    out_ds["edge_lon"] = xr.DataArray(
        data=edge_lon, dims=ugrid.EDGE_DIM, attrs=ugrid.EDGE_LON_ATTRS
    )
    out_ds["edge_lat"] = xr.DataArray(
        data=edge_lat, dims=ugrid.EDGE_DIM, attrs=ugrid.EDGE_LAT_ATTRS
    )

    # face coordinates
    face_lon = 180.0 * in_ds["clon"] / np.pi
    face_lat = 180.0 * in_ds["clat"] / np.pi

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
