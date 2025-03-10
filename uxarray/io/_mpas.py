import xarray as xr
import numpy as np

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.conventions import ugrid, descriptors


def _mpas_to_ugrid_dims(in_ds, primal=True):
    """TODO:"""
    source_dims_dict = {}
    if primal:
        source_dims_dict["nVertices"] = ugrid.NODE_DIM
        source_dims_dict[in_ds["verticesOnCell"].dims[0]] = ugrid.FACE_DIM
        source_dims_dict[in_ds["verticesOnCell"].dims[1]] = ugrid.N_MAX_FACE_NODES_DIM

        if "verticesOnEdge" in in_ds:
            source_dims_dict[in_ds["verticesOnEdge"].dims[0]] = "n_edge"
    else:
        source_dims_dict[in_ds["latCell"].dims[0]] = ugrid.NODE_DIM
        source_dims_dict[in_ds["cellsOnVertex"].dims[0]] = ugrid.FACE_DIM
        source_dims_dict[in_ds["cellsOnVertex"].dims[1]] = ugrid.N_MAX_FACE_NODES_DIM

        if "cellsOnEdge" in in_ds:
            source_dims_dict[in_ds["cellsOnEdge"].dims[0]] = "n_edge"

    return source_dims_dict


def _primal_to_ugrid(in_ds, out_ds):
    """Encodes the MPAS Primal-Mesh in the UGRID conventions."""
    source_dims_dict = {}

    if "lonVertex" in in_ds:
        _parse_node_latlon_coords(in_ds, out_ds, mesh_type="primal")

    if "xVertex" in in_ds:
        _parse_node_xyz_coords(in_ds, out_ds, mesh_type="primal")

    if "lonCell" in in_ds:
        _parse_face_latlon_coords(in_ds, out_ds, mesh_type="primal")

    if "xCell" in in_ds:
        _parse_face_xyz_coords(in_ds, out_ds, mesh_type="primal")

    if "lonEdge" in in_ds:
        _parse_edge_latlon_coords(in_ds, out_ds, mesh_type="primal")

    if "xEdge" in in_ds:
        _parse_edge_xyz_coords(in_ds, out_ds, mesh_type="primal")

    _parse_face_nodes(in_ds, out_ds, mesh_type="primal")
    _parse_node_faces(in_ds, out_ds, mesh_type="primal")

    if "verticesOnEdge" in in_ds:
        _parse_edge_nodes(in_ds, out_ds, "primal")
        source_dims_dict[in_ds["verticesOnEdge"].dims[0]] = "n_edge"

    if "edgesOnCell" in in_ds:
        _parse_face_edges(in_ds, out_ds, mesh_type="primal")

    if "cellsOnEdge" in in_ds:
        _parse_edge_faces(in_ds, out_ds, mesh_type="primal")

    if "dvEdge" in in_ds:
        _parse_edge_node_distances(in_ds, out_ds)

    if "dcEdge" in in_ds:
        _parse_edge_face_distances(in_ds, out_ds)

    if "cellsOnCell" in in_ds:
        _parse_face_faces(in_ds, out_ds)

    if "areaCell" in in_ds:
        _parse_face_areas(in_ds, out_ds, mesh_type="primal")

    if "boundaryVertex" in in_ds:
        _parse_boundary_node_indices(in_ds, out_ds, mesh_type="primal")

    # set global attributes
    _parse_global_attrs(in_ds, out_ds)

    # populate source dims
    source_dims_dict["nVertices"] = ugrid.NODE_DIM
    source_dims_dict[in_ds["verticesOnCell"].dims[0]] = ugrid.FACE_DIM
    source_dims_dict[in_ds["verticesOnCell"].dims[1]] = ugrid.N_MAX_FACE_NODES_DIM

    return source_dims_dict


def _dual_to_ugrid(in_ds, out_ds):
    """Encodes the MPAS Dual-Mesh in the UGRID conventions."""
    source_dims_dict = {}

    if "lonCell" in in_ds:
        _parse_node_latlon_coords(in_ds, out_ds, mesh_type="dual")

    if "xCell" in in_ds:
        _parse_node_xyz_coords(in_ds, out_ds, mesh_type="dual")

    if "lonVertex" in in_ds:
        _parse_face_latlon_coords(in_ds, out_ds, mesh_type="dual")

    if "xVertex" in in_ds:
        _parse_face_xyz_coords(in_ds, out_ds, mesh_type="dual")

    if "lonEdge" in in_ds:
        _parse_edge_latlon_coords(in_ds, out_ds, mesh_type="dual")

    if "xEdge" in in_ds:
        _parse_edge_xyz_coords(in_ds, out_ds, mesh_type="dual")

    _parse_face_nodes(in_ds, out_ds, mesh_type="dual")
    _parse_node_faces(in_ds, out_ds, mesh_type="dual")

    if "cellsOnEdge" in in_ds:
        _parse_edge_nodes(in_ds, out_ds, mesh_type="dual")
        source_dims_dict[in_ds["cellsOnEdge"].dims[0]] = "n_edge"

    if "edgesOnVertex" in in_ds:
        _parse_face_edges(in_ds, out_ds, mesh_type="dual")

    if "verticesOnEdge" in in_ds:
        _parse_edge_faces(in_ds, out_ds, mesh_type="dual")

    if "dvEdge" in in_ds:
        _parse_edge_node_distances(in_ds, out_ds)

    if "dcEdge" in in_ds:
        _parse_edge_face_distances(in_ds, out_ds)

    if "areaTriangle" in in_ds:
        _parse_face_areas(in_ds, out_ds, mesh_type="dual")

    # set global attributes
    _parse_global_attrs(in_ds, out_ds)

    # populate source dims
    source_dims_dict[in_ds["latCell"].dims[0]] = ugrid.NODE_DIM
    source_dims_dict[in_ds["cellsOnVertex"].dims[0]] = ugrid.FACE_DIM
    source_dims_dict[in_ds["cellsOnVertex"].dims[1]] = ugrid.N_MAX_FACE_NODES_DIM

    return source_dims_dict


def _parse_node_latlon_coords(in_ds, out_ds, mesh_type):
    """Parses cartesian corner node coordinates for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        node_lon = 180.0 * in_ds["lonVertex"] / np.pi
        node_lat = 180.0 * in_ds["latVertex"] / np.pi

        # Ensure correct dimension name
        node_lon = node_lon.rename({"nVertices": ugrid.NODE_DIM})
        node_lat = node_lat.rename({"nVertices": ugrid.NODE_DIM})
    else:
        node_lon = 180.0 * in_ds["lonCell"] / np.pi
        node_lat = 180.0 * in_ds["latCell"] / np.pi

        # Ensure correct dimension name
        node_lon = node_lon.rename({"nCells": ugrid.NODE_DIM})
        node_lat = node_lat.rename({"nCells": ugrid.NODE_DIM})

    out_ds["node_lon"] = node_lon.assign_attrs(ugrid.NODE_LON_ATTRS)
    out_ds["node_lat"] = node_lat.assign_attrs(ugrid.NODE_LAT_ATTRS)


def _parse_node_xyz_coords(in_ds, out_ds, mesh_type):
    """Parses cartesian corner node coordinates for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        node_x = in_ds["xVertex"]
        node_y = in_ds["yVertex"]
        node_z = in_ds["zVertex"]

        # Ensure correct dimension name
        node_x = node_x.rename({"nVertices": ugrid.NODE_DIM})
        node_y = node_y.rename({"nVertices": ugrid.NODE_DIM})
        node_z = node_z.rename({"nVertices": ugrid.NODE_DIM})
    else:
        node_x = in_ds["xCell"]
        node_y = in_ds["yCell"]
        node_z = in_ds["zCell"]

        # Ensure correct dimension name
        node_x = node_x.rename({"nCells": ugrid.NODE_DIM})
        node_y = node_y.rename({"nCells": ugrid.NODE_DIM})
        node_z = node_z.rename({"nCells": ugrid.NODE_DIM})

    out_ds["node_x"] = node_x.assign_attrs(ugrid.NODE_X_ATTRS)
    out_ds["node_y"] = node_y.assign_attrs(ugrid.NODE_Y_ATTRS)
    out_ds["node_z"] = node_z.assign_attrs(ugrid.NODE_Z_ATTRS)


def _parse_face_latlon_coords(in_ds, out_ds, mesh_type):
    """Parses latlon face center coordinates for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        face_lon = 180.0 * in_ds["lonCell"] / np.pi
        face_lat = 180.0 * in_ds["latCell"] / np.pi

        # Ensure correct dimension name
        face_lon = face_lon.rename({"nCells": ugrid.FACE_DIM})
        face_lat = face_lat.rename({"nCells": ugrid.FACE_DIM})
    else:
        face_lon = 180.0 * in_ds["lonVertex"] / np.pi
        face_lat = 180.0 * in_ds["latVertex"] / np.pi

        # Ensure correct dimension name
        face_lon = face_lon.rename({"nVertices": ugrid.FACE_DIM})
        face_lat = face_lat.rename({"nVertices": ugrid.FACE_DIM})

    out_ds["face_lon"] = face_lon.assign_attrs(ugrid.FACE_LON_ATTRS)
    out_ds["face_lat"] = face_lat.assign_attrs(ugrid.FACE_LAT_ATTRS)


def _parse_face_xyz_coords(in_ds, out_ds, mesh_type):
    """Parses cartesian face center coordinates for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        face_x = in_ds["xCell"]
        face_y = in_ds["yCell"]
        face_z = in_ds["zCell"]

        # Ensure correct dimension name
        face_x = face_x.rename({"nCells": ugrid.FACE_DIM})
        face_y = face_y.rename({"nCells": ugrid.FACE_DIM})
        face_z = face_z.rename({"nCells": ugrid.FACE_DIM})
    else:
        face_x = in_ds["xVertex"]
        face_y = in_ds["yVertex"]
        face_z = in_ds["zVertex"]

        # Ensure correct dimension name
        face_x = face_x.rename({"nVertices": ugrid.FACE_DIM})
        face_y = face_y.rename({"nVertices": ugrid.FACE_DIM})
        face_z = face_z.rename({"nVertices": ugrid.FACE_DIM})

    out_ds["face_x"] = face_x.assign_attrs(ugrid.FACE_X_ATTRS)
    out_ds["face_y"] = face_y.assign_attrs(ugrid.FACE_Y_ATTRS)
    out_ds["face_z"] = face_z.assign_attrs(ugrid.FACE_Z_ATTRS)


def _parse_edge_latlon_coords(in_ds, out_ds, mesh_type):
    """Parses latlon edge node coordinates."""
    edge_lon = 180.0 * in_ds["lonEdge"] / np.pi
    edge_lat = 180.0 * in_ds["latEdge"] / np.pi

    # Ensure correct dimension name
    edge_lon = edge_lon.rename({"nEdges": ugrid.EDGE_DIM})
    edge_lat = edge_lat.rename({"nEdges": ugrid.EDGE_DIM})

    out_ds["edge_lon"] = edge_lon.assign_attrs(ugrid.EDGE_LON_ATTRS)
    out_ds["edge_lat"] = edge_lat.assign_attrs(ugrid.EDGE_LAT_ATTRS)


def _parse_edge_xyz_coords(in_ds, out_ds, mesh_type):
    """Parses cartesian edge node coordinates."""
    edge_x = in_ds["xEdge"]
    edge_y = in_ds["yEdge"]
    edge_z = in_ds["zEdge"]

    # Ensure correct dimension name
    edge_x = edge_x.rename({"nEdges": ugrid.EDGE_DIM})
    edge_y = edge_y.rename({"nEdges": ugrid.EDGE_DIM})
    edge_z = edge_z.rename({"nEdges": ugrid.EDGE_DIM})

    out_ds["edge_x"] = edge_x.assign_attrs(ugrid.EDGE_X_ATTRS)
    out_ds["edge_y"] = edge_y.assign_attrs(ugrid.EDGE_Y_ATTRS)
    out_ds["edge_z"] = edge_z.assign_attrs(ugrid.EDGE_Z_ATTRS)


def _parse_face_nodes(in_ds, out_ds, mesh_type):
    """Parses face node connectivity for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        verticesOnCell = in_ds["verticesOnCell"].astype(INT_DTYPE)
        nEdgesOnCell = in_ds["nEdgesOnCell"].astype(INT_DTYPE)

        # Replace padded values with fill values
        verticesOnCell = _replace_padding(verticesOnCell, nEdgesOnCell)

        # Replace missing/zero values with fill values
        verticesOnCell = _replace_zeros(verticesOnCell)

        # Convert to zero-indexed
        face_nodes = _to_zero_index(verticesOnCell)

    else:
        cellsOnVertex = in_ds["cellsOnVertex"].astype(INT_DTYPE)

        # Replace missing/zero values with fill values
        cellsOnVertex = _replace_zeros(cellsOnVertex)

        # Convert to zero-indexed
        face_nodes = _to_zero_index(cellsOnVertex)

    out_ds["face_node_connectivity"] = face_nodes.assign_attrs(
        ugrid.FACE_NODE_CONNECTIVITY_ATTRS
    ).rename(dict(zip(face_nodes.dims, ugrid.FACE_NODE_CONNECTIVITY_DIMS)))


def _parse_edge_nodes(in_ds, out_ds, mesh_type):
    """Parses edge node connectivity for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        verticesOnEdge = in_ds["verticesOnEdge"].astype(INT_DTYPE)

        # Replace missing/zero values with fill values
        verticesOnEdge = _replace_zeros(verticesOnEdge)

        # Convert to zero-indexed
        edge_nodes = _to_zero_index(verticesOnEdge)

    else:
        cellsOnEdge = in_ds["cellsOnEdge"].astype(INT_DTYPE)

        # Replace missing/zero values with fill values
        cellsOnEdge = _replace_zeros(cellsOnEdge)

        # Convert to zero-indexed
        edge_nodes = _to_zero_index(cellsOnEdge)

    out_ds["edge_node_connectivity"] = edge_nodes.assign_attrs(
        ugrid.EDGE_NODE_CONNECTIVITY_ATTRS
    ).rename(dict(zip(edge_nodes.dims, ugrid.EDGE_NODE_CONNECTIVITY_DIMS)))


def _parse_node_faces(in_ds, out_ds, mesh_type):
    """Parses node face connectivity for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        cellsOnVertex = in_ds["cellsOnVertex"].astype(INT_DTYPE)

        # Replace missing/zero values with fill values
        cellsOnVertex = _replace_zeros(cellsOnVertex)

        # Convert to zero-indexed
        node_faces = _to_zero_index(cellsOnVertex)
    else:
        verticesOnCell = in_ds["verticesOnCell"].astype(INT_DTYPE)
        nEdgesOnCell = in_ds["nEdgesOnCell"].astype(INT_DTYPE)

        # Replace padded values with fill values
        verticesOnCell = _replace_padding(verticesOnCell, nEdgesOnCell)

        # Replace missing/zero values with fill values
        verticesOnCell = _replace_zeros(verticesOnCell)

        # Convert to zero-indexed
        node_faces = _to_zero_index(verticesOnCell)

    out_ds["node_face_connectivity"] = node_faces.assign_attrs(
        ugrid.NODE_FACE_CONNECTIVITY_ATTRS
    ).rename(dict(zip(node_faces.dims, ugrid.NODE_FACE_CONNECTIVITY_DIMS)))


def _parse_face_edges(in_ds, out_ds, mesh_type):
    """Parses face edge connectivity for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        edgesOnCell = in_ds["edgesOnCell"].astype(INT_DTYPE)
        nEdgesOnCell = in_ds["nEdgesOnCell"].astype(INT_DTYPE)

        # Replace padded values with fill values
        edgesOnCell = _replace_padding(edgesOnCell, nEdgesOnCell)

        # Replace missing/zero values with fill values
        edgesOnCell = _replace_zeros(edgesOnCell)

        # Convert to zero-indexed
        face_edges = _to_zero_index(edgesOnCell)

    else:
        edgesOnVertex = in_ds["edgesOnVertex"].astype(INT_DTYPE)

        # Replace missing/zero values with fill values
        edgesOnVertex = _replace_zeros(edgesOnVertex)

        # Convert to zero-indexed
        face_edges = _to_zero_index(edgesOnVertex)

    out_ds["face_edge_connectivity"] = face_edges.assign_attrs(
        ugrid.FACE_EDGE_CONNECTIVITY_ATTRS
    ).rename(dict(zip(face_edges.dims, ugrid.FACE_EDGE_CONNECTIVITY_DIMS)))


def _parse_edge_faces(in_ds, out_ds, mesh_type):
    """Parses edge face connectivity for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        cellsOnEdge = in_ds["cellsOnEdge"].astype(INT_DTYPE)

        # Replace missing/zero values with fill values
        cellsOnEdge = _replace_zeros(cellsOnEdge)

        # Convert to zero-indexed
        edge_faces = _to_zero_index(cellsOnEdge)

    else:
        verticesOnEdge = in_ds["verticesOnEdge"].astype(INT_DTYPE)

        # Replace missing/zero values with fill values
        verticesOnEdge = _replace_zeros(verticesOnEdge)

        # Convert to zero-indexed
        edge_faces = _to_zero_index(verticesOnEdge)

    out_ds["edge_face_connectivity"] = edge_faces.assign_attrs(
        ugrid.EDGE_FACE_CONNECTIVITY_ATTRS
    ).rename(dict(zip(edge_faces.dims, ugrid.EDGE_FACE_CONNECTIVITY_DIMS)))


def _parse_edge_node_distances(in_ds, out_ds):
    """Parses ``edge_node_distances``"""
    edge_node_distances = in_ds["dvEdge"] / in_ds.attrs["sphere_radius"]

    out_ds["edge_node_distances"] = edge_node_distances.assign_attrs(
        descriptors.EDGE_NODE_DISTANCES_ATTRS
    ).rename({"nEdges": ugrid.EDGE_DIM})


def _parse_edge_face_distances(in_ds, out_ds):
    """Parses ``edge_face_distances``"""
    edge_face_distances = in_ds["dcEdge"] / in_ds.attrs["sphere_radius"]

    out_ds["edge_face_distances"] = edge_face_distances.assign_attrs(
        descriptors.EDGE_FACE_DISTANCES_ATTRS
    ).rename({"nEdges": ugrid.EDGE_DIM})


def _parse_global_attrs(in_ds, out_ds):
    """Helper to parse MPAS global attributes."""
    out_ds.attrs = in_ds.attrs


def _parse_face_faces(in_ds, out_ds):
    """Parses face-face connectivity for Primal Mesh."""
    cellsOnCell = in_ds["cellsOnCell"].astype(INT_DTYPE)
    nEdgesOnCell = in_ds["nEdgesOnCell"].astype(INT_DTYPE)

    # Replace padded values with fill values
    cellsOnCell = _replace_padding(cellsOnCell, nEdgesOnCell)

    # Replace missing/zero values with fill values
    cellsOnCell = _replace_zeros(cellsOnCell)

    # Convert to zero-indexed
    face_face_connectivity = _to_zero_index(cellsOnCell)

    out_ds["face_face_connectivity"] = face_face_connectivity.assign_attrs(
        ugrid.FACE_FACE_CONNECTIVITY_ATTRS
    ).rename(dict(zip(face_face_connectivity.dims, ugrid.FACE_FACE_CONNECTIVITY_DIMS)))


def _parse_face_areas(in_ds, out_ds, mesh_type):
    """Parses the face area for either a primal or dual grid."""
    if mesh_type == "primal":
        face_area = in_ds["areaCell"]
    else:
        face_area = in_ds["areaTriangle"]

    out_ds["face_areas"] = face_area.assign_attrs(descriptors.FACE_AREAS_ATTRS).rename(
        {face_area.dims[0]: ugrid.FACE_DIM}
    )


def _parse_boundary_node_indices(in_ds, out_ds, mesh_type):
    """Parses the boundary node indices."""
    boundary_node_mask = in_ds["boundaryVertex"]
    boundary_node_indices = boundary_node_mask.where(boundary_node_mask).dropna(
        dim=boundary_node_mask.dims[0]
    )

    # Convert to integer indices
    boundary_node_indices = boundary_node_indices.coords[
        boundary_node_indices.dims[0]
    ].astype(INT_DTYPE)

    # Ensure zero-indexed
    boundary_node_indices = boundary_node_indices - 1

    out_ds["boundary_node_indices"] = boundary_node_indices.rename(
        {"nVertices": "n_boundary_nodes"}
    )


def _replace_padding(verticesOnCell, nEdgesOnCell):
    """Replaces padded values in verticesOnCell with fill-value."""
    maxEdges = verticesOnCell.sizes[verticesOnCell.dims[1]]
    edge_indices = xr.DataArray(np.arange(maxEdges), dims=[verticesOnCell.dims[1]])

    mask = edge_indices >= nEdgesOnCell
    verticesOnCell = verticesOnCell.where(~mask, INT_FILL_VALUE)
    return verticesOnCell


def _replace_zeros(grid_var):
    """Replaces zeros with fill-value."""
    grid_var = grid_var.where(grid_var != 0, INT_FILL_VALUE)
    return grid_var


def _to_zero_index(grid_var):
    """Converts one-indexed data to zero-indexed."""
    grid_var = xr.where(grid_var != INT_FILL_VALUE, grid_var - 1, grid_var)
    return grid_var


def _read_mpas(ext_ds, use_dual=False):
    """Reads an MPAS Grid dataset and encodes either the Primal or Dual Mesh in the UGRID conventions."""
    ds = xr.Dataset()

    if use_dual:
        source_dim_map = _dual_to_ugrid(ext_ds, ds)
    else:
        source_dim_map = _primal_to_ugrid(ext_ds, ds)

    return ds, source_dim_map
