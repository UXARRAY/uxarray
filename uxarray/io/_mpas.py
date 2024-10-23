import xarray as xr
import numpy as np

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.conventions import ugrid, descriptors


def _primal_to_ugrid(in_ds, out_ds):
    """Encodes the MPAS Primal-Mesh in the UGRID conventions.

    Parameters
    ----------
    in_ds : xarray.Dataset
        Input MPAS dataset
    out_ds : xarray.Dataset
        Output dataset where the MPAS Primal-Mesh is encoded in the UGRID
        conventions
    """

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
    """Encodes the MPAS Dual-Mesh in the UGRID conventions.

    Parameters
    ----------
    in_ds : xarray.Dataset
        Input MPAS dataset
    out_ds : xarray.Dataset
        Output dataset where the MPAS Dual-Mesh is encoded in the UGRID
        conventions
    """

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
    """Parses cartesian corner node coordinates for either the Primal or Dual
    Mesh."""
    if mesh_type == "primal":
        node_lon = np.rad2deg(in_ds["lonVertex"].values)
        node_lat = np.rad2deg(in_ds["latVertex"].values)
    else:
        node_lon = np.rad2deg(in_ds["lonCell"].values)
        node_lat = np.rad2deg(in_ds["latCell"].values)

    out_ds["node_lon"] = xr.DataArray(
        node_lon, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LON_ATTRS
    )

    out_ds["node_lat"] = xr.DataArray(
        node_lat, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LAT_ATTRS
    )


def _parse_node_xyz_coords(in_ds, out_ds, mesh_type):
    """Parses cartesian corner node coordinates for either the Primal or Dual
    Mesh."""
    if mesh_type == "primal":
        node_x = in_ds["xVertex"].values
        node_y = in_ds["yVertex"].values
        node_z = in_ds["zVertex"].values
    else:
        # corners of dual-mesh cells (artesian)
        node_x = in_ds["xCell"].values
        node_y = in_ds["yCell"].values
        node_z = in_ds["zCell"].values

    out_ds["node_x"] = xr.DataArray(
        data=node_x, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_X_ATTRS
    )

    out_ds["node_y"] = xr.DataArray(
        data=node_y, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Y_ATTRS
    )

    out_ds["node_z"] = xr.DataArray(
        data=node_z, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Z_ATTRS
    )


def _parse_face_latlon_coords(in_ds, out_ds, mesh_type):
    """Parses latlon face center coordinates for either the Primal or Dual
    Mesh."""
    if mesh_type == "primal":
        face_lon = np.rad2deg(in_ds["lonCell"].values)
        face_lat = np.rad2deg(in_ds["latCell"].values)
    else:
        face_lon = np.rad2deg(in_ds["lonVertex"].values)
        face_lat = np.rad2deg(in_ds["latVertex"].values)

    out_ds["face_lon"] = xr.DataArray(
        face_lon, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_LON_ATTRS
    )

    out_ds["face_lat"] = xr.DataArray(
        face_lat, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_LAT_ATTRS
    )


def _parse_face_xyz_coords(in_ds, out_ds, mesh_type):
    """Parses cartesian face center coordinates for either the Primal or Dual
    Mesh."""
    if mesh_type == "primal":
        face_x = in_ds["xCell"].values
        face_y = in_ds["yCell"].values
        face_z = in_ds["zCell"].values
    else:
        # centers of dual-mesh cells (in degrees)
        face_x = in_ds["xVertex"].values
        face_y = in_ds["yVertex"].values
        face_z = in_ds["zVertex"].values

    out_ds["face_x"] = xr.DataArray(
        data=face_x, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_X_ATTRS
    )

    out_ds["face_y"] = xr.DataArray(
        data=face_y, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_Y_ATTRS
    )

    out_ds["face_z"] = xr.DataArray(
        data=face_z, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_Z_ATTRS
    )


def _parse_edge_latlon_coords(in_ds, out_ds, mesh_type):
    """Parses latlon edge node coordinates for either the Primal or Dual
    Mesh."""

    edge_lon = np.rad2deg(in_ds["lonEdge"].values)
    edge_lat = np.rad2deg(in_ds["latEdge"].values)

    out_ds["edge_lon"] = xr.DataArray(
        edge_lon, dims=[ugrid.EDGE_DIM], attrs=ugrid.EDGE_LON_ATTRS
    )

    out_ds["edge_lat"] = xr.DataArray(
        edge_lat, dims=[ugrid.EDGE_DIM], attrs=ugrid.EDGE_LAT_ATTRS
    )


def _parse_edge_xyz_coords(in_ds, out_ds, mesh_type):
    """Parses cartesian edge node coordinates for either the Primal or Dual
    Mesh."""
    edge_x = in_ds["xEdge"].values
    edge_y = in_ds["yEdge"].values
    edge_z = in_ds["zEdge"].values

    out_ds["edge_x"] = xr.DataArray(
        data=edge_x, dims=[ugrid.EDGE_DIM], attrs=ugrid.EDGE_X_ATTRS
    )

    out_ds["edge_y"] = xr.DataArray(
        data=edge_y, dims=[ugrid.EDGE_DIM], attrs=ugrid.EDGE_Y_ATTRS
    )

    out_ds["edge_z"] = xr.DataArray(
        data=edge_z, dims=[ugrid.EDGE_DIM], attrs=ugrid.EDGE_Z_ATTRS
    )


def _parse_face_nodes(in_ds, out_ds, mesh_type):
    """Parses face node connectivity for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        verticesOnCell = np.array(in_ds["verticesOnCell"].values, dtype=INT_DTYPE)

        nEdgesOnCell = np.array(in_ds["nEdgesOnCell"].values, dtype=INT_DTYPE)

        # replace padded values with fill values
        verticesOnCell = _replace_padding(verticesOnCell, nEdgesOnCell)

        # replace missing/zero values with fill values
        verticesOnCell = _replace_zeros(verticesOnCell)

        # convert to zero-indexed
        verticesOnCell = _to_zero_index(verticesOnCell)

        face_nodes = verticesOnCell

    else:
        cellsOnVertex = np.array(in_ds["cellsOnVertex"].values, dtype=INT_DTYPE)

        # replace missing/zero values with fill values
        cellsOnVertex = _replace_zeros(cellsOnVertex)

        # convert to zero-indexed
        cellsOnVertex = _to_zero_index(cellsOnVertex)

        face_nodes = cellsOnVertex

    out_ds["face_node_connectivity"] = xr.DataArray(
        data=face_nodes,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )


def _parse_edge_nodes(in_ds, out_ds, mesh_type):
    """Parses edge node connectivity for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        # vertex indices that saddle a given edge
        verticesOnEdge = np.array(in_ds["verticesOnEdge"].values, dtype=INT_DTYPE)

        # replace missing/zero values with fill value
        verticesOnEdge = _replace_zeros(verticesOnEdge)

        # convert to zero-indexed
        verticesOnEdge = _to_zero_index(verticesOnEdge)

        edge_nodes = verticesOnEdge
    else:
        # vertex indices that saddle a given edge
        cellsOnEdge = np.array(in_ds["cellsOnEdge"].values, dtype=INT_DTYPE)

        # replace missing/zero values with fill values
        cellsOnEdge = _replace_zeros(cellsOnEdge)

        # convert to zero-indexed
        cellsOnEdge = _to_zero_index(cellsOnEdge)

        edge_nodes = cellsOnEdge

    out_ds["edge_node_connectivity"] = xr.DataArray(
        data=edge_nodes,
        dims=ugrid.EDGE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.EDGE_NODE_CONNECTIVITY_ATTRS,
    )


def _parse_node_faces(in_ds, out_ds, mesh_type):
    """Parses node face connectivity for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        cellsOnVertex = np.array(in_ds["cellsOnVertex"].values, dtype=INT_DTYPE)

        # replace missing/zero values with fill values
        cellsOnVertex = _replace_zeros(cellsOnVertex)

        # convert to zero-indexed
        cellsOnVertex = _to_zero_index(cellsOnVertex)

        node_faces = cellsOnVertex
    else:
        verticesOnCell = np.array(in_ds["verticesOnCell"].values, dtype=INT_DTYPE)

        nEdgesOnCell = np.array(in_ds["nEdgesOnCell"].values, dtype=INT_DTYPE)

        # replace padded values with fill values
        verticesOnCell = _replace_padding(verticesOnCell, nEdgesOnCell)

        # replace missing/zero values with fill values
        verticesOnCell = _replace_zeros(verticesOnCell)

        # convert to zero-indexed
        verticesOnCell = _to_zero_index(verticesOnCell)

        node_faces = verticesOnCell

    out_ds["node_face_connectivity"] = xr.DataArray(
        data=node_faces,
        dims=ugrid.NODE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.NODE_FACE_CONNECTIVITY_ATTRS,
    )


def _parse_face_edges(in_ds, out_ds, mesh_type):
    """Parses face edge connectivity for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        edgesOnCell = np.array(in_ds["edgesOnCell"].values, dtype=INT_DTYPE)

        nEdgesOnCell = np.array(in_ds["nEdgesOnCell"].values, dtype=INT_DTYPE)

        # replace padded values with fill values
        edgesOnCell = _replace_padding(edgesOnCell, nEdgesOnCell)

        # replace missing/zero values with fill values
        edgesOnCell = _replace_zeros(edgesOnCell)

        # convert to zero-indexed
        edgesOnCell = _to_zero_index(edgesOnCell)

        face_edges = edgesOnCell

    else:
        edgesOnVertex = np.array(in_ds["edgesOnVertex"].values, dtype=INT_DTYPE)

        # replace missing/zero values with fill values
        edgesOnVertex = _replace_zeros(edgesOnVertex)

        # convert to zero-indexed
        edgesOnVertex = _to_zero_index(edgesOnVertex)

        face_edges = edgesOnVertex

    out_ds["face_edge_connectivity"] = xr.DataArray(
        data=face_edges,
        dims=ugrid.FACE_EDGE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_EDGE_CONNECTIVITY_ATTRS,
    )


def _parse_edge_faces(in_ds, out_ds, mesh_type):
    """Parses edge node connectivity for either the Primal or Dual Mesh."""
    if mesh_type == "primal":
        # vertex indices that saddle a given edge
        cellsOnEdge = np.array(in_ds["cellsOnEdge"].values, dtype=INT_DTYPE)

        # replace missing/zero values with fill values
        cellsOnEdge = _replace_zeros(cellsOnEdge)

        # convert to zero-indexed
        cellsOnEdge = _to_zero_index(cellsOnEdge)

        edge_faces = cellsOnEdge

    else:
        # vertex indices that saddle a given edge
        verticesOnEdge = np.array(in_ds["verticesOnEdge"].values, dtype=INT_DTYPE)

        # replace missing/zero values with fill value
        verticesOnEdge = _replace_zeros(verticesOnEdge)

        # convert to zero-indexed
        verticesOnEdge = _to_zero_index(verticesOnEdge)

        edge_faces = verticesOnEdge

    out_ds["edge_face_connectivity"] = xr.DataArray(
        data=edge_faces,
        dims=ugrid.EDGE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.EDGE_FACE_CONNECTIVITY_ATTRS,
    )


def _parse_edge_node_distances(in_ds, out_ds):
    """Parses ``edge_node_distances``"""
    edge_node_distances = in_ds["dvEdge"].values

    out_ds["edge_node_distances"] = xr.DataArray(
        data=edge_node_distances,
        dims=descriptors.EDGE_NODE_DISTANCES_DIMS,
        attrs=descriptors.EDGE_NODE_DISTANCES_ATTRS,
    )


def _parse_edge_face_distances(in_ds, out_ds):
    """Parses ``edge_face_distances``"""
    edge_face_distances = in_ds["dcEdge"].values

    out_ds["edge_face_distances"] = xr.DataArray(
        data=edge_face_distances,
        dims=descriptors.EDGE_FACE_DISTANCES_DIMS,
        attrs=descriptors.EDGE_FACE_DISTANCES_ATTRS,
    )


def _parse_global_attrs(in_ds, out_ds):
    """Helper to parse MPAS global attributes.

    Parameters
    ----------
    in_ds : xarray.Dataset
        Input MPAS dataset
    out_ds : xarray.Dataset
        Output UGRID dataset with parsed global attributes
    """

    out_ds.attrs = in_ds.attrs


def _parse_face_faces(in_ds, out_ds):
    """Parses face-face connectivity for Primal Mesh."""
    cellsOnCell = np.array(in_ds["cellsOnCell"].values, dtype=INT_DTYPE)
    nEdgesOnCell = np.array(in_ds["nEdgesOnCell"].values, dtype=INT_DTYPE)

    # replace padded values with fill values
    cellsOnCell = _replace_padding(cellsOnCell, nEdgesOnCell)

    # replace missing/zero values with fill values
    cellsOnCell = _replace_zeros(cellsOnCell)

    # make zero-indexed
    cellsOnCell = _to_zero_index(cellsOnCell)

    face_face_connectivity = cellsOnCell

    out_ds["face_face_connectivity"] = xr.DataArray(
        data=face_face_connectivity,
        dims=ugrid.FACE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_FACE_CONNECTIVITY_ATTRS,
    )


def _parse_face_areas(in_ds, out_ds, mesh_type):
    """Parses the face area for either a primal or dual grid."""

    if mesh_type == "primal":
        face_area = in_ds["areaCell"].data
    else:
        face_area = in_ds["areaTriangle"].data

    out_ds["face_areas"] = xr.DataArray(
        data=face_area,
        dims=descriptors.FACE_AREAS_DIMS,
        attrs=descriptors.FACE_AREAS_ATTRS,
    )


def _parse_boundary_node_indices(in_ds, out_ds, mesh_type):
    """Parses the face area for either a primal or dual grid."""

    boundary_node_mask = in_ds["boundaryVertex"].values
    boundary_node_indices = np.argwhere(boundary_node_mask).flatten().astype(INT_DTYPE)

    out_ds["boundary_node_indices"] = xr.DataArray(
        data=boundary_node_indices,
        dims=[
            "n_boundary_nodes",
        ],
    )


def _replace_padding(verticesOnCell, nEdgesOnCell):
    """Replaces the padded values in verticesOnCell defined by nEdgesOnCell
    with a fill-value.

    Parameters
    ----------
    verticesOnCell : numpy.ndarray
        Vertex indices that surround a given cell

    nEdgesOnCell : numpy.ndarray
        Number of edges on a given cell

    Returns
    -------
    verticesOnCell : numpy.ndarray
        Vertex indices that surround a given cell with padded values replaced
        by fill values, done in-place
    """

    # max vertices/edges per cell
    maxEdges = verticesOnCell.shape[1]

    # mask for non-padded values
    mask = np.arange(maxEdges) < nEdgesOnCell[:, None]

    # replace remaining padding or zeros with INT_FILL_VALUE
    verticesOnCell[np.logical_not(mask)] = INT_FILL_VALUE

    return verticesOnCell


def _replace_zeros(grid_var):
    """Replaces all instances of a zero (invalid/missing MPAS value) with a
    fill value.

    Parameters
    ----------
    grid_var : numpy.ndarray
        Grid variable that may contain zeros that need to be replaced

    Returns
    -------
    grid_var : numpy.ndarray
        Grid variable with zero replaced by fill values, done in-place
    """

    # replace all zeros with INT_FILL_VALUE
    grid_var[grid_var == 0] = INT_FILL_VALUE

    return grid_var


def _to_zero_index(grid_var):
    """Given an input using that is one-indexed, subtracts one from all non-
    fill value entries to convert to zero-indexed.

    Parameters
    ----------
    grid_var : numpy.ndarray
        Grid variable that is one-indexed

    Returns
    -------
    grid_var : numpy.ndarray
        Grid variable that is converted to zero-indexed, done in-place
    """

    # convert non-fill values to zero-indexed
    grid_var[grid_var != INT_FILL_VALUE] -= 1

    return grid_var


def _read_mpas(ext_ds, use_dual=False):
    """Function to read in a MPAS Grid dataset and encode either the Primal or
    Dual Mesh in the UGRID conventions.

    Adheres to the MPAS Mesh Specifications outlined in the following document:
    https://mpas-dev.github.io/files/documents/MPAS-MeshSpec.pdf

    Parameters
    ----------
    ext_ds : xarray.Dataset, required
        MPAS datafile of interest
    use_dual : bool, optional
        Flag to select whether to encode the Dual-Mesh. Defaults to False

    Returns
    -------
    ds : xarray.Dataset
        UGRID dataset derived from inputted MPAS dataset
    """

    # empty dataset that will contain our encoded MPAS mesh
    ds = xr.Dataset()

    # convert dual-mesh to UGRID
    if use_dual:
        source_dim_map = _dual_to_ugrid(ext_ds, ds)
    # convert primal-mesh to UGRID
    else:
        source_dim_map = _primal_to_ugrid(ext_ds, ds)

    return ds, source_dim_map
