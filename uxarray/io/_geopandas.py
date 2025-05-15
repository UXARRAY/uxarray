import os
import warnings

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE, WGS84_CRS
from uxarray.conventions import ugrid

# --- New/Modified Helper Functions ---


def _process_polygon(
    polygon: Polygon,
    node_lat_list: list,
    node_lon_list: list,
    face_conn_list: list,
    node_index: int,
) -> tuple:
    """Extract nodes and face connectivity for a single Polygon."""
    if polygon.exterior is None or polygon.is_empty:
        # Handle empty polygons
        return node_lat_list, node_lon_list, face_conn_list, node_index

    face_indices = []
    # Extract exterior ring coordinates
    for lon, lat in polygon.exterior.coords[
        :-1
    ]:  # Skip the last point as it's a duplicate of the first
        node_lon_list.append(lon)
        node_lat_list.append(lat)
        face_indices.append(node_index)
        node_index += 1

    # Handle interior rings (holes) - UGRID supports this, but this simple implementation does NOT
    if polygon.interiors:
        warnings.warn(
            "Interior rings (holes) in polygons are not currently supported and will be ignored."
        )
        # You would need more complex logic here to handle interior rings for full UGRID compliance

    if face_indices:  # Only append if the polygon had actual nodes
        face_conn_list.append(face_indices)

    return node_lat_list, node_lon_list, face_conn_list, node_index


def _process_multipolygon(
    multipolygon: MultiPolygon,
    node_lat_list: list,
    node_lon_list: list,
    face_conn_list: list,
    node_index: int,
) -> tuple:
    """Process each polygon within a MultiPolygon."""
    if multipolygon.is_empty:
        return node_lat_list, node_lon_list, face_conn_list, node_index

    for polygon in multipolygon.geoms:
        node_lat_list, node_lon_list, face_conn_list, node_index = _process_polygon(
            polygon, node_lat_list, node_lon_list, face_conn_list, node_index
        )
    return node_lat_list, node_lon_list, face_conn_list, node_index


def _process_linestring(
    linestring: LineString,
    node_lat_list: list,
    node_lon_list: list,
    edge_conn_list: list,
    node_index: int,
) -> tuple:
    """Extract nodes and edge connectivity for a single LineString."""
    if linestring.is_empty:
        return node_lat_list, node_lon_list, edge_conn_list, node_index

    coords = list(linestring.coords)
    if len(coords) < 2:
        warnings.warn("Skipping LineString with fewer than 2 coordinates.")
        return (
            node_lat_list,
            node_lon_list,
            edge_conn_list,
            node_index,
        )  # Not a valid edge sequence

    line_node_indices = []
    # Extract nodes and store their indices
    for lon, lat in coords:
        node_lon_list.append(lon)
        node_lat_list.append(lat)
        line_node_indices.append(node_index)
        node_index += 1

    # Create edge connectivity (pairs of node indices)
    for i in range(len(line_node_indices) - 1):
        edge_conn_list.append([line_node_indices[i], line_node_indices[i + 1]])

    return node_lat_list, node_lon_list, edge_conn_list, node_index


def _process_multilinestring(
    multilinestring: MultiLineString,
    node_lat_list: list,
    node_lon_list: list,
    edge_conn_list: list,
    node_index: int,
) -> tuple:
    """Process each linestring within a MultiLineString."""
    if multilinestring.is_empty:
        return node_lat_list, node_lon_list, edge_conn_list, node_index

    for linestring in multilinestring.geoms:
        node_lat_list, node_lon_list, edge_conn_list, node_index = _process_linestring(
            linestring, node_lat_list, node_lon_list, edge_conn_list, node_index
        )
    return node_lat_list, node_lon_list, edge_conn_list, node_index


def _process_point(
    point: Point, node_lat_list: list, node_lon_list: list, node_index: int
) -> tuple:
    """Extract node for a single Point."""
    if point.is_empty:
        warnings.warn("Skipping empty Point geometry.")
        return node_lat_list, node_lon_list, node_index

    lon, lat = point.coords[0]
    node_lon_list.append(lon)
    node_lat_list.append(lat)
    # Points just contribute to the node list, no connectivity created here
    return node_lat_list, node_lon_list, node_index + 1


def _process_multipoint(
    multipoint: MultiPoint, node_lat_list: list, node_lon_list: list, node_index: int
) -> tuple:
    """Process each point within a MultiPoint."""
    if multipoint.is_empty:
        return node_lat_list, node_lon_list, node_index

    for point in multipoint.geoms:
        node_lat_list, node_lon_list, node_index = _process_point(
            point, node_lat_list, node_lon_list, node_index
        )
    return node_lat_list, node_lon_list, node_index


def _get_max_nodes_per_face(face_connectivity_list: list) -> int:
    """Calculate the maximum number of nodes among all collected face connectivities."""
    if not face_connectivity_list:
        return 0
    return max(len(face) for face in face_connectivity_list)


# --- Modified Core Functions ---


def _read_geodataframe(filepath, driver=None, **kwargs):
    """Read geospatial data, using geopandas, supporting Polygons, Lines, and Points.

    Parameters
    ----------
    filepath : str
        Filepath to geospatial data.
    driver : str, optional
        Driver to use, by default None, in which case geopandas will try to infer the driver from the file extension.
    **kwargs
        Keyword arguments to pass to geopandas.read_file().

    Returns
    -------
    xr.Dataset
        ugrid aware xarray.Dataset containing nodes, and face/edge connectivity
        if those geometry types were present.
    dict
        Dictionary mapping source mesh dimensions to UGRID dimensions.
    """
    grid_ds = xr.Dataset()

    gdf = _gpd_read(filepath, driver=driver, **kwargs)

    # The logic for extracting geometry info is now unified in _extract_geometry_info
    node_lon, node_lat, face_node_connectivity, edge_node_connectivity = (
        _extract_geometry_info(gdf)
    )

    # Source dimensions dictionary to return
    source_dims_dict = {}

    # Add node coordinates to the dataset
    if node_lon.size > 0:
        grid_ds[ugrid.NODE_DIM] = np.arange(node_lon.size, dtype=INT_DTYPE)
        grid_ds["node_lon"] = xr.DataArray(
            data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
        )
        grid_ds["node_lat"] = xr.DataArray(
            data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
        )
        source_dims_dict["n_node"] = ugrid.NODE_DIM

    # Add face connectivity if polygons were found or create empty face data for validation
    max_nodes_per_face = 0
    if face_node_connectivity.size > 0:
        grid_ds[ugrid.FACE_DIM] = np.arange(
            face_node_connectivity.shape[0], dtype=INT_DTYPE
        )
        grid_ds[ugrid.N_MAX_FACE_NODES_DIM] = np.arange(
            face_node_connectivity.shape[1], dtype=INT_DTYPE
        )
        source_dims_dict["n_face"] = ugrid.FACE_DIM
        source_dims_dict["n_max_face_nodes"] = ugrid.N_MAX_FACE_NODES_DIM
        max_nodes_per_face = face_node_connectivity.shape[1]

        grid_ds["face_node_connectivity"] = xr.DataArray(
            data=face_node_connectivity,
            dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
            attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
        )
    else:
        # Create empty face connectivity for non-polygon geometries
        # This is needed for Grid.validate() to work
        grid_ds[ugrid.FACE_DIM] = np.arange(0, dtype=INT_DTYPE)
        max_nodes_per_face = 1  # Minimum value to avoid dimension size issues
        grid_ds[ugrid.N_MAX_FACE_NODES_DIM] = np.arange(
            max_nodes_per_face, dtype=INT_DTYPE
        )
        source_dims_dict["n_face"] = ugrid.FACE_DIM
        source_dims_dict["n_max_face_nodes"] = ugrid.N_MAX_FACE_NODES_DIM

        # Empty face_node_connectivity
        empty_face_conn = np.empty((0, max_nodes_per_face), dtype=INT_DTYPE)
        grid_ds["face_node_connectivity"] = xr.DataArray(
            data=empty_face_conn,
            dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
            attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
        )

    # Add edge connectivity if linestrings were found
    if edge_node_connectivity.size > 0:
        grid_ds[ugrid.EDGE_DIM] = np.arange(
            edge_node_connectivity.shape[0], dtype=INT_DTYPE
        )
        grid_ds["two"] = np.arange(2, dtype=INT_DTYPE)
        source_dims_dict["n_edge"] = ugrid.EDGE_DIM
        source_dims_dict["two"] = "two"

        grid_ds["edge_node_connectivity"] = xr.DataArray(
            data=edge_node_connectivity,
            dims=ugrid.EDGE_NODE_CONNECTIVITY_DIMS,
            attrs=ugrid.EDGE_NODE_CONNECTIVITY_ATTRS,
        )

    # Add standard UGRID convention attributes
    grid_ds.attrs["Conventions"] = ugrid.CONVENTIONS_ATTR
    grid_ds.attrs["title"] = os.path.basename(filepath)

    # Create base grid topology attributes
    topology_attrs = dict(ugrid.BASE_GRID_TOPOLOGY_ATTRS)

    # Adjust topology attributes based on what's present in the dataset
    if "node_lon" in grid_ds and "node_lat" in grid_ds:
        topology_attrs["node_coordinates"] = "node_lon node_lat"

    # Add face_node_connectivity to topology
    if "face_node_connectivity" in grid_ds:
        topology_attrs["face_node_connectivity"] = "face_node_connectivity"

    # Add edge_node_connectivity to topology if present
    if "edge_node_connectivity" in grid_ds:
        topology_attrs["edge_node_connectivity"] = "edge_node_connectivity"

    # Add grid topology variable pointing to dimensions
    grid_topology = xr.DataArray(
        data=np.int32(0),  # Placeholder integer value
        attrs=topology_attrs,
    )
    grid_ds["mesh"] = grid_topology

    return grid_ds, source_dims_dict


def _gpd_read(filepath, driver=None, **kwargs):
    """Read a geospatial data using geopandas and set CRS.

    Parameters
    ----------
    filepath : str
        Filepath to geospatial data.
    driver : str, optional
        Driver to use, by default None, in which case geopandas will try to infer the driver from the file extension.
    **kwargs
        Keyword arguments to pass to geopandas.read_file().

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with geometries and CRS set.
    """
    try:
        gdf = gpd.read_file(filepath, driver=driver, **kwargs)
        gdf = _set_crs(gdf)
        return gdf
    except Exception as e:
        print(f"An error occurred while reading the geospatial data: {e}")
        # Re-raise the exception after printing, so the calling function knows it failed
        raise


def _set_crs(gdf):
    """Set CRS for GeoDataFrame if not already set or transform to WGS84.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to set CRS for.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with CRS set or transformed to WGS84.
    """
    original_crs = gdf.crs
    if original_crs is None:
        gdf = gdf.set_crs(WGS84_CRS)
        print("Original CRS: None\nAssigned CRS:", gdf.crs)
    elif original_crs != WGS84_CRS:
        gdf = gdf.to_crs(WGS84_CRS)
        print("Transformed CRS:", gdf.crs)
    else:
        print("CRS already WGS84:", gdf.crs)

    return gdf


def _extract_geometry_info(gdf) -> tuple:
    """Extract node, face, and edge connectivity information from GeoDataFrame.

    Handles Polygon, MultiPolygon, LineString, MultiLineString, Point, MultiPoint.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with geometries.

    Returns
    -------
    np.ndarray
        Array of longitudes.
    np.ndarray
        Array of latitudes.
    np.ndarray
        Face node connectivity array (padded), or empty if no polygons.
    np.ndarray
        Edge node connectivity array (shape (N, 2)), or empty if no lines.
    """
    node_lon_list = []
    node_lat_list = []
    face_connectivity_list = []  # List of lists, each inner list is indices for a face
    edge_connectivity_list = []  # List of [start_node, end_node] pairs for edges

    node_index = 0  # Keep track of the global node index

    for _, row in gdf.iterrows():
        geometry = row["geometry"]
        if geometry is None or getattr(geometry, "is_empty", False):
            continue  # Skip empty or None geometries

        # Use isinstance checks for robustness
        if isinstance(geometry, Polygon):
            node_lat_list, node_lon_list, face_connectivity_list, node_index = (
                _process_polygon(
                    geometry,
                    node_lat_list,
                    node_lon_list,
                    face_connectivity_list,
                    node_index,
                )
            )
        elif isinstance(geometry, MultiPolygon):
            node_lat_list, node_lon_list, face_connectivity_list, node_index = (
                _process_multipolygon(
                    geometry,
                    node_lat_list,
                    node_lon_list,
                    face_connectivity_list,
                    node_index,
                )
            )
        elif isinstance(geometry, LineString):
            node_lat_list, node_lon_list, edge_connectivity_list, node_index = (
                _process_linestring(
                    geometry,
                    node_lat_list,
                    node_lon_list,
                    edge_connectivity_list,
                    node_index,
                )
            )
        elif isinstance(geometry, MultiLineString):
            node_lat_list, node_lon_list, edge_connectivity_list, node_index = (
                _process_multilinestring(
                    geometry,
                    node_lat_list,
                    node_lon_list,
                    edge_connectivity_list,
                    node_index,
                )
            )
        elif isinstance(geometry, Point):
            node_lat_list, node_lon_list, node_index = _process_point(
                geometry, node_lat_list, node_lon_list, node_index
            )
        elif isinstance(geometry, MultiPoint):
            node_lat_list, node_lon_list, node_index = _process_multipoint(
                geometry, node_lat_list, node_lon_list, node_index
            )
        else:
            # Use warnings for unsupported types encountered
            geom_type = getattr(geometry, "geom_type", str(type(geometry)))
            warnings.warn(
                f"Unsupported geometry type encountered: {geom_type}. Skipping."
            )

    # Convert lists to numpy arrays at the end
    node_lon = np.array(node_lon_list, dtype=np.float64)  # Use float64 for coords
    node_lat = np.array(node_lat_list, dtype=np.float64)

    # --- Process Face Connectivity ---
    max_nodes_per_face = _get_max_nodes_per_face(face_connectivity_list)
    if face_connectivity_list:
        # Pad face connectivity lists to the max number of nodes per face
        padded_face_connectivity = np.full(
            (len(face_connectivity_list), max_nodes_per_face),
            INT_FILL_VALUE,  # Use fill value for padding
            dtype=INT_DTYPE,
        )
        for i, face_indices in enumerate(face_connectivity_list):
            padded_face_connectivity[i, : len(face_indices)] = face_indices
        face_node_connectivity = padded_face_connectivity
    else:
        # Return an empty array with at least one column for compatibility with UGRID
        face_node_connectivity = np.empty(
            (0, max(1, max_nodes_per_face)), dtype=INT_DTYPE
        )

    # --- Process Edge Connectivity ---
    if edge_connectivity_list:
        # Edge connectivity is always pairs of nodes
        edge_node_connectivity = np.array(edge_connectivity_list, dtype=INT_DTYPE)
    else:
        # Return an empty array with 2 columns (for start/end nodes)
        edge_node_connectivity = np.empty((0, 2), dtype=INT_DTYPE)

    return node_lon, node_lat, face_node_connectivity, edge_node_connectivity


def _get_num_nodes(geom):
    """Helper to get node count for specific geometry types (used for potential max calculations).
    Note: This function is not strictly needed for the new _extract_geometry_info,
    but kept for completeness or if needed elsewhere.
    """
    if geom is None or geom.is_empty:
        return 0
    if isinstance(geom, (Polygon, LineString)):
        return len(geom.coords)
    elif isinstance(geom, MultiPolygon):
        # Sum nodes from all constituent polygons (exterior only for simple count)
        return sum(len(p.exterior.coords) for p in geom.geoms if p.exterior)
    elif isinstance(geom, MultiLineString):
        # Sum nodes from all constituent linestrings
        return sum(len(ls.coords) for ls in geom.geoms)
    elif isinstance(geom, Point):
        return 1
    elif isinstance(geom, MultiPoint):
        return len(geom.geoms)
    else:
        return 0
