import numpy as np
import xarray as xr
from shapely.geometry import (
    GeometryCollection,  # Import GeometryCollection
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from uxarray.constants import (  # Assuming WGS84_CRS is defined, e.g., as "EPSG:4326"
    INT_DTYPE,
    INT_FILL_VALUE,
    WGS84_CRS,
)
from uxarray.conventions import (
    ugrid,  # Assuming ugrid contains constants like NODE_DIM, FACE_DIM, etc.
)


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
    int
        Maximum number of nodes in a polygon/multipolygon (including closing node).
        Returns 0 if no polygons are found.
    """
    try:
        gdf = gpd.read_file(filepath, driver=driver, **kwargs)
        gdf = _set_crs(gdf)
    except Exception as e:
        # Added a more specific error message for the read failure
        print(f"Error reading geospatial data from {filepath}: {e}")
        # Re-raise the exception after printing, so the calling function knows it failed
        raise

    # Calculate max nodes per face (including closing node, as per original logic)
    # Handle potential empty GeoDataFrame or lack of polygon geometries
    max_polygon_nodes = 0
    if not gdf.empty:
        # Apply _get_num_nodes only to polygon geometries
        polygon_geoms = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        if not polygon_geoms.empty:
            # Use apply on the geometry column and handle potential errors or None
            max_polygon_nodes = polygon_geoms.geometry.apply(
                lambda geom: _get_num_nodes(geom) if geom else 0
            ).max()

    return gdf, max_polygon_nodes


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
    # Use .to_epsg() for WGS84 code 4326 for robustness
    # Assuming WGS84_CRS constant is equivalent to "EPSG:4326" or a pyproj.CRS object for it.
    # Using "EPSG:4326" string directly is often safer for standard CRS.
    # Let's assume WGS84_CRS is a usable representation like "EPSG:4326" or pyproj.CRS.from_epsg(4326)
    wgs84_crs = WGS84_CRS  # Use the constant provided by the caller

    original_crs = gdf.crs
    if original_crs is None:
        print("Original CRS: None. Assigning WGS84...")
        gdf = gdf.set_crs(wgs84_crs)
        print("Assigned CRS:", gdf.crs)
    elif (
        original_crs != wgs84_crs
    ):  # Directly compare CRS objects or their string representations
        print(f"Transforming CRS from {original_crs} to {wgs84_crs}...")
        gdf = gdf.to_crs(wgs84_crs)
        print("Transformed CRS:", gdf.crs)
    else:
        print("CRS already WGS84:", gdf.crs)

    return gdf


def _get_num_nodes(geom):
    """Get number of nodes in a polygon/multipolygon (including closing node).

    This helper is used by _gpd_read to determine the max width
    needed for connectivity padding based on the original code's logic.
    """
    if geom is None or geom.is_empty:
        return 0
    if isinstance(geom, Polygon):
        return len(geom.exterior.coords)  # Includes closing node
    elif isinstance(geom, MultiPolygon):
        # Max nodes from any constituent polygon (exterior only)
        return max((len(p.exterior.coords) if p.exterior else 0) for p in geom.geoms)
    # Return 0 for other types as this is specifically for polygon connectivity width
    return 0


def _process_polygon(
    polygon: Polygon,
    node_lon_list: list,
    node_lat_list: list,
    node_index: int,
    max_coord_size: int,
) -> tuple:
    """Extract nodes and generate a single face connectivity row for a Polygon."""
    if polygon.exterior is None or polygon.is_empty:
        return node_index, None  # No nodes added, no connectivity row

    face_indices = []
    # Extract exterior ring coordinates (excluding the last point as it's a duplicate)
    for lon, lat in polygon.exterior.coords[:-1]:
        node_lon_list.append(lon)
        node_lat_list.append(lat)
        face_indices.append(node_index)
        node_index += 1

    # Handle interior rings (holes) - UGRID supports this, but this simple implementation does NOT
    if polygon.interiors:
        warnings.warn(
            "Interior rings (holes) in polygons are not currently supported and will be ignored."
        )

    if not face_indices:  # Should not happen for valid polygons with >=3 nodes
        return node_index, None

    # Create a single connectivity row (np.array)
    coord_size_polygon = len(face_indices)  # Number of unique nodes in this face
    max_conn_width = max_coord_size - 1  # Max width of connectivity array

    if coord_size_polygon > max_conn_width:
        # This indicates an inconsistency if max_coord_size from _gpd_read
        # was based on actual max exterior coords. Should not happen if _gpd_read is correct.
        warnings.warn(
            f"Polygon has more unique nodes ({coord_size_polygon}) than max connectivity width ({max_conn_width}). Truncating connectivity row."
        )
        face_indices = face_indices[:max_conn_width]
        coord_size_polygon = max_conn_width

    new_row = np.full(max_conn_width, INT_FILL_VALUE, dtype=INT_DTYPE)
    new_row[:coord_size_polygon] = face_indices

    return node_index, new_row  # Return the node index and the single connectivity row


def _process_multipolygon(
    multipolygon: MultiPolygon,
    node_lon_list: list,
    node_lat_list: list,
    node_index: int,
    max_coord_size: int,
) -> tuple:
    """Process each polygon within a MultiPolygon, collecting connectivity rows."""
    if multipolygon.is_empty:
        return node_index, []  # Return empty list of rows

    face_connectivity_rows = []
    for polygon in multipolygon.geoms:
        node_index, row = _process_polygon(
            polygon, node_lon_list, node_lat_list, node_index, max_coord_size
        )
        if row is not None:
            face_connectivity_rows.append(row)

    return node_index, face_connectivity_rows  # Return list of rows


def _process_linestring(
    linestring: LineString,
    node_lon_list: list,
    node_lat_list: list,
    node_index: int,
) -> tuple:
    """Extract nodes and generate edge connectivity rows for a LineString."""
    if linestring.is_empty:
        return node_index, []  # Return empty list of rows

    coords = list(linestring.coords)
    if len(coords) < 2:
        warnings.warn("Skipping LineString with fewer than 2 coordinates.")
        return node_index, []  # Not a valid edge sequence

    line_node_indices = []
    # Extract nodes and store their indices
    for lon, lat in coords:
        node_lon_list.append(lon)
        node_lat_list.append(lat)
        line_node_indices.append(node_index)
        node_index += 1

    # Create edge connectivity rows (pairs of node indices)
    edge_connectivity_rows = []
    if len(line_node_indices) >= 2:
        for i in range(len(line_node_indices) - 1):
            edge_connectivity_rows.append(
                np.array(
                    [line_node_indices[i], line_node_indices[i + 1]], dtype=INT_DTYPE
                )
            )

    return node_index, edge_connectivity_rows  # Return list of rows


def _process_multilinestring(
    multilinestring: MultiLineString,
    node_lon_list: list,
    node_lat_list: list,
    node_index: int,
) -> tuple:
    """Process each linestring within a MultiLineString, collecting connectivity rows."""
    if multilinestring.is_empty:
        return node_index, []  # Return empty list of rows

    edge_connectivity_rows = []
    for linestring in multilinestring.geoms:
        node_index, rows = _process_linestring(
            linestring, node_lon_list, node_lat_list, node_index
        )
        edge_connectivity_rows.extend(rows)  # Extend the list of rows

    return node_index, edge_connectivity_rows  # Return list of rows


def _process_point(
    point: Point, node_lon_list: list, node_lat_list: list, node_index: int
) -> int:
    """Extract node for a single Point."""
    if point.is_empty:
        warnings.warn("Skipping empty Point geometry.")
        return node_index

    lon, lat = point.coords[0]
    node_lon_list.append(lon)
    node_lat_list.append(lat)
    # Points just contribute to the node list, no connectivity created here
    return node_index + 1


def _process_multipoint(
    multipoint: MultiPoint, node_lon_list: list, node_lat_list: list, node_index: int
) -> int:
    """Process each point within a MultiPoint."""
    if multipoint.is_empty:
        return node_index

    for point in multipoint.geoms:
        node_index = _process_point(point, node_lon_list, node_lat_list, node_index)
    return node_index


def _process_geometry_collection(
    geometry_collection: GeometryCollection,
    node_lon_list: list,
    node_lat_list: list,
    node_index: int,
    max_coord_size: int,
) -> tuple:
    """Recursively process geometries within a GeometryCollection, collecting rows."""
    if geometry_collection.is_empty:
        return node_index, [], []  # Return empty lists of rows

    face_connectivity_rows = []
    edge_connectivity_rows = []

    for geom in geometry_collection.geoms:
        if geom is None or getattr(geom, "is_empty", False):
            continue

        geom_type = geom.geom_type

        if geom_type == "Point":
            node_index = _process_point(geom, node_lon_list, node_lat_list, node_index)
        elif geom_type == "MultiPoint":
            node_index = _process_multipoint(
                geom, node_lon_list, node_lat_list, node_index
            )
        elif geom_type == "LineString":
            node_index, rows = _process_linestring(
                geom, node_lon_list, node_lat_list, node_index
            )
            edge_connectivity_rows.extend(rows)
        elif geom_type == "MultiLineString":
            node_index, rows = _process_multilinestring(
                geom, node_lon_list, node_lat_list, node_index
            )
            edge_connectivity_rows.extend(rows)
        elif geom_type == "Polygon":
            node_index, row = _process_polygon(
                geom, node_lon_list, node_lat_list, node_index, max_coord_size
            )
            if row is not None:
                face_connectivity_rows.append(row)
        elif geom_type == "MultiPolygon":
            node_index, rows = _process_multipolygon(
                geom, node_lon_list, node_lat_list, node_index, max_coord_size
            )
            face_connectivity_rows.extend(rows)
        elif geom_type == "GeometryCollection":
            # Recursive call
            node_index, face_rows, edge_rows = _process_geometry_collection(
                geom, node_lon_list, node_lat_list, node_index, max_coord_size
            )
            face_connectivity_rows.extend(face_rows)
            edge_connectivity_rows.extend(edge_rows)
        else:
            warnings.warn(
                f"Unsupported geometry type encountered: {geom_type}. Skipping."
            )

    return node_index, face_connectivity_rows, edge_connectivity_rows


def _extract_geometry_info(gdf, max_coord_size):
    """Extract node, face, and edge connectivity information from GeoDataFrame.

    Collects nodes and connectivity rows into lists, then converts to NumPy arrays once.
    """
    node_lon_list = []
    node_lat_list = []
    face_connectivity_rows = []  # List to collect face connectivity rows (np.arrays)
    edge_connectivity_rows = []  # List to collect edge connectivity rows (np.arrays)

    node_index = 0

    for _, row in gdf.iterrows():
        geometry = row.get("geometry")  # Use .get for safety
        if geometry is None or getattr(geometry, "is_empty", False):
            continue

        geom_type = geometry.geom_type

        if geom_type == "Polygon":
            node_index, row = _process_polygon(
                geometry, node_lon_list, node_lat_list, node_index, max_coord_size
            )
            if row is not None:
                face_connectivity_rows.append(row)
        elif geom_type == "MultiPolygon":
            node_index, rows = _process_multipolygon(
                geometry, node_lon_list, node_lat_list, node_index, max_coord_size
            )
            face_connectivity_rows.extend(rows)
        elif geom_type == "LineString":
            node_index, rows = _process_linestring(
                geometry, node_lon_list, node_lat_list, node_index
            )
            edge_connectivity_rows.extend(rows)
        elif geom_type == "MultiLineString":
            node_index, rows = _process_multilinestring(
                geometry, node_lon_list, node_lat_list, node_index
            )
            edge_connectivity_rows.extend(rows)
        elif geom_type == "Point":
            node_index = _process_point(
                geometry, node_lon_list, node_lat_list, node_index
            )
        elif geom_type == "MultiPoint":
            node_index = _process_multipoint(
                geometry, node_lon_list, node_lat_list, node_index
            )
        elif geom_type == "GeometryCollection":
            # Handle GeometryCollection recursively
            node_index, face_rows, edge_rows = _process_geometry_collection(
                geometry, node_lon_list, node_lat_list, node_index, max_coord_size
            )
            face_connectivity_rows.extend(face_rows)
            edge_connectivity_rows.extend(edge_rows)
        else:
            # Warning handled within _process_geometry_collection or specific type handlers if needed
            warnings.warn(
                f"Unsupported geometry type encountered: {geom_type}. Skipping."
            )

    # Convert lists to numpy arrays at the end (more efficient than vstack in loop)
    node_lon = np.array(node_lon_list, dtype=np.float64)  # Use float64 for coords
    node_lat = np.array(node_lat_list, dtype=np.float64)

    # Stack connectivity rows
    # Max connectivity width is max_coord_size - 1 based on _gpd_read and _get_num_nodes
    max_conn_width = max(
        1, max_coord_size - 1
    )  # Ensure min width is 1 even if no polygons

    if face_connectivity_rows:
        face_node_connectivity = np.vstack(face_connectivity_rows)
    else:
        # Create empty array with correct dimensions if no faces
        face_node_connectivity = np.empty((0, max_conn_width), dtype=INT_DTYPE)

    if edge_connectivity_rows:
        edge_node_connectivity = np.vstack(edge_connectivity_rows)
    else:
        # Create empty array with 2 columns for edges if no edges
        edge_node_connectivity = np.empty((0, 2), dtype=INT_DTYPE)

    return (
        node_lon,
        node_lat,
        face_node_connectivity,
        edge_node_connectivity,
        max_conn_width,
    )


def _read_geodataframe(filepath, driver=None, **kwargs):
    """Read geospatial data, using geopandas, supporting various geometry types,
    and build a UGRID-aware xarray.Dataset.

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

    # max_coord_size is calculated here based on the original code's pattern
    gdf, max_coord_size = _gpd_read(filepath, driver=driver, **kwargs)

    # Extract geometry info into numpy arrays (more efficient stacking)
    (
        node_lon,
        node_lat,
        face_node_connectivity,
        edge_node_connectivity,
        max_conn_width,
    ) = _extract_geometry_info(gdf, max_coord_size)

    # Source dimensions dictionary to return
    source_dims_dict = {}

    # Add node coordinates to the dataset if any nodes were found
    if node_lon.size > 0:
        grid_ds[ugrid.NODE_DIM] = np.arange(node_lon.size, dtype=INT_DTYPE)
        grid_ds["node_lon"] = xr.DataArray(
            data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
        )
        grid_ds["node_lat"] = xr.DataArray(
            data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
        )
        source_dims_dict["n_node"] = ugrid.NODE_DIM  # Map source dim to UGRID dim

    # Add face connectivity if polygons were found or create empty face data for validation
    if face_node_connectivity.shape[0] > 0:
        grid_ds[ugrid.FACE_DIM] = np.arange(
            face_node_connectivity.shape[0], dtype=INT_DTYPE
        )
        # Use max_conn_width determined during extraction for this dimension
        grid_ds[ugrid.N_MAX_FACE_NODES_DIM] = np.arange(max_conn_width, dtype=INT_DTYPE)
        source_dims_dict["n_face"] = ugrid.FACE_DIM
        source_dims_dict["n_max_face_nodes"] = ugrid.N_MAX_FACE_NODES_DIM

        grid_ds["face_node_connectivity"] = xr.DataArray(
            data=face_node_connectivity,
            dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
            attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
        )
    else:
        # Create empty face connectivity and dimensions for non-polygon geometries
        # This is needed for Grid.validate() to work even if no faces are present.
        grid_ds[ugrid.FACE_DIM] = np.arange(0, dtype=INT_DTYPE)
        # Use max_conn_width even for empty faces
        grid_ds[ugrid.N_MAX_FACE_NODES_DIM] = np.arange(max_conn_width, dtype=INT_DTYPE)
        source_dims_dict["n_face"] = ugrid.FACE_DIM
        source_dims_dict["n_max_face_nodes"] = ugrid.N_MAX_FACE_NODES_DIM

        # Empty face_node_connectivity matching the expected dimensions
        empty_face_conn = np.empty((0, max_conn_width), dtype=INT_DTYPE)
        grid_ds["face_node_connectivity"] = xr.DataArray(
            data=empty_face_conn,
            dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
            attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
        )

    # Add edge connectivity if linestrings were found
    if edge_node_connectivity.shape[0] > 0:
        grid_ds[ugrid.EDGE_DIM] = np.arange(
            edge_node_connectivity.shape[0], dtype=INT_DTYPE
        )
        # The 'two' dimension for edge connectivity is always size 2
        grid_ds["two"] = np.arange(2, dtype=INT_DTYPE)  # Create a dimension variable
        source_dims_dict["n_edge"] = ugrid.EDGE_DIM
        source_dims_dict["two"] = "two"  # Map source dim to UGRID dim

        grid_ds["edge_node_connectivity"] = xr.DataArray(
            data=edge_node_connectivity,
            dims=ugrid.EDGE_NODE_CONNECTIVITY_DIMS,  # Assumes this is ('edge', 'two')
            attrs=ugrid.EDGE_NODE_CONNECTIVITY_ATTRS,
        )
    # Note: No 'else' needed for edges as validate doesn't strictly require edge data if no edges exist.

    # Add standard UGRID convention attributes
    grid_ds.attrs["Conventions"] = ugrid.CONVENTIONS_ATTR
    grid_ds.attrs["title"] = os.path.basename(filepath)

    # Create base grid topology attributes
    topology_attrs = dict(ugrid.BASE_GRID_TOPOLOGY_ATTRS)

    # Adjust topology attributes based on what's present in the dataset
    if "node_lon" in grid_ds and "node_lat" in grid_ds:
        topology_attrs["node_coordinates"] = "node_lon node_lat"

    # Add face_node_connectivity to topology if actual faces are present
    if "face_node_connectivity" in grid_ds and face_node_connectivity.shape[0] > 0:
        topology_attrs["face_node_connectivity"] = "face_node_connectivity"

    # Add edge_node_connectivity to topology if present
    if "edge_node_connectivity" in grid_ds and edge_node_connectivity.shape[0] > 0:
        topology_attrs["edge_node_connectivity"] = "edge_node_connectivity"

    # Add grid topology variable pointing to dimensions
    grid_topology = xr.DataArray(
        data=np.int32(0),  # Placeholder integer value, shape should be ()
        attrs=topology_attrs,
    )
    grid_ds["mesh"] = grid_topology

    # The source_dims_dict mapping is used internally by UXarray, not added to the dataset itself.
    return grid_ds, source_dims_dict
