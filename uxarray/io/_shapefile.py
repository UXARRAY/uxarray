import geopandas as gpd
import xarray as xr
import numpy as np
from uxarray.conventions import ugrid
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


def _read_shpfile(filepath):
    """Read shape file, use geopandas.

    Parameters
    ----------
    filepath : str
        Filepath to shapefile, note all files .shp, .shx, .dbf, etc. must be in the same directory.

    Returns
    -------
    xr.Dataset
        ugrid aware xarray.Dataset.
    """
    grid_ds = xr.Dataset()

    gdf, max_coord_size = _gpd_read(filepath)

    node_lon, node_lat, connectivity = _extract_geometry_info(gdf, max_coord_size)

    grid_ds["node_lon"] = xr.DataArray(
        data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )
    grid_ds["node_lat"] = xr.DataArray(
        data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )
    grid_ds["face_node_connectivity"] = xr.DataArray(
        data=connectivity,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    return grid_ds, None


def _gpd_read(filepath):
    """Read a shapefile using geopandas.

    Parameters
    ----------
    filepath : str
        Filepath to shapefile.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with geometries.
    int
        Maximum number of nodes in a polygon/multipolygon.
    """
    try:
        gdf = gpd.read_file(filepath)
        gdf = _set_crs(gdf)
    except Exception as e:
        print(f"An error occurred while reading the shapefile: {e}")

    max_polygon_nodes = gdf["geometry"].apply(_get_num_nodes).max()

    return gdf, max_polygon_nodes


def _set_crs(gdf):
    """Set CRS for GeoDataFrame if not already set.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to set CRS for.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with CRS set.
    """
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
        print("Original CRS: None\nAssigned CRS:", gdf.crs)

    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
        print("Transformed CRS:", gdf.crs)

    return gdf


def _extract_geometry_info(gdf, max_coord_size):
    """Extract node and connectivity information from GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with geometries.
    max_coord_size : int
        Maximum number of nodes in a polygon/multipolygon.

    Returns
    -------
    np.ndarray
        Array of longitudes.
    np.ndarray
        Array of latitudes.
    np.ndarray
        Connectivity array.
    """
    node_lon = np.array([])
    node_lat = np.array([])
    connectivity = np.empty((0, max_coord_size - 1), dtype=INT_DTYPE)

    node_index = 0

    for _, row in gdf.iterrows():
        geometry = row["geometry"]
        if geometry.geom_type == "Polygon":
            node_lat, node_lon, connectivity, node_index = _read_polygon(
                geometry, node_lat, node_lon, connectivity, node_index
            )
        elif geometry.geom_type == "MultiPolygon":
            node_lat, node_lon, connectivity, node_index = _read_multipolygon(
                geometry, node_lat, node_lon, connectivity, node_index
            )
        else:
            print(f"Unsupported geometry type: {geometry.geom_type}")

    return node_lon, node_lat, connectivity


def _get_num_nodes(geom):
    """Get number of nodes in a polygon/multipolygon.

    Parameters
    ----------
    geom : gpd.GeoSeries
        GeoPandas geometry object.

    Returns
    -------
    int
        Maximum number of nodes in a polygon or multipolygon.
    """
    if geom.geom_type == "Polygon":
        return len(geom.exterior.coords)
    elif geom.geom_type == "MultiPolygon":
        return max(len(polygon.exterior.coords) for polygon in geom.geoms)
    else:
        return 0  # Not a polygon or multipolygon


def _read_multipolygon(geometry, node_lat, node_lon, connectivity, node_index):
    """Read a multipolygon.

    Parameters
    ----------
    geometry : gpd.GeoSeries
        GeoPandas geometry object.
    node_lat : np.ndarray
        Array of latitudes.
    node_lon : np.ndarray
        Array of longitudes.
    connectivity : np.ndarray
        Connectivity array.
    node_index : int
        Index of the current node.

    Returns
    -------
    np.ndarray
        Updated array of latitudes.
    np.ndarray
        Updated array of longitudes.
    np.ndarray
        Updated connectivity array.
    int
        Updated node index.
    """
    for polygon in geometry.geoms:
        node_lon, node_lat, connectivity, node_index = _append_polygon_coords(
            polygon, node_lat, node_lon, connectivity, node_index
        )
    return node_lat, node_lon, connectivity, node_index


def _read_polygon(polygon, node_lat, node_lon, connectivity, node_index):
    """Read a polygon.

    Parameters
    ----------
    polygon : gpd.GeoSeries
        GeoPandas geometry object.
    node_lat : np.ndarray
        Array of latitudes.
    node_lon : np.ndarray
        Array of longitudes.
    connectivity : np.ndarray
        Connectivity array.
    node_index : int
        Index of the current node.

    Returns
    -------
    np.ndarray
        Updated array of latitudes.
    np.ndarray
        Updated array of longitudes.
    np.ndarray
        Updated connectivity array.
    int
        Updated node index.
    """
    return _append_polygon_coords(polygon, node_lat, node_lon, connectivity, node_index)


def _append_polygon_coords(polygon, node_lat, node_lon, connectivity, node_index):
    """Append polygon coordinates to node and connectivity arrays.

    Parameters
    ----------
    polygon : gpd.GeoSeries
        GeoPandas geometry object.
    node_lat : np.ndarray
        Array of latitudes.
    node_lon : np.ndarray
        Array of longitudes.
    connectivity : np.ndarray
        Connectivity array.
    node_index : int
        Index of the current node.

    Returns
    -------
    np.ndarray
        Updated array of latitudes.
    np.ndarray
        Updated array of longitudes.
    np.ndarray
        Updated connectivity array.
    int
        Updated node index.
    """
    node_lon = np.append(node_lon, polygon.exterior.coords.xy[0][:-1])
    node_lat = np.append(node_lat, polygon.exterior.coords.xy[1][:-1])

    coord_size_polygon = len(polygon.exterior.coords.xy[0][:-1])
    max_coord_size = connectivity.shape[1]

    new_row = np.array(range(node_index, node_index + coord_size_polygon))
    padding_length = max_coord_size - len(new_row)
    padding_array = np.full(padding_length, INT_FILL_VALUE)
    new_row = np.concatenate((new_row, padding_array))

    connectivity = np.vstack((connectivity, new_row))
    node_index += coord_size_polygon

    return node_lat, node_lon, connectivity, node_index
