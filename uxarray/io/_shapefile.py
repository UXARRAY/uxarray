import geopandas as gpd
import xarray as xr
import numpy as np
from uxarray.conventions import ugrid
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


def _read_shpfile(filepath):
    """Read shape file, use geopandas.

    Parameters: xarray.Dataset, required
    Returns: ugrid aware xarray.Dataset
    """

    grid_ds = xr.Dataset()

    # TODO: Fix conn to not use max_coord_size
    gdf, max_coord_size = _gpd_read(filepath)

    # Initialize as an empty numpy array
    node_lon = np.array([])
    node_lat = np.array([])
    connectivity = np.empty((0, max_coord_size - 1), dtype=INT_DTYPE)

    node_index = 0

    for index, row in gdf.iterrows():
        geometry = row["geometry"]

        # Handle polygons
        if geometry.geom_type == "Polygon":
            node_lat, node_lon, connectivity, node_index = _read_polygon(
                geometry, node_lat, node_lon, connectivity, node_index
            )

        # Handle multipolygons
        elif geometry.geom_type == "MultiPolygon":
            node_lat, node_lon, connectivity, node_index = _read_multipolygon(
                geometry, node_lat, node_lon, connectivity, node_index
            )

        # Handle other geometry types
        else:
            print(f"Geometry type: {geometry.geom_type}")
            # print("- Coordinates:", geometry.coords.xy

    # Now we have the node_lon, node_lat, and connectivity, we can create the xarray.Dataset
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

    Parameters: filepath (string): filepath to shapefile, note all files .shp, .shx, .dbf, etc. must be in the same directory
    Returns: gpd.GeoDataFrame
    """

    try:
        gdf = gpd.read_file(filepath)

        # Check if the CRS is already set
        if gdf.crs is None:
            # Manually set the CRS if it's missing (e.g., EPSG:4326 for WGS84)
            gdf = gdf.set_crs("EPSG:4326")
            print("Original CRS: None\nAssigned CRS:", gdf.crs)

        # If the shapefile is not in WGS84 (latitude and longitude), transform it
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
            print("Transformed CRS:", gdf.crs)
    except Exception as e:
        print(f"An error occurred while reading the shapefile: {e}")

    # Apply the function to each geometry in the GeoDataFrame
    num_nodes = gdf["geometry"].apply(get_num_nodes)

    # Find the maximum number of nodes
    max_polygon_nodes = num_nodes.max()

    return gdf, max_polygon_nodes


def get_num_nodes(geom):
    """Function to get number of nodes in a polygon/multipolygon.

    Parameters: geom (gpd.geom): GeoPandas geometry object
    Returns: Maximum number of nodes in a polygon or Multipolygon, return 0 for other types of geometry
    """
    if geom.geom_type == "Polygon":
        return len(geom.exterior.coords)
    elif geom.geom_type == "MultiPolygon":
        return max(len(polygon.exterior.coords) for polygon in geom.geoms)
    else:
        return 0  # Not a polygon or multipolygon


def _read_multipolygon(geometry, node_lat, node_lon, connectivity, node_index):
    """Read a multipolygon.

    Parameters: xarray.Dataset, required
    Returns: ugrid aware xarray.Dataset
    """

    # Note: geometry.geoms is a list of Polygon objects that don't have holes and are not self-intersecting
    for polygon in geometry.geoms:
        # Append the longitude and latitude coordinates of the polygon's exterior (excluding the last coordinate)
        # This is because a square has 5 points where the last point is the same as the first one
        node_lon = np.append(node_lon, polygon.exterior.coords.xy[0][:-1])
        node_lat = np.append(node_lat, polygon.exterior.coords.xy[1][:-1])

        # Calculate the size of the polygon's coordinates
        coord_size_polygon = len(polygon.exterior.coords.xy[0][:-1])

        # TODO: Fix to not use max_coord_size and implement a variable dim 2D array
        max_coord_size = connectivity.shape[1]

        # Create a new row for the connectivity array with indices for the new polygon nodes
        new_row = np.array(range(node_index, node_index + coord_size_polygon))

        # Determine the number of elements to pad to match the max_coord_size
        padding_length = max_coord_size - len(new_row)

        # Create a padding array filled with the specified INT_FILL_VALUE
        padding_array = np.full(padding_length, INT_FILL_VALUE)

        # Concatenate the new row with the padding array to maintain the required size
        new_row = np.concatenate((new_row, padding_array))

        # Stack the new row onto the connectivity array
        connectivity = np.vstack((connectivity, new_row))

        # Update the node index to reflect the addition of the new polygon nodes
        node_index += coord_size_polygon

    # Return the updated latitude and longitude arrays, connectivity array, and node index
    return node_lat, node_lon, connectivity, node_index



def _read_polygon(polygon, node_lat, node_lon, connectivity, node_index):
    """Read a polygon.

    Parameters: xarray.Dataset, required
    Returns: ugrid aware xarray.Dataset
    """

    # Determine the maximum coordinate size from the connectivity array
    max_coord_size = connectivity.shape[1]

    # Append the longitude and latitude coordinates of the polygon's exterior (excluding the last coordinate)
    node_lon = np.append(node_lon, polygon.exterior.coords.xy[0][:-1])
    node_lat = np.append(node_lat, polygon.exterior.coords.xy[1][:-1])

    # Calculate the size of the polygon's coordinates
    coord_size_polygon = len(polygon.exterior.coords.xy[0][:-1])

    # Create a new row for the connectivity array with indices for the new polygon nodes
    new_row = np.array(range(node_index, node_index + coord_size_polygon))

    # Determine the number of elements to pad to match the max_coord_size
    padding_length = max_coord_size - len(new_row)

    # Create a padding array filled with the specified INT_FILL_VALUE
    padding_array = np.full(padding_length, INT_FILL_VALUE)

    # Concatenate the new row with the padding array to maintain the required size
    new_row = np.concatenate((new_row, padding_array))

    # Stack the new row onto the connectivity array
    connectivity = np.vstack((connectivity, new_row))

    # Update the node index to reflect the addition of the new polygon nodes
    node_index += coord_size_polygon

    # Return the updated latitude and longitude arrays, connectivity array, and node index
    return node_lat, node_lon, connectivity, node_index

