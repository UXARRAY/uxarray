import geopandas as gpd
import xarray as xr
import numpy as np
from uxarray.conventions import ugrid
from uxarray.constants import INT_FILL_VALUE


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
    connectivity = np.empty((0, max_coord_size), dtype=np.int32)

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

    print("Node Longitude:", node_lon)
    print("Node Latitude:", node_lat)
    print("Connectivity:", connectivity)

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

        # Print general information about the GeoDataFrame
        print("GeoDataFrame Shapefile Information:")
        print(gdf.info())
    except Exception as e:
        print(f"An error occurred while reading the shapefile: {e}")

    # Apply the function to each geometry in the GeoDataFrame
    num_nodes = gdf["geometry"].apply(get_num_nodes)

    # Find the maximum number of nodes
    max_polygon_nodes = num_nodes.max()

    return gdf, max_polygon_nodes


# Function to get number of nodes in a polygon/multipolygon
def get_num_nodes(geom):
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

    # Note: geometry.geoms is a list of Polygon objects - that don't have a hole and are not self-intersecting
    for polygon in geometry.geoms:
        # leave the last one, as a square has 5 points, 5 point is the same as the first one
        node_lon = np.append(node_lon, polygon.exterior.coords.xy[0][:-1])
        node_lat = np.append(node_lat, polygon.exterior.coords.xy[1][:-1])

        coord_size_polygon = len(polygon.exterior.coords.xy[0][:-1])

        # TODO: Fix to not use max_coord_size and something like a variable dim 2d array
        max_coord_size = connectivity.shape[1]
        new_row = np.array(range(node_index, node_index + coord_size_polygon))
        new_row = np.pad(
            new_row,
            (0, max_coord_size - len(new_row)),
            "constant",
            constant_values=INT_FILL_VALUE,
        )
        connectivity = np.vstack((connectivity, new_row))

        node_index += coord_size_polygon

    return node_lat, node_lon, connectivity, node_index


def _read_polygon(polygon, node_lat, node_lon, connectivity, node_index):
    """Read a polygon.

    Parameters: xarray.Dataset, required
    Returns: ugrid aware xarray.Dataset
    """

    max_coord_size = connectivity.shape[1]

    node_lon = np.append(node_lon, polygon.exterior.coords.xy[0][:-1])
    node_lat = np.append(node_lat, polygon.exterior.coords.xy[1][:-1])
    coord_size_polygon = len(polygon.exterior.coords.xy[0][:-1])

    # TODO: Fix to not use max_coord_size and something like a variable dim 2d array
    new_row = np.array(range(node_index, node_index + coord_size_polygon))
    new_row = np.pad(
        new_row,
        (0, max_coord_size - len(new_row)),
        "constant",
        constant_values=INT_FILL_VALUE,
    )
    connectivity = np.vstack((connectivity, new_row))

    node_index += coord_size_polygon

    # print(connectivity, "Connectivity size:", len(connectivity), type(connectivity))

    return node_lat, node_lon, connectivity, node_index
