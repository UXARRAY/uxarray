from warnings import warn
import geopandas as gpd

def _read_shpfile(filepath):
    """Read shape file, use geopandas.

    Parameters: xarray.Dataset, required
    Returns: ugrid aware xarray.Dataset
    """
    
    try:
        gdf = gpd.read_file(filepath)

        # Print general information about the GeoDataFrame
        print("Shapefile information:")
        print(gdf.info())

        # Access and process geometry types
        for index, row in gdf.iterrows():
            geometry = row['geometry']

            # Handle polygons
            if geometry.geom_type == 'Polygon':
                print("Polygon found:")
                
                # print("- Coordinates:", geometry.exterior.coords.xy)
                print("- Number of coordinates:", len(geometry.exterior.coords))
                print("- Area:", geometry.area)
                print("- Perimeter:", geometry.length)

            # Handle multipolygons
            elif geometry.geom_type == 'MultiPolygon':
                print("MultiPolygon found:")
                for polygon in geometry:
                    print("- Polygon coordinates:", polygon.exterior.coords.xy)

            # Handle other geometry types
            else:
                print(f"Geometry type: {geometry.geom_type}")
                print("- Coordinates:", geometry.coords.xy)

    except Exception as e:
        print(f"An error occurred while reading the shapefile: {e}")

    # raise RuntimeError(
    #     "Function not implemented yet. FYI, attempted to read SHAPE file: "
    #     + str(filepath)
    # )
    uxgrid=0
    source_dims_dict = 0
    return uxgrid, source_dims_dict
    # TODO: create ds
