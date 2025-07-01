import os
import numpy as np
from pathlib import Path
import uxarray as ux
import tempfile
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, Polygon

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

shp_filename = current_path / "meshfiles" / "shp" / "cb_2018_us_nation_20m" / "cb_2018_us_nation_20m.shp"
shp_filename_5poly = current_path / "meshfiles" / "shp" / "5poly/5poly.shp"
shp_filename_multi = current_path / "meshfiles" / "shp" / "multipoly/multipoly.shp"
geojson_filename = current_path / "meshfiles" / "geojson" / "sample_chicago_buildings.geojson"
nc_filename = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"

def test_read_shpfile():
    """Read a shapefile."""
    uxgrid = ux.Grid.from_file(str(shp_filename))
    assert uxgrid.validate()

def test_read_shpfile_multi():
    """Read a shapefile that consists of multipolygons."""
    uxgrid = ux.Grid.from_file(str(shp_filename_multi))
    assert uxgrid.validate()

def test_read_shpfile_5poly():
    """Read a shapefile that consists of 5 polygons of different shapes."""
    uxgrid = ux.Grid.from_file(str(shp_filename_5poly))
    assert uxgrid.validate()

def test_read_geojson():
    """Read a geojson file with a few of Chicago buildings.

    Number of polygons: 10
    Polygon 1: 26 sides
    Polygon 2: 36 sides
    Polygon 3: 29 sides
    Polygon 4: 10 sides
    Polygon 5: 30 sides
    Polygon 6: 8 sides
    Polygon 7: 7 sides
    Polygon 8: 9 sides
    Polygon 9: 7 sides
    Polygon 10: 19 sides
    """
    uxgrid = ux.Grid.from_file(str(geojson_filename))
    assert uxgrid.n_face == 10
    assert uxgrid.n_max_face_nodes == 36

def test_load_xarray_with_from_file():
    """ Use backend xarray to call the from_file method."""
    uxgrid = ux.Grid.from_file(str(nc_filename), backend="xarray")
    uxgrid.validate()

def create_temp_shapefile(geom_type, save_path=None):
    """Helper to create a temporary shapefile with specified geometry type for testing.

    Parameters
    ----------
    geom_type : str
        Type of geometry to create: 'linestring', 'multilinestring', 'point', or 'multipoint'
    save_path : str, optional
        Path to save the temporary shapefile, by default None (creates temp dir)

    Returns
    -------
    str
        Path to the created shapefile
    """
    # Create sample geometries based on type
    geometries = []

    if geom_type == 'linestring':
        # Create a simple line from Chicago to New York
        geometries = [
            LineString([(-87.6298, 41.8781), (-74.0060, 40.7128)]),  # Chicago to New York
            LineString([(-118.2437, 34.0522), (-122.4194, 37.7749)])  # Los Angeles to San Francisco
        ]
    elif geom_type == 'multilinestring':
        # Create a multilinestring representing a route with multiple segments
        geometries = [
            MultiLineString([
                [(-87.6298, 41.8781), (-90.1994, 38.6270)],  # Chicago to St. Louis
                [(-90.1994, 38.6270), (-95.7129, 39.0997)]   # St. Louis to Kansas City
            ]),
            MultiLineString([
                [(-122.4194, 37.7749), (-122.3321, 47.6062)],  # San Francisco to Seattle
                [(-122.3321, 47.6062), (-123.1207, 49.2827)]   # Seattle to Vancouver
            ])
        ]
    elif geom_type == 'point':
        # Create some city points
        geometries = [
            Point(-87.6298, 41.8781),  # Chicago
            Point(-74.0060, 40.7128),  # New York
            Point(-118.2437, 34.0522)  # Los Angeles
        ]
    elif geom_type == 'multipoint':
        # Create multipoints representing clusters of locations
        geometries = [
            MultiPoint([
                (-87.6298, 41.8781),  # Chicago
                (-86.1581, 39.7684)   # Indianapolis
            ]),
            MultiPoint([
                (-74.0060, 40.7128),  # New York
                (-71.0589, 42.3601)   # Boston
            ])
        ]

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")

    # Save to temporary file if no path specified
    if save_path is None:
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, f"test_{geom_type}.shp")

    gdf.to_file(save_path)
    return save_path

def test_read_linestring():
    """Test reading a shapefile with LineString geometries."""
    temp_shp = create_temp_shapefile('linestring')
    try:
        uxgrid = ux.Grid.from_file(temp_shp)
        # Skip validation for non-polygon geometries
        assert uxgrid.n_node > 0
        assert uxgrid.n_edge > 0
        assert uxgrid.n_edge == 2  # We created 2 linestrings
        assert uxgrid.n_face == 0  # LineStrings don't create faces
    finally:
        # Clean up temporary file
        if os.path.exists(temp_shp):
            os.remove(temp_shp)
            # Remove auxiliary files if they exist
            for ext in ['.shx', '.dbf', '.prj', '.cpg', '.lock']:
                aux_file = temp_shp.replace('.shp', ext)
                if os.path.exists(aux_file):
                    os.remove(aux_file)
            # Try to remove directory, but don't fail if not empty
            try:
                os.rmdir(os.path.dirname(temp_shp))
            except OSError:
                pass  # Ignore if directory is not empty

def test_read_multilinestring():
    """Test reading a shapefile with MultiLineString geometries."""
    temp_shp = create_temp_shapefile('multilinestring')
    try:
        uxgrid = ux.Grid.from_file(temp_shp)
        # Skip validation for non-polygon geometries
        assert uxgrid.n_node > 0
        assert uxgrid.n_edge > 0
        assert uxgrid.n_edge == 4  # We created 2 multilinestrings with 2 segments each
        assert uxgrid.n_face == 0  # MultiLineStrings don't create faces
    finally:
        # Clean up temporary file
        if os.path.exists(temp_shp):
            os.remove(temp_shp)
            # Remove auxiliary files if they exist
            for ext in ['.shx', '.dbf', '.prj', '.cpg', '.lock']:
                aux_file = temp_shp.replace('.shp', ext)
                if os.path.exists(aux_file):
                    os.remove(aux_file)
            # Try to remove directory, but don't fail if not empty
            try:
                os.rmdir(os.path.dirname(temp_shp))
            except OSError:
                pass  # Ignore if directory is not empty

def test_read_point():
    """Test reading a shapefile with Point geometries."""
    temp_shp = create_temp_shapefile('point')
    try:
        uxgrid = ux.Grid.from_file(temp_shp)
        # Skip validation for non-polygon geometries
        assert uxgrid.n_node == 3  # We created 3 points
        assert uxgrid.n_edge == 0  # Points don't create edges
        assert uxgrid.n_face == 0  # Points don't create faces
    finally:
        # Clean up temporary file
        if os.path.exists(temp_shp):
            os.remove(temp_shp)
            # Remove auxiliary files if they exist
            for ext in ['.shx', '.dbf', '.prj', '.cpg', '.lock']:
                aux_file = temp_shp.replace('.shp', ext)
                if os.path.exists(aux_file):
                    os.remove(aux_file)
            # Try to remove directory, but don't fail if not empty
            try:
                os.rmdir(os.path.dirname(temp_shp))
            except OSError:
                pass  # Ignore if directory is not empty

def test_read_multipoint():
    """Test reading a shapefile with MultiPoint geometries."""
    temp_shp = create_temp_shapefile('multipoint')
    try:
        uxgrid = ux.Grid.from_file(temp_shp)
        # Skip validation for non-polygon geometries
        assert uxgrid.n_node == 4  # We created 2 multipoints with 2 points each
        assert uxgrid.n_edge == 0  # MultiPoints don't create edges
        assert uxgrid.n_face == 0  # MultiPoints don't create faces
    finally:
        # Clean up temporary file
        if os.path.exists(temp_shp):
            os.remove(temp_shp)
            # Remove auxiliary files if they exist
            for ext in ['.shx', '.dbf', '.prj', '.cpg', '.lock']:
                aux_file = temp_shp.replace('.shp', ext)
                if os.path.exists(aux_file):
                    os.remove(aux_file)
            # Try to remove directory, but don't fail if not empty
            try:
                os.rmdir(os.path.dirname(temp_shp))
            except OSError:
                pass  # Ignore if directory is not empty

def test_mixed_geometry_types():
    """Test reading a file with mixed geometry types using GeoJSON."""
    # Create GeoDataFrame with mixed geometry types
    geometries = [
        Polygon([(-90, 40), (-85, 40), (-85, 45), (-90, 45), (-90, 40)]),  # Simple polygon
        LineString([(-80, 35), (-75, 35)]),  # Simple line
        Point(-70, 30)  # Point
    ]
    gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")

    # Save to temporary GeoJSON file (supports mixed geometry types)
    temp_dir = tempfile.mkdtemp()
    mixed_geojson = os.path.join(temp_dir, "mixed_geom.geojson")
    gdf.to_file(mixed_geojson, driver="GeoJSON")

    try:
        uxgrid = ux.Grid.from_file(mixed_geojson)
        # Skip validation for mixed geometries
        assert uxgrid.n_node > 0
        # Verify all geometries are present
        assert uxgrid.n_edge == 1  # One LineString
        assert uxgrid.n_face == 1  # One Polygon
        # One Point should be processed as a node
    finally:
        # Clean up
        if os.path.exists(mixed_geojson):
            os.remove(mixed_geojson)
            # Try to remove directory, but don't fail if not empty
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass  # Ignore if directory is not empty
