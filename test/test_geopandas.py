import os
import numpy as np
from pathlib import Path
import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

shp_filename = current_path / "meshfiles" / "shp" / "cb_2018_us_nation_20m" / "cb_2018_us_nation_20m.shp"
shp_filename_5poly = current_path / "meshfiles" / "shp" / "5poly/5poly.shp"
shp_filename_multi = current_path / "meshfiles" / "shp" / "multipoly/multipoly.shp"
geojson_filename = current_path / "meshfiles" / "geojson" / "sample_chicago_buildings.geojson"
nc_filename = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"

def test_read_shpfile():
    """Read a shapefile."""
    uxgrid = ux.Grid.from_file(shp_filename)
    assert uxgrid.validate()

def test_read_shpfile_multi():
    """Read a shapefile that consists of multipolygons."""
    uxgrid = ux.Grid.from_file(shp_filename_multi)
    assert uxgrid.validate()

def test_read_shpfile_5poly():
    """Read a shapefile that consists of 5 polygons of different shapes."""
    uxgrid = ux.Grid.from_file(shp_filename_5poly)
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
    uxgrid = ux.Grid.from_file(geojson_filename)
    assert uxgrid.n_face == 10
    assert uxgrid.n_max_face_nodes == 36

def test_load_xarray_with_from_file():
    """ Use backend xarray to call the from_file method."""
    uxgrid = ux.Grid.from_file(nc_filename, backend="xarray")
    uxgrid.validate()
