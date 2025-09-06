import os
import numpy as np
from pathlib import Path
import uxarray as ux

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *


nc_filename = SCRIP_OUTCSNE8

def test_read_shpfile():
    """Read a shapefile."""
    uxgrid = ux.Grid.from_file(SHP_US_NATION_FILE)
    assert uxgrid.validate()

def test_read_shpfile_multi():
    """Read a shapefile that consists of multipolygons."""
    uxgrid = ux.Grid.from_file(SHP_MULTIPOLY_FILE)
    assert uxgrid.validate()

def test_read_shpfile_5poly():
    """Read a shapefile that consists of 5 polygons of different shapes."""
    uxgrid = ux.Grid.from_file(SHP_5POLY_FILE)
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
    uxgrid = ux.Grid.from_file(GEOJSON_CHICAGO_BUILDINGS)
    assert uxgrid.n_face == 10
    assert uxgrid.n_max_face_nodes == 36

def test_load_xarray_with_from_file():
    """ Use backend xarray to call the from_file method."""
    uxgrid = ux.Grid.from_file(nc_filename, backend="xarray")
    uxgrid.validate()
