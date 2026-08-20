import numpy as np
import uxarray as ux

def test_read_shpfile(test_data_dir):
    """Read a shapefile."""
    uxgrid = ux.Grid.from_file(str(test_data_dir / "shp" / "cb_2018_us_nation_20m" / "cb_2018_us_nation_20m.shp"))
    assert uxgrid.validate()

def test_read_shpfile_multi(test_data_dir):
    """Read a shapefile that consists of multipolygons."""
    uxgrid = ux.Grid.from_file(str(test_data_dir / "shp" / "multipoly" / "multipoly.shp"))
    assert uxgrid.validate()

def test_read_shpfile_5poly(test_data_dir):
    """Read a shapefile that consists of 5 polygons of different shapes."""
    uxgrid = ux.Grid.from_file(str(test_data_dir / "shp" / "5poly" / "5poly.shp"))
    assert uxgrid.validate()

def test_read_geojson(test_data_dir):
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
    uxgrid = ux.Grid.from_file(str(test_data_dir / "geojson" / "sample_chicago_buildings.geojson"))
    assert uxgrid.n_face == 10
    assert uxgrid.n_max_face_nodes == 36

def test_load_xarray_with_from_file(gridpath):
    """ Use backend xarray to call the from_file method."""
    nc_filename = gridpath("scrip", "outCSne8", "outCSne8.nc")
    uxgrid = ux.Grid.from_file(nc_filename, backend="xarray")
    uxgrid.validate()


def test_read_failure_raises(tmp_path):
    """A read failure must surface the backend's own error rather than being
    printed and swallowed into an UnboundLocalError.

    Regression test for issue #1693. Requires a geopandas file-IO backend,
    otherwise read_file raises ImportError and the test would pass for the
    wrong reason.
    """
    import pytest

    pytest.importorskip("pyogrio")

    from uxarray.io._geopandas import _gpd_read

    not_geospatial = tmp_path / "not_geospatial.shp"
    not_geospatial.write_text("this is not a shapefile")

    with pytest.raises(Exception) as excinfo:
        _gpd_read(str(not_geospatial))

    assert not isinstance(excinfo.value, UnboundLocalError)
