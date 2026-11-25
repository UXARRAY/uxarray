import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

import uxarray as ux
from uxarray.io._geopandas import _extract_geometry_info, _gpd_read, _set_crs

def test_read_shpfile(test_data_dir):
    """Read a shapefile."""
    uxgrid = ux.Grid.from_file(str(test_data_dir / "shp" / "cb_2018_us_nation_20m" / "cb_2018_us_nation_20m.shp"))
    assert uxgrid.validate()

    # tiny regression test for #1754: since this file has just the U.S., expect all latitudes>17.
    assert np.all(uxgrid.node_lat.values > 17)

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

    Regression test for issue #1693.
    """
    not_geospatial = tmp_path / "not_geospatial.shp"
    not_geospatial.write_text("this is not a shapefile")

    with pytest.raises(Exception) as excinfo:
        _gpd_read(str(not_geospatial))

    assert not isinstance(excinfo.value, UnboundLocalError)


def test_set_crs_warns_when_crs_is_missing():
    """Assuming WGS84 for CRS-less data is a guess and must be announced."""
    gdf = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs=None
    )

    with pytest.warns(UserWarning, match="no CRS"):
        out = _set_crs(gdf)

    assert out.crs is not None


def test_read_warnings_name_the_file_and_blame_the_caller(tmp_path):
    """A warning about a file has to say which file, and point at the line the
    user wrote rather than at whichever uxarray helper happens to raise it.

    A hardcoded ``stacklevel`` gets the second half wrong: from ``_set_crs`` it
    lands on ``_gpd_read``, so the traceback stops inside ``_geopandas.py``.
    """
    no_crs = tmp_path / "no_crs.shp"
    gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])], crs=None
    ).to_file(no_crs)

    mixed = tmp_path / "mixed.geojson"
    gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]), Point(5, 5)],
        crs="EPSG:4326",
    ).to_file(mixed, driver="GeoJSON")

    for path, match in ((no_crs, "no CRS"), (mixed, "unsupported geometry type")):
        with pytest.warns(UserWarning, match=match) as record:
            ux.Grid.from_file(str(path), backend="geopandas")

        warning = [w for w in record if match in str(w.message)][0]
        assert path.name in str(warning.message)
        # Attributed to this test, not to uxarray/io/_geopandas.py.
        assert warning.filename == __file__


def test_unsupported_geometry_is_reported():
    """Dropping a geometry silently would yield a grid missing a face with no
    indication that anything was skipped."""
    gdf = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]), Point(5, 5)],
        crs="EPSG:4326",
    )

    with pytest.warns(UserWarning, match="unsupported geometry type"):
        node_lon, node_lat, connectivity = _extract_geometry_info(gdf, 4)

    # Only the polygon contributes a face; the point is skipped.
    assert connectivity.shape[0] == 1


def test_multipolygon_preserves_lon_lat():
    """Every part of a MultiPolygon must keep its lon/lat order.

    Regression test for issue #1754: _read_multipolygon unpacked
    _read_polygon's (lat, lon) return as (lon, lat), so the lists traded
    places after each part and every other part was read with lon/lat swapped.
    """
    # Longitudes are well outside [-90, 90], so a swap can't be mistaken for valid data.
    parts = [
        Polygon([(100, 10), (110, 10), (110, 20), (100, 20)]),
        Polygon([(120, 30), (130, 30), (130, 40), (120, 40)]),
        Polygon([(140, 50), (150, 50), (150, 60)]),
    ]
    gdf = gpd.GeoDataFrame(geometry=[MultiPolygon(parts)], crs="EPSG:4326")
    max_nodes = max(len(p.exterior.coords) for p in parts)

    node_lon, node_lat, connectivity = _extract_geometry_info(gdf, max_nodes)

    expected = np.concatenate([np.asarray(p.exterior.coords)[:-1] for p in parts])
    np.testing.assert_array_equal(node_lon, expected[:, 0])
    np.testing.assert_array_equal(node_lat, expected[:, 1])
    assert connectivity.shape[0] == len(parts)
