import warnings

import geopandas as gpd
from shapely.geometry import Polygon

import uxarray as ux
from uxarray.io._geopandas import _gpd_read, _set_crs
from uxarray.utils.warnings import find_stack_level


def test_find_stack_level_is_independent_of_caller_depth():
    """The level must name the caller's frame however many of its own frames
    sit between it and the ``warnings.warn`` call."""

    def one_frame_deep():
        warnings.warn("boom", stacklevel=find_stack_level())

    def two_frames_deep():
        one_frame_deep()

    for call in (one_frame_deep, two_frames_deep):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        # Both helpers live outside uxarray, so the warning belongs to whichever
        # of them called ``warn`` -- this file either way.
        assert caught[0].filename == __file__


def test_find_stack_level_skips_however_many_internal_frames(tmp_path):
    """Entering the reader at different depths must not move the blame.

    ``_set_crs`` raises the warning with one uxarray frame on the stack when
    called directly, two through ``_gpd_read``, and four through
    ``Grid.from_file``. A hardcoded ``stacklevel`` can only be right for one of
    the three.
    """
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])

    no_crs = tmp_path / "no_crs.shp"
    gpd.GeoDataFrame(geometry=[polygon], crs=None).to_file(no_crs)

    entry_points = (
        lambda: _set_crs(gpd.GeoDataFrame(geometry=[polygon], crs=None)),
        lambda: _gpd_read(str(no_crs)),
        lambda: ux.Grid.from_file(str(no_crs), backend="geopandas"),
    )

    for call in entry_points:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        crs_warnings = [w for w in caught if "no CRS" in str(w.message)]
        assert len(crs_warnings) == 1
        assert crs_warnings[0].filename == __file__
