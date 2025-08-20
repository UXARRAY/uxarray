import os
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parent

gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"


def test_engine_geodataframe():
    uxgrid = ux.open_grid(gridfile_geoflow)
    for engine in ['geopandas', 'spatialpandas']:
        gdf = uxgrid.to_geodataframe(engine=engine)


def test_periodic_elements_geodataframe():
    uxgrid = ux.open_grid(gridfile_geoflow)
    for periodic_elements in ['ignore', 'exclude', 'split']:
        gdf = uxgrid.to_geodataframe(periodic_elements=periodic_elements)


def test_to_gdf_geodataframe():
    uxgrid = ux.open_grid(gridfile_geoflow)

    gdf_with_am = uxgrid.to_geodataframe(exclude_antimeridian=False)

    gdf_without_am = uxgrid.to_geodataframe(exclude_antimeridian=True)


def test_cache_and_override_geodataframe():
    """Tests the cache and override functionality for GeoDataFrame conversion."""
    uxgrid = ux.open_grid(gridfile_geoflow)

    gdf_a = uxgrid.to_geodataframe(exclude_antimeridian=False)

    gdf_b = uxgrid.to_geodataframe(exclude_antimeridian=False)

    assert gdf_a is gdf_b

    gdf_c = uxgrid.to_geodataframe(exclude_antimeridian=True)
