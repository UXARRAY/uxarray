import uxarray as ux
import cartopy.crs as ccrs
import os
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_geos_cs = current_path / "meshfiles" / "geos-cs" / "c12" / "test-c12.native.nc4"

def test_geodataframe_projection():
    """Test the projection of a GeoDataFrame."""
    uxgrid = ux.open_grid(gridfile_geos_cs)
    gdf = uxgrid.to_geodataframe(projection=ccrs.Robinson(), periodic_elements='exclude')

    # Example assertion to check if gdf is not None
    assert gdf is not None
