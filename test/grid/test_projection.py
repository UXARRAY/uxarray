import uxarray as ux
import cartopy.crs as ccrs
import os
from pathlib import Path

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *

gridfile_geos_cs = GEOS_CS_C12_GRID

def test_geodataframe_projection():
    """Test the projection of a GeoDataFrame."""
    uxgrid = ux.open_grid(gridfile_geos_cs)
    gdf = uxgrid.to_geodataframe(projection=ccrs.Robinson(), periodic_elements='exclude')

    # Example assertion to check if gdf is not None
    assert gdf is not None
