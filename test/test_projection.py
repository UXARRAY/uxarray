import uxarray as ux
import cartopy.crs as ccrs

import os
from pathlib import Path
from unittest import TestCase

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_geos_cs = current_path / "meshfiles" / "geos-cs" / "c12" / "test-c12.native.nc4"


class TestProjection(TestCase):
    def test_geodataframe_projection(self):
        uxgrid = ux.open_grid(gridfile_geos_cs)

        gdf = uxgrid.to_geodataframe(projection=ccrs.Robinson(), periodic_elements='exclude')
