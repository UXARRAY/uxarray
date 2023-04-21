import os
from pathlib import Path
from unittest import TestCase

import xarray as xr
import uxarray as ux

from uxarray.plot.helpers import compute_antimeridian_crossing

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestPlot(TestCase):
    ug_filename1 = current_path / "meshfiles" / 'ugrid' / "geoflow-small" / "grid.nc"
    grid1_ds = xr.open_dataset(ug_filename1)
    grid1 = ux.Grid(grid1_ds)

    def test_compute_antimeridian_crossing(self):
        compute_antimeridian_crossing(self.grid1)
        pass
