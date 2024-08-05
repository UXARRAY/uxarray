import os
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]

grid_path = current_path / "test" / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
data_path = current_path / "test" / "meshfiles" / "ugrid" / "outCSne30" / "relhum.nc"


class ZonalMean:
    def setup(self):
        self.uxds = ux.open_dataset(grid_path, data_path)

    def teardown(self):
        del self.uxds


    def time_zonal_mean(self):
        self.uxds['relhum'].zonal_mean()
