import os
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

grid_path = current_path / "test" / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"
data_path = current_path / "test" / "meshfiles" / "ugrid" / "quad-hexagon" / "data.nc"


class QuadHexagon:
    def time_open_grid(self):
        ux.open_grid(grid_path)

    def mem_open_grid(self):
        return ux.open_grid(grid_path)

    def peakmem_open_grid(self):
        uxgrid = ux.open_grid(grid_path)


QuadHexagon()
