import os
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]

grid_path = current_path / "test" / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"
data_path = current_path / "test" / "meshfiles" / "ugrid" / "quad-hexagon" / "data.nc"


class QuadHexagon:
    def time_open_grid(self):
        """Time to open a `Grid`"""
        ux.open_grid(grid_path)

    def mem_open_grid(self):
        """Memory Occupied by a `Grid`"""
        return ux.open_grid(grid_path)

    def peakmem_open_grid(self):
        """Peak memory usage of a `Grid`"""
        uxgrid = ux.open_grid(grid_path)


    def time_open_dataset(self):
        """Time to open a `UxDataset`"""
        ux.open_dataset(grid_path, data_path)

    def mem_open_dataset(self):
        """Memory occupied by a `UxDataset`"""
        return ux.open_dataset(grid_path, data_path)

    def peakmem_open_dataset(self):
        """Peak memory usage of a `UxDataset`"""
        uxds = ux.open_dataset(grid_path, data_path)
