import os
from pathlib import Path

import uxarray as ux
from .helpers._memsize import dataset_nbytes, grid_nbytes

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]

grid_path = current_path / "test" / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"
data_path = current_path / "test" / "meshfiles" / "ugrid" / "quad-hexagon" / "data.nc"


class QuadHexagon:
    def time_open_grid(self):
        """Time to open a `Grid`"""
        ux.open_grid(grid_path)

    def track_nbytes_open_grid(self):
        """Memory occupied by a `Grid`"""
        return grid_nbytes(ux.open_grid(grid_path))

    track_nbytes_open_grid.unit = "bytes"

    def time_open_dataset(self):
        """Time to open a `UxDataset`"""
        ux.open_dataset(grid_path, data_path)

    def track_nbytes_open_dataset(self):
        """Memory occupied by a `UxDataset`, including its grid"""
        return dataset_nbytes(ux.open_dataset(grid_path, data_path))

    track_nbytes_open_dataset.unit = "bytes"
