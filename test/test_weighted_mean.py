import uxarray as ux
import os

import pytest

from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


grid_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'grid.nc'
data_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'data.nc'



def test_weighted_mean_faces():
    uxds = ux.open_dataset(grid_path, data_path)

    res = uxds['t2m'].weighted_mean()
