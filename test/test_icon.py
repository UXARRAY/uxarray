import uxarray as ux
import os
import pytest
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

grid_path = current_path / 'meshfiles' / "icon" / "R02B04" / 'icon_grid_0010_R02B04_G.nc'

def test_read_icon_grid():
    uxgrid = ux.open_grid(grid_path)

def test_read_icon_dataset():
    uxds = ux.open_dataset(grid_path, grid_path)
