import uxarray as ux
import os
import pytest
from pathlib import Path

def test_read_icon_grid(gridpath):
    grid_path = gridpath("icon", "R02B04", "icon_grid_0010_R02B04_G.nc")
    uxgrid = ux.open_grid(grid_path)

def test_read_icon_dataset(gridpath):
    grid_path = gridpath("icon", "R02B04", "icon_grid_0010_R02B04_G.nc")
    uxds = ux.open_dataset(grid_path, grid_path)
