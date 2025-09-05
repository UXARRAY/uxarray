import uxarray as ux
import os
import pytest
from pathlib import Path

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *

grid_path = ICON_R02B04_GRID

def test_read_icon_grid():
    uxgrid = ux.open_grid(grid_path)

def test_read_icon_dataset():
    uxds = ux.open_dataset(grid_path, grid_path)
