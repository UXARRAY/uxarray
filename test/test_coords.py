import uxarray as ux

from uxarray.constants import INT_FILL_VALUE
import numpy.testing as nt
import os

import pytest

from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

GRID_PATHS = [
    current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc",
    current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
]




def test_lonlat_to_xyz():
    from uxarray.grid.veccoordinates import _populate_node_xyz

    for grid_path in GRID_PATHS:
        uxgrid = ux.open_grid(grid_path)

        _populate_node_xyz(uxgrid)


def test_xyz_to_lonlat():
    pass
