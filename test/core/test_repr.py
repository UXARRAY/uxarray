import uxarray as ux
import os

import pytest

from pathlib import Path

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *






def test_grid_repr():
    uxgrid = ux.open_grid(QUAD_HEXAGON_GRID)

    out = uxgrid._repr_html_()

    assert out is not None


def test_dataset_repr():
    uxds = ux.open_dataset(QUAD_HEXAGON_GRID, QUAD_HEXAGON_DATA)

    out = uxds._repr_html_()

    assert out is not None


def test_dataarray_repr():
    uxds = ux.open_dataset(QUAD_HEXAGON_GRID, QUAD_HEXAGON_DATA)

    out = uxds['t2m']._repr_html_()

    assert out is not None
