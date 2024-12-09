import uxarray as ux
import os

import pytest

from pathlib import Path
from unittest import TestCase

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


grid_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'grid.nc'
data_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'data.nc'


class TestRepr(TestCase):
    def test_grid_repr(self):
        uxgrid = ux.open_grid(grid_path)

        out = uxgrid._repr_html_()

        assert out is not None


    def test_dataset_repr(self):
        uxds = ux.open_dataset(grid_path, data_path)

        out = uxds._repr_html_()

        assert out is not None


    def test_dataarray_repr(self):
        uxds = ux.open_dataset(grid_path, data_path)

        out = uxds['t2m']._repr_html_()

        assert out is not None
