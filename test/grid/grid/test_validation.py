import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE


def test_grid_validate(gridpath):
    """Test to check the validate function."""
    grid_mpas = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    assert grid_mpas.validate()
