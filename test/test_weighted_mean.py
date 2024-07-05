import uxarray as ux
import os

import numpy.testing as nt

import pytest

from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


csne30_grid_path = current_path / 'meshfiles' / "ugrid" / "outCSne30" / "outCSne30.ug"
csne30_data_path = current_path / 'meshfiles' / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"

quad_hex_grid_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / "grid.nc"
quad_hex_data_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / "data.nc"


def test_quad_hex():
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)

    # create an array of expected values here (can compute by hand)
    expected = ...

    result = uxds['t2m'].mean(weighted=True)

    # make sure the results are almost equal
    # nt.assert_almost_equal(result.values, expected)


def test_csne30():
    uxds = ux.open_dataset(csne30_grid_path, csne30_data_path)

    # add test here for weighted_mean

    result = uxds['psi'].mean(weighted=True)
