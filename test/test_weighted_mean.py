import uxarray as ux
import os

import pytest

from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


csne30_grid_path = current_path / 'meshfiles' / "ugrid" / "outCSne30" / "outCSne30.ug"
csne30_data_path = current_path / 'meshfiles' / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"


def test_weighted_mean():
    uxds = ux.open_dataset(csne30_grid_path, csne30_data_path)

    # add test here for weighted_mean

    res = uxds['psi'].mean(weighted=True)
