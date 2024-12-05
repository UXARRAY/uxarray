import os
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from unittest import TestCase
import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestZonalCSne30(TestCase):
    gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    datafile_vortex_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"
    dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"
    test_file_2 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_test2.nc"
    test_file_3 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_test3.nc"

    def test_non_conservative_zonal_mean_equator(self):
        """Tests the zonal mean at the equator. This grid contains points that are exactly """
        grid_path = self.gridfile_ne30
        data_path = self.datafile_vortex_ne30
        uxds = ux.open_dataset(grid_path, data_path)

        res = uxds['psi'].zonal_mean(0)

        assert res.values[0] == pytest.approx(1, abs=ERROR_TOLERANCE)

    def test_non_conservative_zonal_mean(self):
        """Tests if the correct number of queries are returned."""
        grid_path = self.gridfile_ne30
        data_path = self.datafile_vortex_ne30
        uxds = ux.open_dataset(grid_path, data_path)

        res = uxds['psi'].zonal_mean((-90, 90, 1))

        assert len(res) == 181

    def test_non_conservative_zonal_mean_at_pole(self):
        """Tests the zonal average at both poles."""
        grid_path = self.gridfile_ne30
        data_path = self.datafile_vortex_ne30
        uxds = ux.open_dataset(grid_path, data_path)

        # Test at the poles
        res_n90 = uxds['psi'].zonal_mean(90)
        res_p90 = uxds['psi'].zonal_mean(-90)

        # Assert results are approximately 1 within a delta of 1
        assert res_n90.values[0] == pytest.approx(1, abs=1)
        assert res_p90.values[0] == pytest.approx(1, abs=1)

    def test_zonal_mean_dask(self):
        """Tests if zonal average returns Dask arrays when appropriate."""
        grid_path = self.gridfile_ne30
        data_path = self.datafile_vortex_ne30
        uxds = ux.open_dataset(grid_path, data_path)

        uxds['psi'] = uxds['psi'].chunk()

        res = uxds['psi'].zonal_mean((-90, 90, 1))

        assert isinstance(res.data, da.Array)

        res_computed = res.compute()

        assert isinstance(res_computed.data, np.ndarray)
