import uxarray as ux
import numpy as np
import pytest
from uxarray.constants import ERROR_TOLERANCE
import os
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestZonalCSne30:
    gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    datafile_vortex_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"
    dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"
    test_file_2 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_test2.nc"
    test_file_3 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_test3.nc"


    @pytest.mark.parametrize("use_spherical_bounding_box", [True, False])
    def test_non_conservative_zonal_mean_equator(self, use_spherical_bounding_box):
        """Test _non_conservative_zonal_mean function with outCSne30 data.

        Low error tolerance test at the equator.
        """
        # Create test data
        grid_path = self.gridfile_ne30
        data_path = self.datafile_vortex_ne30
        uxds = ux.open_dataset(grid_path, data_path)

        # Test everything away from the pole
        res = uxds['psi'].zonal_mean(0, use_spherical_bounding_box=use_spherical_bounding_box)

        # Assert res.values[0] is approximately 1 within ERROR_TOLERANCE
        assert res.values[0] == pytest.approx(1, abs=ERROR_TOLERANCE)

    @pytest.mark.parametrize("use_spherical_bounding_box", [True, False])
    def test_non_conservative_zonal_mean(self, use_spherical_bounding_box):
        """Test _non_conservative_zonal_mean function with outCSne30 data.

        Dummy test to ensure the function runs from -90 to 90 with a step of 1.
        """
        # Create test data
        grid_path = self.gridfile_ne30
        data_path = self.datafile_vortex_ne30
        uxds = ux.open_dataset(grid_path, data_path)

        res = uxds['psi'].zonal_mean((-90, 90, 1), use_spherical_bounding_box=use_spherical_bounding_box)
        print(res)

    @pytest.mark.parametrize("use_spherical_bounding_box", [True, False])
    def test_non_conservative_zonal_mean_at_pole(self, use_spherical_bounding_box):
        """Test _non_conservative_zonal_mean function with outCSne30 data.

        Dummy test to ensure the function runs at the pole.
        """
        # Create test data
        grid_path = self.gridfile_ne30
        data_path = self.datafile_vortex_ne30
        uxds = ux.open_dataset(grid_path, data_path)

        # Test at the poles
        res_n90 = uxds['psi'].zonal_mean(90, use_spherical_bounding_box=use_spherical_bounding_box)
        res_p90 = uxds['psi'].zonal_mean(-90, use_spherical_bounding_box=use_spherical_bounding_box)

        # Assert results are approximately 1 within a delta of 1
        assert res_n90.values[0] == pytest.approx(1, abs=1)
        assert res_p90.values[0] == pytest.approx(1, abs=1)
