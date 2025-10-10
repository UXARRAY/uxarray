
import dask.array as da
import numpy as np
import pytest

import numpy.testing as nt

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE

class TestZonalCSne30:

    gridfile_ne30 = None  # Will be set in test methods
    datafile_vortex_ne30 = None  # Will be set in test methods
    dsfile_var2_ne30 = None  # Will be set in test methods
    test_file_2 = None  # Will be set in test methods
    test_file_3 = None  # Will be set in test methods

    def test_non_conservative_zonal_mean_equator(self, gridpath, datasetpath):
        """Tests the zonal mean at the equator. This grid contains points that are exactly """
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        res = uxds['psi'].zonal_mean(0)

        assert res.values[0] == pytest.approx(1, abs=ERROR_TOLERANCE)

    def test_non_conservative_zonal_mean(self, gridpath, datasetpath):
        """Tests if the correct number of queries are returned."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        res = uxds['psi'].zonal_mean((-90.0, 90.0, 1))

        assert len(res) == 181

    def test_non_conservative_zonal_mean_at_pole(self, gridpath, datasetpath):
        """Tests the zonal average execution at both poles."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Test at the poles
        res_n90 = uxds['psi'].zonal_mean(90)
        res_p90 = uxds['psi'].zonal_mean(-90)

        # result should be a scalar
        assert len(res_n90.values) == 1
        assert len(res_p90.values) == 1

    def test_zonal_mean_dask(self, gridpath, datasetpath):
        """Tests if zonal average returns Dask arrays when appropriate."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        uxds['psi'] = uxds['psi'].chunk()

        res = uxds['psi'].zonal_mean((-90, 90, 1))

        assert isinstance(res.data, da.Array)

        res_computed = res.compute()

        assert isinstance(res_computed.data, np.ndarray)

    def test_zonal_weights(self, gridpath, datasetpath):
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        za_1 = uxds['psi'].zonal_mean((-90, 90, 30), use_robust_weights=True)
        za_2 = uxds['psi'].zonal_mean((-90, 90, 30), use_robust_weights=False)

        nt.assert_almost_equal(za_1.data, za_2.data)

    def test_lat_inputs(self, gridpath, datasetpath):
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        assert len(uxds['psi'].zonal_mean(lat=1)) == 1
        assert len(uxds['psi'].zonal_mean(lat=(-90, 90, 1))) == 181

def test_mismatched_dims():
    uxgrid = ux.Grid.from_healpix(zoom=0)
    uxda = ux.UxDataArray(np.ones((10, uxgrid.n_face, 5)), dims=['a', 'n_face', 'b'], uxgrid=uxgrid)

    za = uxda.zonal_average()

    assert za.shape == (10, 19, 5)
    assert za.dims[1] == "latitudes"


class TestConservativeZonalMean:
    """Test conservative zonal mean functionality."""

    def test_conservative_zonal_mean_basic(self, gridpath, datasetpath):
        """Test basic conservative zonal mean with bands."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Test with explicit bands
        bands = np.array([-90, -30, 0, 30, 90])
        result = uxds["psi"].zonal_mean(lat=bands, conservative=True)

        # Should have one less value than bands (4 bands from 5 edges)
        assert result.shape == (len(bands) - 1,)
        assert np.all(np.isfinite(result.values))

    def test_conservative_float_step_size(self, gridpath, datasetpath):
        """Test conservative zonal mean with float step sizes."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Test with float step size (5.5 degrees)
        result = uxds["psi"].zonal_mean(lat=(-90, 90, 0.05), conservative=True)

        # Should get valid results
        assert len(result) > 0
        assert np.all(np.isfinite(result.values))

        # Test with reasonable float step size (no warning)
        result = uxds["psi"].zonal_mean(lat=(-90, 90, 5.5), conservative=True)
        expected_n_bands = int(np.ceil(180 / 5.5))
        assert result.shape[0] == expected_n_bands
        assert np.all(np.isfinite(result.values))

    def test_conservative_near_pole(self, gridpath, datasetpath):
        """Test conservative zonal mean with bands near the poles."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Test near north pole with float step
        bands_north = np.array([85.0, 87.5, 90.0])
        result_north = uxds["psi"].zonal_mean(lat=bands_north, conservative=True)
        assert result_north.shape == (2,)
        assert np.all(np.isfinite(result_north.values))

        # Test near south pole with float step
        bands_south = np.array([-90.0, -87.5, -85.0])
        result_south = uxds["psi"].zonal_mean(lat=bands_south, conservative=True)
        assert result_south.shape == (2,)
        assert np.all(np.isfinite(result_south.values))

        # Test spanning pole with non-integer step
        bands_span = np.array([88.5, 89.25, 90.0])
        result_span = uxds["psi"].zonal_mean(lat=bands_span, conservative=True)
        assert result_span.shape == (2,)
        assert np.all(np.isfinite(result_span.values))

    def test_conservative_step_size_validation(self, gridpath, datasetpath):
        """Test that step size validation works correctly."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Test negative step size
        with pytest.raises(ValueError, match="Step size must be positive"):
            uxds["psi"].zonal_mean(lat=(-90, 90, -10), conservative=True)

        # Test zero step size
        with pytest.raises(ValueError, match="Step size must be positive"):
            uxds["psi"].zonal_mean(lat=(-90, 90, 0), conservative=True)

    def test_conservative_full_sphere_conservation(self, gridpath, datasetpath):
        """Test that single band covering entire sphere conserves global mean."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Single band covering entire sphere
        bands = np.array([-90, 90])
        result = uxds["psi"].zonal_mean(lat=bands, conservative=True)

        # Compare with global mean
        global_mean = uxds["psi"].mean()

        assert result.shape == (1,)
        assert result.values[0] == pytest.approx(global_mean.values, rel=0.01)

    def test_conservative_vs_nonconservative_comparison(self, gridpath, datasetpath):
        """Compare conservative and non-conservative methods."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Non-conservative at band centers
        lat_centers = np.array([-60, 0, 60])
        non_conservative = uxds["psi"].zonal_mean(lat=lat_centers)

        # Conservative with bands
        bands = np.array([-90, -30, 30, 90])
        conservative = uxds["psi"].zonal_mean(lat=bands, conservative=True)

        # Results should be similar but not identical
        assert non_conservative.shape == conservative.shape
        # Check they are in the same ballpark
        assert np.all(np.abs(conservative.values - non_conservative.values) <
                     np.abs(non_conservative.values) * 0.5)
