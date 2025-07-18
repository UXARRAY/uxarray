import os
from pathlib import Path
import numpy.testing as nt
import xarray as xr
import uxarray as ux
from uxarray import UxDataset
import pytest

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"
gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
dsfile_v1_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"
mpas_ds_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'

def test_uxgrid_setget():
    """Load a dataset with its grid topology file using uxarray's
    open_dataset call and check its grid object."""
    uxds_var2_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)
    uxgrid_var2_ne30 = ux.open_grid(gridfile_ne30)
    assert (uxds_var2_ne30.uxgrid == uxgrid_var2_ne30)

def test_integrate():
    """Load a dataset and calculate integrate()."""
    uxds_var2_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)
    integrate_var2 = uxds_var2_ne30.integrate()
    nt.assert_almost_equal(integrate_var2, constants.VAR2_INTG, decimal=3)

def test_info():
    """Tests custom info containing grid information."""
    uxds_var2_geoflow = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        try:
            uxds_var2_geoflow.info(show_attrs=True)
        except Exception as exc:
            assert False, f"'uxds_var2_geoflow.info()' raised an exception: {exc}"

def test_ugrid_dim_names():
    """Tests the remapping of dimensions to the UGRID conventions."""
    ugrid_dims = ["n_face", "n_node", "n_edge"]
    uxds_remap = ux.open_dataset(mpas_ds_path, mpas_ds_path)

    for dim in ugrid_dims:
        assert dim in uxds_remap.dims

def test_get_dual():
    """Tests the creation of the dual mesh on a data set."""
    uxds = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)
    dual = uxds.get_dual()

    assert isinstance(dual, UxDataset)
    assert len(uxds.data_vars) == len(dual.data_vars)


def test_groupby_preserves_uxgrid():
    """Test that groupby operations preserve the uxgrid attribute."""
    # Create a dataset from a file
    uxds = ux.open_dataset(mpas_ds_path, mpas_ds_path)
    original_grid = uxds.uxgrid

    # Create bins from latitude values (extract data explicitly)
    lat_bins = (uxds.latCell > 0).astype(int).values

    # Add the bins as a coordinate
    uxds = uxds.assign_coords({"lat_bins": ("n_face", lat_bins)})

    # Test DataArray groupby preserves uxgrid
    da_result = uxds.latCell.groupby(uxds.lat_bins).mean()
    assert hasattr(da_result, "uxgrid")
    assert da_result.uxgrid is not None

    # Test Dataset groupby preserves uxgrid
    ds_result = uxds.groupby(uxds.lat_bins).mean()
    assert hasattr(ds_result, "uxgrid")
    assert ds_result.uxgrid is not None
    assert ds_result.uxgrid == original_grid




def test_resample_preserves_uxgrid_and_reduces_time():
    """Test that resample operations preserve uxgrid and reduce time dimension."""
    import numpy as np
    import pandas as pd
    import pytest
    import xarray as xr

    # Create a simple test with only time dimension
    times = pd.date_range("2000-01-01", periods=12, freq="D")
    temp_data = np.random.rand(12)

    # Create a simple xarray Dataset
    xr_ds = xr.Dataset(
        {"temperature": ("time", temp_data)},
        coords={"time": times}
    )

    # Open the minimal dataset with a real grid
    try:
        # Use existing test file we know works
        uxgrid = ux.open_grid(gridfile_ne30)

        # Create a UxDataset with this grid
        uxds = ux.UxDataset(xr_ds, uxgrid=uxgrid)

        print(f"Original dataset dims: {uxds.dims}")
        print(f"Original dataset shape: {uxds.temperature.shape}")

        # Test the resample method directly
        print("Attempting resample...")
        result = uxds.temperature.resample(time="1W").mean()

        print(f"Resampled result dims: {result.dims}")
        print(f"Resampled result shape: {result.shape}")

        # Test assertions
        assert hasattr(result, "uxgrid"), "uxgrid not preserved on resample"
        assert result.uxgrid == uxds.uxgrid, "uxgrid not equal after resample"
        assert len(result.time) < len(uxds.time), "time dimension not reduced"

    except Exception as e:
        import traceback
        traceback.print_exc()
        pytest.fail(f"Error in resample test: {e}")

def test_resample_preserves_uxgrid():
    """Test that resample preserves the uxgrid attribute."""
    import numpy as np
    import pandas as pd
    import pytest

    # Create a simple dataset with a time dimension
    times = pd.date_range("2000-01-01", periods=12, freq="D")
    data = np.random.rand(12)

    # Create a simple xarray Dataset
    ds = xr.Dataset(
        {"temperature": ("time", data)},
        coords={"time": times}
    )

    # Create a UxDataset with a real grid
    uxds = ux.open_dataset(gridfile_ne30, gridfile_ne30)
    original_uxgrid = uxds.uxgrid

    # Create a new UxDataset with our time data and the real grid
    uxds_time = ux.UxDataset(ds, uxgrid=original_uxgrid)

    # Test DataArray resample preserves uxgrid
    da_result = uxds_time.temperature.resample(time="1W").mean()
    assert hasattr(da_result, "uxgrid"), "uxgrid not preserved on DataArray resample"
    assert da_result.uxgrid is original_uxgrid, "uxgrid not identical after DataArray resample"

    # Test Dataset resample preserves uxgrid
    ds_result = uxds_time.resample(time="1W").mean()
    assert hasattr(ds_result, "uxgrid"), "uxgrid not preserved on Dataset resample"
    assert ds_result.uxgrid is original_uxgrid, "uxgrid not identical after Dataset resample"


def test_resample_reduces_time_dimension():
    """Test that resample properly reduces the time dimension."""
    import numpy as np
    import pandas as pd
    import pytest

    # Create dataset with daily data for a year
    times = pd.date_range("2000-01-01", periods=365, freq="D")
    data = np.random.rand(365)

    # Create a simple xarray Dataset
    ds = xr.Dataset(
        {"temperature": ("time", data)},
        coords={"time": times}
    )

    # Create a UxDataset
    uxds = ux.UxDataset(ds, uxgrid=ux.open_grid(gridfile_ne30))

    # Test monthly resampling reduces from 365 days to 12 months
    monthly = uxds.resample(time="1M").mean()
    assert "time" in monthly.dims, "time dimension missing after resample"
    assert monthly.dims["time"] < uxds.dims["time"], "time dimension not reduced"
    assert monthly.dims["time"] <= 12, "monthly resampling should give 12 or fewer time points"


def test_resample_with_cftime():
    """Test that resample works with cftime objects."""
    import numpy as np
    import pytest

    try:
        import cftime
    except ImportError:
        pytest.skip("cftime package not available")

    # Create a dataset with cftime DatetimeNoLeap objects
    times = [cftime.DatetimeNoLeap(2000, month, 15) for month in range(1, 13)]
    data = np.random.rand(12)

    # Create a simple xarray Dataset with cftime
    ds = xr.Dataset(
        {"temperature": ("time", data)},
        coords={"time": times}
    )

    # Create a UxDataset
    uxds = ux.UxDataset(ds, uxgrid=ux.open_grid(gridfile_ne30))

    # Test that quarterly resampling works with cftime
    quarterly = uxds.resample(time="Q").mean()
    assert hasattr(quarterly, "uxgrid"), "uxgrid not preserved with cftime resampling"
    assert "time" in quarterly.dims, "time dimension missing after cftime resample"
    assert quarterly.dims["time"] < uxds.dims["time"], "time dimension not reduced with cftime"

# Uncomment the following test if you want to include it, ensuring you handle potential failures.
# def test_read_from_https():
#     """Tests reading a dataset from a HTTPS link."""
#     import requests
#
#     small_file_480km = requests.get(
#         "https://web.lcrc.anl.gov/public/e3sm/inputdata/share/meshes/mpas/ocean/oQU480.230422.nc"
#     ).content
#
#     ds_small_480km = ux.open_dataset(small_file_480km, small_file_480km)
#     assert isinstance(ds_small_480km, ux.core.dataset.UxDataset)
