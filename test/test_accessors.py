"""
Test module for UxArray accessor functionality (groupby, resample, rolling, etc.).

This module tests that accessor methods properly preserve uxgrid attributes
and return UxDataArray/UxDataset objects.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import uxarray as ux
from uxarray import UxDataset
import pytest


current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"
gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
dsfile_v1_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"
mpas_ds_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'


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

def test_groupby_bins_preserves_uxgrid():
    """Test that groupby_bins operations preserve the uxgrid attribute."""
    # Create a dataset from a file
    uxds = ux.open_dataset(mpas_ds_path, mpas_ds_path)
    original_grid = uxds.uxgrid

    # Create bins from latitude values (extract data explicitly)
    lat_bins = [-90, -45, 0, 45, 90]

    # Test DataArray groupby_bins preserves uxgrid
    da_result = uxds.latCell.groupby_bins(uxds.latCell, bins=lat_bins).mean()
    assert hasattr(da_result, "uxgrid")
    assert da_result.uxgrid is not None

    # Test Dataset groupby_bins preserves uxgrid
    ds_result = uxds.groupby_bins(uxds.latCell, bins=lat_bins).mean()
    assert hasattr(ds_result, "uxgrid")
    assert ds_result.uxgrid is not None
    assert ds_result.uxgrid == original_grid



def test_resample_preserves_uxgrid_and_reduces_time():
    """Test that resample operations preserve uxgrid and reduce time dimension."""

    # Create a simple test with only time dimension
    times = pd.date_range("2000-01-01", periods=12, freq="D")
    temp_data = np.random.rand(12)

    # Create a simple xarray Dataset
    xr_ds = xr.Dataset(
        {"temperature": ("time", temp_data)},
        coords={"time": times}
    )

    # Open the minimal dataset with a real grid
    # Use existing test file we know works
    uxgrid = ux.open_grid(gridfile_ne30)

    # Create a UxDataset with this grid
    uxds = ux.UxDataset(xr_ds, uxgrid=uxgrid)

    # Test the resample method directly
    result = uxds.temperature.resample(time="1W").mean()

    # Test assertions
    assert hasattr(result, "uxgrid"), "uxgrid not preserved on resample"
    assert result.uxgrid == uxds.uxgrid, "uxgrid not equal after resample"
    assert len(result.time) < len(uxds.time), "time dimension not reduced"

def test_resample_preserves_uxgrid():
    """Test that resample preserves the uxgrid attribute."""

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


def test_rolling_preserves_uxgrid():
    """Test that rolling operations preserve the uxgrid attribute."""

    # Create a dataset with time dimension
    times = pd.date_range("2000-01-01", periods=30, freq="D")
    data = np.random.rand(30)

    # Create a simple xarray Dataset
    ds = xr.Dataset(
        {"temperature": ("time", data)},
        coords={"time": times}
    )

    # Create a UxDataset with a real grid
    uxds = ux.UxDataset(ds, uxgrid=ux.open_grid(gridfile_ne30))
    original_uxgrid = uxds.uxgrid

    # Test DataArray rolling preserves uxgrid
    da_rolling = uxds.temperature.rolling(time=7)
    da_result = da_rolling.mean()
    assert hasattr(da_result, "uxgrid"), "uxgrid not preserved on DataArray rolling"
    assert da_result.uxgrid is original_uxgrid, "uxgrid not identical after DataArray rolling"

    # Test Dataset rolling preserves uxgrid
    ds_rolling = uxds.rolling(time=7)
    ds_result = ds_rolling.mean()
    assert hasattr(ds_result, "uxgrid"), "uxgrid not preserved on Dataset rolling"
    assert ds_result.uxgrid is original_uxgrid, "uxgrid not identical after Dataset rolling"

    # Test that rolling window operations work correctly
    assert len(da_result.time) == len(uxds.time), "rolling should preserve time dimension length"
    assert not np.isnan(da_result.values[6:]).any(), "rolling mean should have valid values after window size"


def test_coarsen_preserves_uxgrid():
    """Test that coarsen operations preserve the uxgrid attribute."""

    # Create a dataset with time dimension (multiple of coarsen factor)
    times = pd.date_range("2000-01-01", periods=24, freq="D")
    data = np.random.rand(24)

    # Create a simple xarray Dataset
    ds = xr.Dataset(
        {"temperature": ("time", data)},
        coords={"time": times}
    )

    # Create a UxDataset with a real grid
    uxds = ux.UxDataset(ds, uxgrid=ux.open_grid(gridfile_ne30))
    original_uxgrid = uxds.uxgrid

    # Test DataArray coarsen preserves uxgrid
    da_coarsen = uxds.temperature.coarsen(time=3)
    da_result = da_coarsen.mean()
    assert hasattr(da_result, "uxgrid"), "uxgrid not preserved on DataArray coarsen"
    assert da_result.uxgrid is original_uxgrid, "uxgrid not identical after DataArray coarsen"

    # Test Dataset coarsen preserves uxgrid
    ds_coarsen = uxds.coarsen(time=3)
    ds_result = ds_coarsen.mean()
    assert hasattr(ds_result, "uxgrid"), "uxgrid not preserved on Dataset coarsen"
    assert ds_result.uxgrid is original_uxgrid, "uxgrid not identical after Dataset coarsen"

    # Test that coarsen reduces dimension correctly
    assert len(da_result.time) == 8, "coarsen by 3 should reduce 24 points to 8"
    assert ds_result.dims["time"] == 8, "coarsen should reduce time dimension"


def test_weighted_preserves_uxgrid():
    """Test that weighted operations preserve the uxgrid attribute."""

    # Create a dataset with time and face dimensions
    times = pd.date_range("2000-01-01", periods=10, freq="D")

    # Open a real dataset to get face dimension
    uxds_base = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)
    n_face = uxds_base.dims["n_face"]

    # Create data with time and face dimensions
    temp_data = np.random.rand(10, n_face)
    weights_data = np.random.rand(10)  # weights along time

    # Create a Dataset with both variables
    ds = xr.Dataset(
        {
            "temperature": (["time", "n_face"], temp_data),
            "weights": ("time", weights_data)
        },
        coords={"time": times}
    )

    # Create a UxDataset
    uxds = ux.UxDataset(ds, uxgrid=uxds_base.uxgrid)
    original_uxgrid = uxds.uxgrid

    # Test DataArray weighted preserves uxgrid
    da_weighted = uxds.temperature.weighted(uxds.weights)
    da_result = da_weighted.mean("time")
    assert hasattr(da_result, "uxgrid"), "uxgrid not preserved on DataArray weighted"
    assert da_result.uxgrid is original_uxgrid, "uxgrid not identical after DataArray weighted"

    # Test Dataset weighted preserves uxgrid
    ds_weighted = uxds.weighted(uxds.weights)
    ds_result = ds_weighted.mean("time")
    assert hasattr(ds_result, "uxgrid"), "uxgrid not preserved on Dataset weighted"
    assert ds_result.uxgrid is original_uxgrid, "uxgrid not identical after Dataset weighted"

    # Test that weighted operations reduce dimensions correctly
    assert "time" not in da_result.dims, "weighted mean over time should remove time dimension"
    assert "n_face" in da_result.dims, "face dimension should be preserved"
    assert da_result.shape == (n_face,), "result should only have face dimension"


def test_cumulative_preserves_uxgrid():
    """Test that cumulative operations preserve the uxgrid attribute."""

    # Create a dataset with time dimension
    times = pd.date_range("2000-01-01", periods=10, freq="D")
    data = np.random.rand(10)

    # Create a simple xarray Dataset
    ds = xr.Dataset(
        {"temperature": ("time", data)},
        coords={"time": times}
    )

    # Create a UxDataset with a real grid
    uxds = ux.UxDataset(ds, uxgrid=ux.open_grid(gridfile_ne30))
    original_uxgrid = uxds.uxgrid

    # Test DataArray cumulative preserves uxgrid
    da_cumulative = uxds.temperature.cumulative("time")
    da_result = da_cumulative.sum()
    assert hasattr(da_result, "uxgrid"), "uxgrid not preserved on DataArray cumulative"
    assert da_result.uxgrid is original_uxgrid, "uxgrid not identical after DataArray cumulative"

    # Test Dataset cumulative preserves uxgrid
    ds_cumulative = uxds.cumulative("time")
    ds_result = ds_cumulative.sum()
    assert hasattr(ds_result, "uxgrid"), "uxgrid not preserved on Dataset cumulative"
    assert ds_result.uxgrid is original_uxgrid, "uxgrid not identical after Dataset cumulative"

    # Test that cumulative preserves dimension length
    assert len(da_result.time) == len(uxds.time), "cumulative should preserve time dimension length"
