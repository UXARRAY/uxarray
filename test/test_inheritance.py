import uxarray as ux
import pytest
import numpy as np
from pathlib import Path
import xarray as xr

@pytest.fixture
def uxds_fixture():
    """Fixture to load test dataset."""
    current_path = Path(__file__).resolve().parent
    quad_hex_grid_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'grid.nc'
    quad_hex_data_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'data.nc'

    # Load the dataset
    ds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)

    # Add a dummy coordinate
    if 'n_face' in ds.dims:
        n_face_size = ds.dims['n_face']
        ds = ds.assign_coords(face_id=('n_face', np.arange(n_face_size)))

    return ds

def test_isel(uxds_fixture):
    """Test that isel method preserves UxDataset type."""
    result = uxds_fixture.isel(n_face=[1, 2])
    assert isinstance(result, ux.UxDataset)
    assert hasattr(result, 'uxgrid')

def test_where(uxds_fixture):
    """Test that where method preserves UxDataset type."""
    result = uxds_fixture.where(uxds_fixture['t2m'] > uxds_fixture['t2m'].min())
    assert isinstance(result, ux.UxDataset)
    assert hasattr(result, 'uxgrid')

def test_assign(uxds_fixture):
    """Test that assign method preserves UxDataset type."""
    # Create a new variable based on t2m
    new_var = xr.DataArray(
        np.ones_like(uxds_fixture['t2m']),
        dims=uxds_fixture['t2m'].dims
    )
    result = uxds_fixture.assign(new_var=new_var)
    assert isinstance(result, ux.UxDataset)
    assert hasattr(result, 'uxgrid')
    assert result.uxgrid is uxds_fixture.uxgrid
    assert 'new_var' in result.data_vars


def test_drop_vars(uxds_fixture):
    """Test that drop_vars method preserves UxDataset type."""
    # Create a copy with a new variable so we can drop it
    ds_copy = uxds_fixture.copy(deep=True)
    ds_copy['t2m_copy'] = ds_copy['t2m'].copy()
    result = ds_copy.drop_vars('t2m_copy')
    assert isinstance(result, ux.UxDataset)
    assert hasattr(result, 'uxgrid')
    assert result.uxgrid is ds_copy.uxgrid
    assert 't2m_copy' not in result.data_vars

def test_transpose(uxds_fixture):
    """Test that transpose method preserves UxDataset type."""
    # Get all dimensions
    dims = list(uxds_fixture.dims)
    if len(dims) > 1:
        # Reverse dimensions for transpose
        reversed_dims = dims.copy()
        reversed_dims.reverse()
        result = uxds_fixture.transpose(*reversed_dims)
        assert isinstance(result, ux.UxDataset)
        assert hasattr(result, 'uxgrid')
        assert result.uxgrid is uxds_fixture.uxgrid

def test_fillna(uxds_fixture):
    """Test that fillna method preserves UxDataset type."""
    # Create a copy with some NaN values in t2m
    ds_with_nans = uxds_fixture.copy(deep=True)
    t2m_data = ds_with_nans['t2m'].values
    if t2m_data.size > 0:
        t2m_data.ravel()[0:2] = np.nan
        result = ds_with_nans.fillna(0)
        assert isinstance(result, ux.UxDataset)
        assert hasattr(result, 'uxgrid')
        # Verify NaNs were filled
        assert not np.isnan(result['t2m'].values).any()

def test_rename(uxds_fixture):
    """Test that rename method preserves UxDataset type."""
    result = uxds_fixture.rename({'t2m': 't2m_renamed'})
    assert isinstance(result, ux.UxDataset)
    assert hasattr(result, 'uxgrid')
    assert result.uxgrid is uxds_fixture.uxgrid
    assert 't2m_renamed' in result.data_vars
    assert 't2m' not in result.data_vars

def test_to_array(uxds_fixture):
    """Test that to_array method preserves UxDataArray type for its result."""
    # Create a dataset with multiple variables to test to_array
    ds_multi = uxds_fixture.copy(deep=True)
    ds_multi['t2m_celsius'] = ds_multi['t2m']
    ds_multi['t2m_kelvin'] = ds_multi['t2m'] + 273.15

    result = ds_multi.to_array()
    assert isinstance(result, ux.UxDataArray)
    assert hasattr(result, 'uxgrid')
    assert result.uxgrid == uxds_fixture.uxgrid

def test_arithmetic_operations(uxds_fixture):
    """Test arithmetic operations preserve UxDataset type."""
    # Test addition
    result = uxds_fixture['t2m'] + 1
    assert isinstance(result, ux.UxDataArray)
    assert hasattr(result, 'uxgrid')

    # Test dataset-level operations
    result = uxds_fixture * 2
    assert isinstance(result, ux.UxDataset)
    assert hasattr(result, 'uxgrid')

    # Test more complex operations
    result = uxds_fixture.copy(deep=True)
    result['t2m_squared'] = uxds_fixture['t2m'] ** 2
    assert isinstance(result, ux.UxDataset)
    assert hasattr(result, 'uxgrid')
    assert 't2m_squared' in result.data_vars

def test_reduction_methods(uxds_fixture):
    """Test reduction methods preserve UxDataset type when dimensions remain."""
    if len(uxds_fixture.dims) > 1:
        # Get a dimension to reduce over
        dim_to_reduce = list(uxds_fixture.dims)[0]

        # Test mean
        result = uxds_fixture.mean(dim=dim_to_reduce)
        assert isinstance(result, ux.UxDataset)
        assert hasattr(result, 'uxgrid')

        # Test sum on specific variable
        result = uxds_fixture['t2m'].sum(dim=dim_to_reduce)
        assert isinstance(result, ux.UxDataArray)
        assert hasattr(result, 'uxgrid')

def test_groupby(uxds_fixture):
    """Test that groupby operations preserve UxDataset type."""
    # Use face_id coordinate for grouping
    if 'face_id' in uxds_fixture.coords:
        # Create a discrete grouping variable
        grouper = uxds_fixture['face_id'] % 2  # Group by even/odd
        uxds_fixture = uxds_fixture.assign_coords(parity=grouper)
        groups = uxds_fixture.groupby('parity')
        result = groups.mean()
        assert isinstance(result, ux.UxDataset)
        assert hasattr(result, 'uxgrid')


def test_assign_coords(uxds_fixture):
    """Test that assign_coords preserves UxDataset type."""
    if 'n_face' in uxds_fixture.dims:
        dim = 'n_face'
        size = uxds_fixture.dims[dim]
        # Create a coordinate that's different from face_id
        new_coord = xr.DataArray(np.arange(size) * 10, dims=[dim])
        result = uxds_fixture.assign_coords(scaled_id=new_coord)
        assert isinstance(result, ux.UxDataset)
        assert hasattr(result, 'uxgrid')
        assert 'scaled_id' in result.coords


def test_expand_dims(uxds_fixture):
    """Test that expand_dims preserves UxDataset type."""
    result = uxds_fixture.expand_dims(dim='time')
    assert isinstance(result, ux.UxDataset)
    assert hasattr(result, 'uxgrid')
    assert 'time' in result.dims
    assert result.dims['time'] == 1

    # Verify data variable shape was updated correctly
    assert result['t2m'].shape[0] == 1

def test_method_chaining(uxds_fixture):
    """Test that methods can be chained while preserving UxDataset type."""
    # Chain multiple operations
    result = (uxds_fixture
              .assign(t2m_kelvin=uxds_fixture['t2m'] + 273.15)
              .rename({'t2m': 't2m_celsius'})
              .fillna(0))
    assert isinstance(result, ux.UxDataset)
    assert hasattr(result, 'uxgrid')
    assert 't2m_celsius' in result.data_vars
    assert 't2m_kelvin' in result.data_vars


def test_stack_unstack(uxds_fixture):
    """Test that stack and unstack preserve UxDataset type."""
    if len(uxds_fixture.dims) >= 2:
        # Get two dimensions to stack
        dims = list(uxds_fixture.dims)[:2]
        # Stack the dimensions
        stacked_name = f"{dims[0]}_{dims[1]}"
        stacked = uxds_fixture.stack({stacked_name: dims})
        assert isinstance(stacked, ux.UxDataset)
        assert hasattr(stacked, 'uxgrid')

        # Unstack them
        unstacked = stacked.unstack(stacked_name)
        assert isinstance(unstacked, ux.UxDataset)
        assert hasattr(unstacked, 'uxgrid')


def test_sortby(uxds_fixture):
    """Test that sortby preserves UxDataset type."""
    if 'face_id' in uxds_fixture.coords:
        # Create a reverse sorted coordinate
        size = len(uxds_fixture.face_id)
        uxds_fixture = uxds_fixture.assign_coords(reverse_id=('n_face', np.arange(size)[::-1]))

        # Sort by this new coordinate
        result = uxds_fixture.sortby('reverse_id')
        assert isinstance(result, ux.UxDataset)
        assert hasattr(result, 'uxgrid')
        # Verify sorting changed the order
        assert np.array_equal(result.face_id.values, np.sort(uxds_fixture.face_id.values)[::-1])

def test_shift(uxds_fixture):
    """Test that shift preserves UxDataset type."""
    if 'n_face' in uxds_fixture.dims:
        result = uxds_fixture.shift(n_face=1)
        assert isinstance(result, ux.UxDataset)
        assert hasattr(result, 'uxgrid')
        # Verify data has shifted (first element now NaN)
        assert np.isnan(result['t2m'].isel(n_face=0).values.item())
