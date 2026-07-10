import uxarray as ux
import pytest

def test_read_icon_grid(gridpath):
    grid_path = gridpath("icon", "R02B04", "icon_grid_0010_R02B04_G.nc")
    uxgrid = ux.open_grid(grid_path)

def test_read_icon_dataset(gridpath):
    grid_path = gridpath("icon", "R02B04", "icon_grid_0010_R02B04_G.nc")
    uxds = ux.open_dataset(grid_path, grid_path)

def test_icon_cross_section(gridpath):
    grid_path = gridpath("icon", "R02B04", "icon_grid_0010_R02B04_G.nc")
    uxds = ux.open_dataset(grid_path, grid_path)

    # Test cross_section with cell_area variable
    result = uxds.cell_area.cross_section(start=(0, -90), end=(0, 90))
    assert result is not None
    assert len(result) == 100  # 100 steps by default
    assert result.dims == ('steps',)
    assert all(result.coords['lon'] == 0.0)  # Constant longitude
    assert result.coords['lat'].min() == -90.0
    assert result.coords['lat'].max() == 90.0
    assert result.attrs['units'] == 'steradian'
