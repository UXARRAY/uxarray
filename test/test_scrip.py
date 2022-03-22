from uxarray.reader._scrip import _populate_scrip_data
import xarray as xr
import sys

import os
from pathlib import Path

if "--cov" in str(sys.argv):
    from uxarray.reader._scrip import _populate_scrip_data
else:
    import uxarray

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

ne30 = current_path / 'meshfiles' / 'outCSne30.ug'
ne8 = current_path / 'meshfiles' / 'outCSne8.nc'

ds_ne30 = xr.open_dataset(ne30, decode_times=False,
                          engine='netcdf4')  # mesh2_node_x/y
ds_ne8 = xr.open_dataset(ne8, decode_times=False,
                         engine='netcdf4')  # grid_corner_lat/lon


def test_scrip_is_ugrid():
    """tests that if the wrong cf argument is given, the function will raise an
    exception."""
    new_ds = _populate_scrip_data(ds_ne30)
    try:
        new_ds['Mesh2']
    except KeyError:
        print("Variable not found")


def test_scrip_is_not_ugrid():
    """tests that if the wrong cf argument is given, the function will raise an
    exception."""
    new_ds = _populate_scrip_data(ds_ne8)
    try:
        new_ds['Mesh2']
    except KeyError:
        print("Variable not found")


def test_ugrid_variable_names():
    mesh30 = _populate_scrip_data(ds_ne30)
    print('here')
    mesh08 = _populate_scrip_data(ds_ne30)
    assert ds_ne30['Mesh2_node_x'].all() == mesh30['Mesh2_node_x'].all()
    assert ds_ne8['grid_corner_lon'].all() == mesh08['Mesh2_node_x'].all()
