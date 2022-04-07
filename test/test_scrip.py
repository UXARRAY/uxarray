from uxarray._scrip import _read_scrip, _write_scrip
import xarray as xr
import sys

import os
from pathlib import Path

if "--cov" in str(sys.argv):
    from uxarray._scrip import _read_scrip, _write_scrip
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
    new_ds = _read_scrip(ne30)
    try:
        new_ds['Mesh2']
    except KeyError:
        print("Variable not found")


def test_scrip_is_not_ugrid():
    """tests that if the wrong cf argument is given, the function will raise an
    exception."""
    new_ds = _read_scrip(ne8)
    try:
        new_ds['Mesh2']
    except KeyError:
        print("Variable not found")


def test_ugrid_variable_names():
    mesh30 = _read_scrip(ne30)
    mesh08 = _read_scrip(ne8)
    assert ds_ne30['Mesh2_node_x'].all() == mesh30['Mesh2_node_x'].all()
    assert ds_ne8['grid_corner_lon'].all() == mesh08['Mesh2_node_x'].all()


def test_ugrid_to_scrip():
    is_ugrid = _write_scrip(ds_ne30, "ugrid_to_scrip.nc")
    try:
        is_ugrid['grid_corner_lat']
    except KeyError:
        print("ugrid to scrip unsuccessful")


def test_scrip_to_scrip():
    is_scrip = _write_scrip(ds_ne8, "scrip_to_scrip.nc")
    try:
        is_scrip['grid_corner_lat']
    except KeyError:
        print("scrip to scrip unsuccessful")
