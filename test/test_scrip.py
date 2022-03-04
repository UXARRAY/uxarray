from uxarray.reader import populate_scrip_data
import xarray as xr
import sys
import pytest
from unittest import TestCase

if "--cov" in str(sys.argv):
    from uxarray.reader import populate_scrip_data
else:
    import uxarray

# camse = 'meshfiles/f.asd2017.cesm20b05.FAMIPC6CLM5.ne0conus30x8_t12.cam.i.1980-01-01-00000.nc'
ne30 = 'meshfiles/outCSne30.ug'
ne8 = 'meshfiles/outCSne8.nc'

# ds_camse = xr.open_dataset(camse, decode_times=False, decode_cf=True)  # lat/lon
ds_ne30 = xr.open_dataset(ne30, decode_times=False,
                          engine='netcdf4')  # mesh2_node_x/y
ds_ne8 = xr.open_dataset(ne8, decode_times=False,
                         engine='netcdf4')  # grid_corner_lat/lon


def test_scrip_is_not_cf():
    """tests that if the wrong cf argument is given, the function will raise an
    exception."""
    with pytest.raises(Exception):
        # populate_scrip_data(ds_camse, is_cf=True)
        populate_scrip_data(ds_ne8, is_cf=True)
        populate_scrip_data(ds_ne30, is_cf=False)


def test_new_var_name():
    """tests that cf compliant variable names have been correctly created based
    on prior variable names."""
    # populate_scrip_data(ds_camse, is_cf=False)
    populate_scrip_data(ds_ne8, is_cf=False)
    # assert ds_camse['Mesh2_node_x'].all() == ds_camse['lat'].all()
    assert ds_ne8['Mesh2_node_x'].all() == ds_ne8['grid_corner_lat'].all()


def test_var_name_pass():
    """tests that cf compliant names and dataset overall is returned
    untouched."""
    populate_scrip_data(ds_ne30)
    try:
        ds_ne30['Mesh2_node_x']

    except NameError:
        print('Mesh2_node_x is not in this dataset')
