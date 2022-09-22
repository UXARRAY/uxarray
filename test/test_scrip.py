from uxarray._scrip import _read_scrip, _write_scrip
import uxarray as ux
import xarray as xr
from unittest import TestCase
import numpy as np

import os
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

ne30 = current_path / 'meshfiles' / 'outCSne30.ug'
ne8 = current_path / 'meshfiles' / 'outCSne8.nc'

ds_ne30 = xr.open_dataset(ne30, decode_times=False,
                          engine='netcdf4')  # mesh2_node_x/y
ds_ne8 = xr.open_dataset(ne8, decode_times=False,
                         engine='netcdf4')  # grid_corner_lat/lon


class TestScrip(TestCase):

    def test_exception_nonSCRIP(self):
        """Checks that exception is raised if non-SCRIP formatted file is
        passed to function."""

        self.assertRaises(TypeError, _read_scrip(ds_ne30))

    def test_scrip_is_not_ugrid(self):
        """tests that function has correctly created a ugrid function and no
        longer uses SCRIP variable names (grid_corner_lat), the function will
        raise an exception."""
        new_ds = _read_scrip(ds_ne8)

        assert ds_ne8['grid_corner_lat'].any()

        with self.assertRaises(KeyError):
            new_ds['grid_corner_lat']

    def test_scrip_writer(self):
        """Tests that input UGRID file has been successfully translated to a
        SCRIP file by looking for specific variable names in the input and
        returned datasets."""
        # Create UGRID from SCRIP file
        to_ugrid = _read_scrip(ds_ne8)
        new_path = current_path / "meshfiles" / "scrip_to_ugrid.ug"
        to_ugrid.to_netcdf(str(new_path))  # Save as new file

        # Use uxarray open_dataset to then create SCRIP file from new UGRID file
        make_ux = ux.open_dataset(str(new_path))
        to_scrip = _write_scrip(make_ux.Mesh2_face_nodes, make_ux.Mesh2_node_x,
                                make_ux.Mesh2_node_y, make_ux.face_areas)

        # Test newly created SCRIP is same as original SCRIP
        np.testing.assert_array_almost_equal(to_scrip['grid_corner_lat'],
                                             ds_ne8['grid_corner_lat'])
        np.testing.assert_array_almost_equal(to_scrip['grid_corner_lon'],
                                             ds_ne8['grid_corner_lon'])

        # Tests that calculated center lat/lon values are equivalent to original
        np.testing.assert_array_almost_equal(to_scrip['grid_center_lon'],
                                             ds_ne8['grid_center_lon'])
        np.testing.assert_array_almost_equal(to_scrip['grid_center_lat'],
                                             ds_ne8['grid_center_lat'])

        # Tests that calculated face area values are equivalent to original
        np.testing.assert_array_almost_equal(to_scrip['grid_area'],
                                             ds_ne8['grid_area'])

        # Tests that calculated grid imask values are equivalent to original
        np.testing.assert_array_almost_equal(to_scrip['grid_imask'],
                                             ds_ne8['grid_imask'])

        # Test that "mesh" variables are not in new file
        with self.assertRaises(KeyError):
            assert to_scrip['Mesh2_node_x']
            assert to_scrip['Mesh2_node_y']

    def test_scrip_variable_names(self):
        """Tests that returned dataset from writer function has all required
        SCRIP variables."""
        ux_ne30 = ux.open_dataset(ne30)
        scrip30 = _write_scrip(ux_ne30.Mesh2_face_nodes, ux_ne30.Mesh2_node_x,
                               ux_ne30.Mesh2_node_y, ux_ne30.face_areas)

        # List of relevant variable names for a scrip file
        var_list = [
            'grid_corner_lat', 'grid_dims', 'grid_imask', 'grid_area',
            'grid_center_lon'
        ]

        for i in range(len(var_list) - 1):
            assert scrip30[var_list[i]].any()

    def test_ugrid_variable_names(self):
        """Tests that returned dataset uses UGRID compliant variables."""
        mesh08 = _read_scrip(ds_ne8)

        # Create a flattened and unique array for comparisons
        corner_lon = ds_ne8['grid_corner_lon'].values
        corner_lon = corner_lon.flatten()
        strip_lon = np.unique(corner_lon)

        assert strip_lon.all() == mesh08['Mesh2_node_x'].all()
