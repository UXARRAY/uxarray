import uxarray as ux
import pytest
from pathlib import Path
import os

import numpy.testing as nt

# Define the current path and file paths for grid and data
current_path = Path(os.path.dirname(os.path.realpath(__file__)))
quad_hex_grid_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'grid.nc'
quad_hex_data_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'data.nc'



class TestQuadHex:
    """The quad hexagon grid contains four faces.

    Top Left Face: Index 1

    Top Right Face: Index 2

    Bottom Left Face: Index 0

    Bottom Right Face: Index 3

    The top two faces intersect a constant latitude of 0.1

    The bottom two faces intersect a constant latitude of -0.1

    All four faces intersect a constant latitude of 0.0
    """

    def test_constant_lat_cross_section_grid(self):
        uxgrid = ux.open_grid(quad_hex_grid_path)

        grid_top_two = uxgrid.constant_latitude_cross_section(lat=0.1)

        assert grid_top_two.n_face == 2

        grid_bottom_two = uxgrid.constant_latitude_cross_section(lat=-0.1)

        assert grid_bottom_two.n_face == 2

        grid_all_four = uxgrid.constant_latitude_cross_section(lat=0.0)

        assert grid_all_four.n_face == 4

        with pytest.raises(ValueError):
            # no intersections found at this line
            uxgrid.constant_latitude_cross_section(lat=10.0)


    def test_constant_lat_cross_section_uxds(self):
        uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)

        da_top_two = uxds['t2m'].constant_latitude_cross_section(lat=0.1)

        nt.assert_array_equal(da_top_two.data, uxds['t2m'].isel(n_face=[1, 2]).data)

        da_bottom_two = uxds['t2m'].constant_latitude_cross_section(lat=-0.1)

        nt.assert_array_equal(da_bottom_two.data, uxds['t2m'].isel(n_face=[0, 3]).data)

        da_all_four = uxds['t2m'].constant_latitude_cross_section(lat=0.0)

        nt.assert_array_equal(da_all_four.data , uxds['t2m'].data)

        with pytest.raises(ValueError):
            # no intersections found at this line
            uxds['t2m'].constant_latitude_cross_section(lat=10.0)