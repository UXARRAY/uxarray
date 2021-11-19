import sys
from unittest import TestCase

import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from uxarray import Grid
else:
    from uxarray import Grid


class test_grid(TestCase):

    def test_load_exofile(self):
        exo_filename = "./test/hex_2x2x2_ss.exo"
        tgrid = Grid(exo_filename)

        # check get_filename function
        assert (tgrid.get_filename() == exo_filename)

        # check rename filename function
        new_filename = "1hex.exo"
        tgrid.rename_file(new_filename)
        assert (tgrid.get_filename() == new_filename)

    def test_load_exo2file(self):
        exo_filename = "./test/outCSne8.g"
        tgrid = Grid(exo_filename)

    def test_load_scrip(self):
        exo_filename = "./test/outCSne8.nc"
        tgrid = Grid(exo_filename)

    def test_load_ugrid(self):
        ugrid_file = "./test/sphere_mixed.1.lb8.ugrid"
        tgrid = Grid(ugrid_file)

    # use external package to read?
    # https://gis.stackexchange.com/questions/113799/how-to-read-a-shapefile-in-python
    def test_load_shpfile(self):
        ugrid_file = "./test/grid_fire.shp"
        tgrid = Grid(ugrid_file)
