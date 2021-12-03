import sys
from unittest import TestCase

import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    import uxarray as ux
else:
    import uxarray as ux


class test_grid(TestCase):

    def test_load_exofile(self):

        try:
            exo_filename = "./hex_2x2x2_ss.exo"
            tgrid = ux.Grid(exo_filename)
        except:
            exo_filename = "./test/hex_2x2x2_ss.exo"
            tgrid = ux.Grid(exo_filename)

        # check get_filename function
        assert (tgrid.get_filename() == exo_filename)

        # check rename filename function
        new_filename = "1hex.exo"
        tgrid.rename_file(new_filename)
        assert (tgrid.get_filename() == new_filename)

    def test_load_exo2file(self):

        try:
            exo_filename = "./outCSne8.g"
            tgrid = ux.Grid(exo_filename)
        except:
            exo_filename = "./test/outCSne8.g"
            tgrid = ux.Grid(exo_filename)

    def test_load_scrip(self):

        try:
            exo_filename = "./outCSne8.nc"
            tgrid = ux.Grid(exo_filename)
        except:
            exo_filename = "./test/outCSne8.nc"
            tgrid = ux.Grid(exo_filename)

    # def test_load_ugrid(self):
    #     ugrid_file = "./sphere_mixed.1.lb8.ugrid"
    #     tgrid = ux.Grid(ugrid_file)
    #
    # # use external package to read?
    # # https://gis.stackexchange.com/questions/113799/how-to-read-a-shapefile-in-python
    # def test_load_shpfile(self):
    #     ugrid_file = "./grid_fire.shp"
    #     tgrid = ux.Grid(ugrid_file)
