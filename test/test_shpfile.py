import os
import numpy as np

from unittest import TestCase
from pathlib import Path

import uxarray as ux
import matplotlib.pyplot as plt

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestShpfile(TestCase):

    shp_filename = current_path / "meshfiles" / "shp" / "cb_2018_us_nation_20m"/"cb_2018_us_nation_20m.shp"
    shp_filename_5poly = current_path / "meshfiles" / "shp" / "5poly/5poly.shp"
    shp_filename_multi = current_path / "meshfiles" / "shp" / "multipoly/multipoly.shp"

    def test_read_shpfile(self):
        """Read a shapefile."""

        uxgrid = ux.Grid.from_shapefile(self.shp_filename)
        assert(uxgrid.validate())

    def test_read_shpfile_multi(self):
        """Read a shapefile, that consists of multipolygons."""

        uxgrid = ux.Grid.from_shapefile(self.shp_filename_multi)
        assert(uxgrid.validate())

    def test_read_shpfile_5poly(self):
        """Read a shapefile, that consists of 5 polygons of different
        shapes."""

        uxgrid = ux.Grid.from_shapefile(self.shp_filename_5poly)
        assert(uxgrid.validate())
