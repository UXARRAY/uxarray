import os
import numpy as np

from unittest import TestCase
from pathlib import Path

import uxarray as ux
import matplotlib.pyplot as plt

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestShpfile(TestCase):

    # shp_filename = current_path / "meshfiles" / "shp" / "cb_2018_us_nation_20m"/"cb_2018_us_nation_20m.shp"
    shp_filename = current_path / "meshfiles" / "shp" / "5poly/polygons.shp"

    def test_read_shpfile(self):
        """Read a shapefile."""

        uxgrid = ux.Grid.from_shapefile(self.shp_filename)
        assert(uxgrid.validate())
