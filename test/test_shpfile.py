import os
import numpy as np

from unittest import TestCase
from pathlib import Path

import uxarray as ux
import matplotlib.pyplot as plt

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestShpfile(TestCase):

    # shp_filename = current_path / "meshfiles" / "shp" / "cb_2018_us_nation_20m.shp"
    # shp_filename = "/Users/mbook/Downloads/boundaries_comm_area_chicago/geo_export_f1c20b37-f84d-4a5f-9452-f0d52104af02.shp"
    shp_filename = "/Users/mbook/Downloads/peoria/peoria.shp"

    def test_read_shpfile(self):
        """Read a shapefile."""

        uxgrid = ux.open_grid(self.shp_filename)
        print(uxgrid)
        # TODO: Add assertions
