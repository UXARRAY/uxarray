import os
import numpy as np

from unittest import TestCase
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestShpfile(TestCase):

    shp_filename = current_path / "meshfiles" / "shp" / "cb_2018_us_cd116_20m.shp"

    def test_read_shpfile(self):
        """Read an shapefile."""

        uxgrid = ux.open_grid(self.shp_filename)
        pass