import os
import numpy as np
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

class TestFaceAreas(TestCase):
    def test_compute_face_areas_geoflow_small(self):
        """Checks if the GeoFlow Small can generate a face areas output."""
        geoflow_small_grid = current_path / "meshfiles" / "geoflow-small" / "grid.nc"
        grid_1 = ux.open_dataset(geoflow_small_grid)
        grid_1.compute_face_areas()
        
    def test_compute_face_areas_fesom(self):
        """Checks if the FESOM PI-Grid Output can generate a face areas output."""

        fesom_grid_small = current_path / "meshfiles" / "fesom" / "fesom.mesh.diag.nc"
        grid_2 = ux.open_dataset(fesom_grid_small)
        grid_2.compute_face_areas()