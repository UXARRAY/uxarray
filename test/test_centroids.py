import os
from unittest import TestCase
import numpy as np
import xarray as xr
import uxarray as ux
from pathlib import Path
from uxarray.grid.coordinates import _centroid_from_mean_verts

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestCentroids(TestCase):

    def test_centroids_from_mean_verts(self):
        path = xr.open_dataset(current_path / "meshfiles" / "mpas" / "QU" /
                               "mesh.QU.1920km.151026.nc")

        # Open the dataset and find the centroids
        grid = ux.open_grid(path)
        _centroid_from_mean_verts(grid)
        self.assertIsNotNone(grid.Mesh2_face_x)
