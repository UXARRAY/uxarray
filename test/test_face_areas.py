import os
import numpy as np
import numpy.testing as nt

from unittest import TestCase
from pathlib import Path

import uxarray as ux

try:
    import constants
except ImportError:
    from . import constants

# Data files
current_path = Path(os.path.dirname(os.path.realpath(__file__)))

mesh_file30 = current_path / "meshfiles" / "outCSne30.ug"

class TestFaceAreas(TestCase):

    def test_calculate_total_face_area_triangle(self):
        """Create a uxarray grid from vertices and 
        calculate the area using gaussian and triangular quadrature rules."""
        verts = np.array([[0.57735027, -5.77350269e-01, -0.57735027],
                          [0.57735027, 5.77350269e-01, -0.57735027],
                          [-0.57735027, 5.77350269e-01, -0.57735027]])
        vgrid = ux.Grid(verts)

        # get node names for each grid object
        x_var = vgrid.ds_var_names["Mesh2_node_x"]
        y_var = vgrid.ds_var_names["Mesh2_node_y"]
        z_var = vgrid.ds_var_names["Mesh2_node_z"]

        vgrid.ds[x_var].attrs["units"] = "m"
        vgrid.ds[y_var].attrs["units"] = "m"
        vgrid.ds[z_var].attrs["units"] = "m"

        area_gaussian = vgrid.calculate_total_face_area(
            quadrature_rule="gaussian", order=5)
        nt.assert_almost_equal(area_gaussian, constants.TRI_AREA, decimal=3)

        area_triangular = vgrid.calculate_total_face_area(
            quadrature_rule="triangular", order=4)
        nt.assert_almost_equal(area_triangular, constants.TRI_AREA, decimal=1)

    def test_calculate_total_face_area_file(self):
        """Load a grid from file and calculate the total face area of the mesh."""

        grid = ux.open_dataset(str(mesh_file30))
        area = grid.calculate_total_face_area()
        nt.assert_almost_equal(area, constants.MESH30_AREA, decimal=3)
        
    def test_compute_face_areas(self):
        """Load a grid from file and calculate the area of each face of the mesh."""
        
        grid = ux.open_dataset(str(mesh_file30))
        area = grid.compute_face_areas()
        assert(area.size == grid.ds.nMesh2_face.size)
        # sum the area of all faces with np.sum
        nt.assert_almost_equal(np.sum(area), constants.MESH30_AREA, decimal=3)
