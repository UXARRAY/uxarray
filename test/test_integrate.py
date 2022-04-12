import os
import sys
import numpy as np
import uxarray as ux
from unittest import TestCase
from pathlib import Path
from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

mesh_file30 = current_path / "meshfiles" / "outCSne30.ug"
data_file30 = current_path / "meshfiles" / "outCSne30_vortex.nc"
data_file30_v2 = current_path / "meshfiles" / "outCSne30_var2.ug"


class TestIntegrate(TestCase):

    def test_compute_triangle_area(self):
        """Create a uxarray grid from vertices and saves an exodus file."""
        verts = np.array([[0.57735027, -5.77350269e-01, -0.57735027],
                          [0.57735027, 5.77350269e-01, -0.57735027],
                          [-0.57735027, 5.77350269e-01, -0.57735027]])
        vgrid = ux.Grid(verts)
        vgrid.ds.Mesh2_node_x.attrs["units"] = "cartesian"
        vgrid.ds.Mesh2_node_y.attrs["units"] = "cartesian"
        vgrid.ds.Mesh2_node_z.attrs["units"] = "cartesian"

        area = vgrid.calculate_total_face_area()
        area = round(area, 3)

        assert (area == constants.TRI_AREA)

    def test_compute_area_file(self):
        """Create a uxarray grid from vertices and saves an exodus file."""

        grid = ux.Grid(str(mesh_file30))
        area = round(grid.calculate_total_face_area(), 3)

        assert (area == constants.MESH30_AREA)

    def test_integration(self):
        uds = ux.open_dataset(mesh_file30, data_file30, data_file30_v2)
        integral_psi = round(uds.integrate("psi"), 3)
        integral_var2 = round(uds.integrate("var2"), 3)

        assert (integral_psi == constants.PSI_INTG)
        assert (integral_var2 == constants.VAR2_INTG)
