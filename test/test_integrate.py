import os
import numpy as np
import numpy.testing as nt

from unittest import TestCase
from pathlib import Path

import uxarray as ux
from . import constants

# Data files
current_path = Path(os.path.dirname(os.path.realpath(__file__)))

mesh_file30 = current_path / "meshfiles" / "outCSne30.ug"
data_file30 = current_path / "meshfiles" / "outCSne30_vortex.nc"
data_file30_v2 = current_path / "meshfiles" / "outCSne30_var2.ug"


class TestIntegrate(TestCase):

    def test_calculate_total_face_area_triangle(self):
        """Create a uxarray grid from vertices and saves an exodus file."""
        verts = np.array([[0.57735027, -5.77350269e-01, -0.57735027],
                          [0.57735027, 5.77350269e-01, -0.57735027],
                          [-0.57735027, 5.77350269e-01, -0.57735027]])
        vgrid = ux.Grid(verts)

        # get node names for each grid object
        x_var = vgrid.ds_var_names["Mesh2_node_x"]
        y_var = vgrid.ds_var_names["Mesh2_node_y"]
        z_var = vgrid.ds_var_names["Mesh2_node_z"]

        vgrid.ds[x_var].attrs["units"] = "cartesian"
        vgrid.ds[y_var].attrs["units"] = "cartesian"
        vgrid.ds[z_var].attrs["units"] = "cartesian"

        area = vgrid.calculate_total_face_area()

        nt.assert_almost_equal(area, constants.TRI_AREA, decimal=3)

    def test_calculate_total_face_area_file(self):
        """Create a uxarray grid from vertices and saves an exodus file."""

        grid = ux.Grid(str(mesh_file30))

        area = grid.calculate_total_face_area()

        nt.assert_almost_equal(area, constants.MESH30_AREA, decimal=3)

    def test_integrate(self):
        uds = ux.open_dataset(mesh_file30, data_file30, data_file30_v2)

        integral_psi = uds.integrate("psi")
        integral_var2 = uds.integrate("var2")

        nt.assert_almost_equal(integral_psi, constants.PSI_INTG, decimal=3)
        nt.assert_almost_equal(integral_var2, constants.VAR2_INTG, decimal=3)
