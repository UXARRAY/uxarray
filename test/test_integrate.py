import os
import numpy as np
import numpy.testing as nt

from unittest import TestCase
from pathlib import Path

import xarray as xr
import uxarray as ux

try:
    import constants
except ImportError:
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
        """Create a uxarray grid from vertices and saves an exodus file."""

        xr_grid = xr.open_dataset(str(mesh_file30))
        grid = ux.Grid(xr_grid)

        area = grid.calculate_total_face_area()

        nt.assert_almost_equal(area, constants.MESH30_AREA, decimal=3)

    def test_integrate(self):
        xr_grid = xr.open_dataset(mesh_file30)
        xr_psi = xr.open_dataset(data_file30)
        xr_v2 = xr.open_dataset(data_file30_v2)

        u_grid = ux.Grid(xr_grid)

        integral_psi = u_grid.integrate(xr_psi)
        integral_var2 = u_grid.integrate(xr_v2)

        nt.assert_almost_equal(integral_psi, constants.PSI_INTG, decimal=3)
        nt.assert_almost_equal(integral_var2, constants.VAR2_INTG, decimal=3)
