import uxarray as ux
import os
from unittest import TestCase
from pathlib import Path
import numpy as np

import numpy.testing as nt

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestIntegrate(TestCase):
    gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"

    def test_single_dim(self):
        """Integral with 1D data mapped to each face."""
        uxgrid = ux.open_grid(self.gridfile_ne30)

        test_data = np.ones(uxgrid.nMesh2_face)

        dims = {"nMesh2_face": uxgrid.nMesh2_face}

        uxda = ux.UxDataArray(data=test_data,
                              dims=dims,
                              uxgrid=uxgrid,
                              name='var2')

        integral = uxda.integrate()

        # integration reduces the dimension by 1
        assert integral.ndim == len(dims) - 1

        nt.assert_almost_equal(integral, 4 * np.pi)

    def test_multi_dim(self):
        """Integral with 3D data mapped to each face."""
        uxgrid = ux.open_grid(self.gridfile_ne30)

        test_data = np.ones((5, 5, uxgrid.nMesh2_face))

        dims = {"a": 5, "b": 5, "nMesh2_face": uxgrid.nMesh2_face}

        uxda = ux.UxDataArray(data=test_data,
                              dims=dims,
                              uxgrid=uxgrid,
                              name='var2')

        integral = uxda.integrate()

        # integration reduces the dimension by 1
        assert integral.ndim == len(dims) - 1

        nt.assert_almost_equal(integral, np.ones((5, 5)) * 4 * np.pi)
