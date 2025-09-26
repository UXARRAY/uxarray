import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE


def test_single_dim(gridpath):
    """Integral with 1D data mapped to each face."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    test_data = np.ones(uxgrid.n_face)
    dims = {"n_face": uxgrid.n_face}
    uxda = ux.UxDataArray(data=test_data, dims=dims, uxgrid=uxgrid, name='var2')
    integral = uxda.integrate()
    assert integral.ndim == len(dims) - 1
    nt.assert_almost_equal(integral, 4 * np.pi)


def test_multi_dim(gridpath):
    """Integral with 3D data mapped to each face."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    test_data = np.ones((5, 5, uxgrid.n_face))
    dims = {"a": 5, "b": 5, "n_face": uxgrid.n_face}
    uxda = ux.UxDataArray(data=test_data, dims=dims, uxgrid=uxgrid, name='var2')
    integral = uxda.integrate()
    assert integral.ndim == len(dims) - 1
    nt.assert_almost_equal(integral, np.ones((5, 5)) * 4 * np.pi)
