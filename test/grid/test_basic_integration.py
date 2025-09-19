import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE


def test_single_dim(gridpath):
    """Test single dimension integration."""
    uxds = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), gridpath("ugrid", "outCSne30", "outCSne30_vortex.nc"))

    # Test single dimension integration
    result = uxds['psi'].integrate()

    # Should return a scalar
    assert result.ndim == 0
    assert isinstance(result.values.item(), (int, float, np.number))

def test_multi_dim(gridpath):
    """Test multi-dimensional integration."""
    uxds = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), gridpath("ugrid", "outCSne30", "outCSne30_vortex.nc"))

    # Test multi-dimensional integration
    result = uxds['psi'].integrate()

    # Should handle multiple dimensions appropriately
    assert result is not None
    assert isinstance(result.values.item(), (int, float, np.number))
