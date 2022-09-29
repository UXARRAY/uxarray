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
data_file30 = current_path / "meshfiles" / "outCSne30_vortex.nc"
data_file30_v2 = current_path / "meshfiles" / "outCSne30_var2.ug"


class TestIntegrate(TestCase):

    def test_integrate(self):
        uds = ux.open_dataset(mesh_file30, data_file30, data_file30_v2)

        integral_psi = uds.integrate("psi")
        integral_var2 = uds.integrate("var2")

        nt.assert_almost_equal(integral_psi, constants.PSI_INTG, decimal=3)
        nt.assert_almost_equal(integral_var2, constants.VAR2_INTG, decimal=3)
