import sys
from unittest import TestCase

from pathlib import Path
import os

import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    import uxarray as ux
else:
    import uxarray as ux


class test_dataset(TestCase):

    def test_open_dataset(self):
        """Loads files of different formats using uxarray's open_dataset
        call."""
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        uds1_name = current_path / "meshfiles" / "ov_RLL10deg_CSne4.ug"
        uds2_name = current_path / "meshfiles" / "outCSne8.g"
        uds3_name = current_path / "meshfiles" / "outCSne30.ug"

        uds1 = ux.open_dataset(uds1_name)
        uds2 = ux.open_dataset(uds2_name)
        uds3 = ux.open_dataset(uds3_name)

        assert (uds1.nMesh2_node.size == 683)
        assert (uds2.nMesh2_node.size == 386)
        assert (uds3.nMesh2_node.size == 5402)
