import uxarray as ux

import os
from pathlib import Path
from unittest import TestCase

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

fesom_ugrid_diag_file= current_path / "meshfiles" / "ugrid" / "fesom" / "fesom.mesh.diag.nc"

class TestFesom(TestCase):
    def test_open_fesom_ugrid(self):
        uxgrid = ux.open_grid(fesom_ugrid_diag_file)
        uxgrid.validate()
