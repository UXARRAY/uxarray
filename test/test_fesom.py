import uxarray as ux
import os
from pathlib import Path
import pytest

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

fesom_ugrid_diag_file = current_path / "meshfiles" / "ugrid" / "fesom" / "fesom.mesh.diag.nc"

def test_open_fesom_ugrid():
    uxgrid = ux.open_grid(fesom_ugrid_diag_file)
    uxgrid.validate()
    assert uxgrid is not None
