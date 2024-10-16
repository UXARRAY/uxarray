import uxarray as ux

import os
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

soufflet_gridpath= current_path / "meshfiles" / "fesom" / "soufflet"
pi_gridpath= current_path / "meshfiles" / "fesom" / "pi"




def test_open_fesom_ascii():
    uxgrid = ux.Grid.from_fesom2(str(soufflet_gridpath))
