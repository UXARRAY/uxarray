import uxarray as ux

import os
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridpath_fesom_ascii = current_path / "meshfiles" / "fesom" / "soufflet"



def test_open_fesom_ascii():
    uxgrid = ux.Grid.from_fesom2(str(gridpath_fesom_ascii))
    pass
