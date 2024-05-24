import uxarray as ux

import os
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_geos_cs = current_path / "meshfiles" / "geos-cs" / "c12" / "test-c12.native.nc4"



def test_read_geos_cs_grid():

    uxgrid = ux.open_grid(gridfile_geos_cs)

    pass


def test_read_geos_cs_uxds():
    uxds = ux.open_dataset(gridfile_geos_cs, gridfile_geos_cs)
