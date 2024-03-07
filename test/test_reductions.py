import uxarray as ux
import os

import pytest

from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


ds_path = current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc'

REDUCTIONS = ["mean", "max", "min", "prod", "sum", "std", "std", "var", "median", "all", "any"]


def test_node_to_face_reductions():
    uxds = ux.open_dataset(ds_path, ds_path)

    for reduction_func in REDUCTIONS:

        grid_reduction = getattr(uxds['areaTriangle'], reduction_func)(destination='face')



def test_node_to_edge_reductions():
    uxds = ux.open_dataset(ds_path, ds_path)

    for reduction_func in REDUCTIONS:

        grid_reduction = getattr(uxds['areaTriangle'], reduction_func)(destination='edge')

        a = 1


    pass
