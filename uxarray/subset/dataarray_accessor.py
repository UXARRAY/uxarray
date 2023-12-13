from __future__ import annotations

import numpy as np
import xarray as xr
from uxarray.constants import INT_FILL_VALUE

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.grid import Grid


class DataArraySubsetAccessor:

    def __init__(self, uxda) -> None:
        self.uxda = uxda

    def __call__(self):
        pass

    def bounding_box(self):
        pass

    def bounding_sphere(self):
        pass

    def nearest_neighbor(self, coords, k, tree_type='nodes', **kwargs):

        grid = self.uxda.uxgrid.subset.nearest_neighbor(coords, k, tree_type,
                                                        **kwargs)

        return self.uxda._slice_uxdataarray(grid)
