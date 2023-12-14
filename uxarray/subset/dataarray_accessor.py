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

    def bounding_circle(self, center_coord, r, tree_type='nodes', **kwargs):
        grid = self.uxda.uxgrid.subset.bounding_circle(center_coord, r,
                                                       tree_type, **kwargs)
        return self.uxda._slice_from_grid(grid)

    def nearest_neighbor(self, center_coord, k, tree_type='nodes', **kwargs):

        grid = self.uxda.uxgrid.subset.nearest_neighbor(center_coord, k,
                                                        tree_type, **kwargs)

        return self.uxda._slice_from_grid(grid)
