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
        return repr(self)

    def __repr__(self):
        prefix = "<uxarray.UxDataArray.subset>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += "  * nearest_neighbor(center_coord, k, tree_type, **kwargs)\n"
        methods_heading += "  * bounding_circle(center_coord, r, tree_type, **kwargs)\n"
        methods_heading += "  * bounding_box(lon_bounds, lat_bounds, method, element)\n"

        return prefix + methods_heading

    def bounding_box(self,
                     lon_bounds,
                     lat_bounds,
                     method='naive',
                     element='nodes'):
        grid = self.uxda.uxgrid.subset.bounding_box(lon_bounds, lat_bounds,
                                                    method, element)

        return self.uxda._slice_from_grid(grid)

    def bounding_circle(self, center_coord, r, tree_type='nodes', **kwargs):
        grid = self.uxda.uxgrid.subset.bounding_circle(center_coord, r,
                                                       tree_type, **kwargs)
        return self.uxda._slice_from_grid(grid)

    def nearest_neighbor(self, center_coord, k, tree_type='nodes', **kwargs):

        grid = self.uxda.uxgrid.subset.nearest_neighbor(center_coord, k,
                                                        tree_type, **kwargs)

        return self.uxda._slice_from_grid(grid)
