from __future__ import annotations

import numpy as np
import xarray as xr
from uxarray.constants import INT_FILL_VALUE

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.grid import Grid


class GridSubsetAccessor:

    def __init__(self, uxgrid: Grid) -> None:
        self.uxgrid = uxgrid

    def __call__(self):
        pass

    def bounding_box(self):
        pass

    def bounding_sphere(self):
        pass

    def nearest_neighbor(self, center_coord, k, tree_type='nodes', **kwargs):

        coords = np.asarray(center_coord)

        if coords.ndim > 1:
            raise ValueError("TODO")

        if len(coords) == 2:
            tree = self.uxgrid.get_ball_tree(tree_type)
        elif len(coords) == 3:
            tree = self.uxgrid.get_kd_tree(tree_type)
        else:
            raise ValueError("TODO")

        _, ind = tree.query(coords, k)

        if tree_type == "nodes":
            return self.uxgrid.isel(n_node=ind)
        elif tree_type == "edges":
            return self.uxgrid.isel(n_edge=ind)
        else:
            return self.uxgrid.isel(n_face=ind)
