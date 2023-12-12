from __future__ import annotations

import numpy as np
import xarray as xr
from uxarray.constants import INT_FILL_VALUE

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.grid import Grid


class DataArraySubgridAccessor:

    def __init__(self, uxda) -> None:
        self.uxda = uxda

    def __call__(self):
        pass

    def bounding_box(self):
        pass

    def bounding_sphere(self):
        pass

    def nearest_neighbor(self, coords, k, tree_type='nodes', **kwargs):

        grid = self.uxda.uxgrid.subgrid.nearest_neighbor(
            coords, k, tree_type, **kwargs)

        return self._subset_dataarray(grid)

    def from_node_indices(self, indices):

        grid = self.uxda.uxgrid.subgrid.from_node_indices(indices)
        return self._subset_dataarray(grid)

    def from_edge_indices(self, indices):
        grid = self.uxda.uxgrid.subgrid.from_edge_indices(indices)
        return self._subset_dataarray(grid)

    def from_face_indices(self, indices):
        grid = self.uxda.uxgrid.subgrid.from_face_indices(indices)
        return self._subset_dataarray(grid)

    def _subset_dataarray(self, grid):

        from uxarray.core.dataarray import UxDataArray

        if self.uxda._face_centered():
            d_var = self.uxda.isel(
                n_face=grid._ds["subgrid_face_indices"]).values

        elif self.uxda._edge_centered():
            d_var = self.uxda.isel(
                n_edge=grid._ds["subgrid_edge_indices"]).values

        elif self.uxda._node_centered():
            d_var = self.uxda.isel(
                n_node=grid._ds["subgrid_node_indices"]).values

        else:
            raise ValueError(
                "Data variable must be either node, edge, or face centered.")

        return UxDataArray(uxgrid=grid,
                           data=d_var,
                           name=self.uxda.name,
                           dims=self.uxda.dims,
                           attrs=self.uxda.attrs)
