from __future__ import annotations

import numpy as np
import xarray as xr
from uxarray.constants import INT_FILL_VALUE

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.grid import Grid


class GridSubgridAccessor:

    def __init__(self, uxgrid: Grid) -> None:
        self.uxgrid = uxgrid

    def __call__(self):
        pass

    def bounding_box(self):
        pass

    def bounding_sphere(self):
        pass

    def nearest_neighbor(self, coords, k, tree_type='nodes', **kwargs):

        coords = np.asarray(coords)

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
            return self.from_node_indices(ind)
        elif tree_type == "edges":
            return self.from_edge_indices(ind)
        else:
            return self.from_face_indices(ind)

    def from_node_indices(self, indices):

        grid = self.uxgrid

        # faces that saddle nodes given in 'indices'
        face_indices = np.unique(
            grid.node_face_connectivity.values[indices].ravel())
        face_indices = face_indices[face_indices != INT_FILL_VALUE]

        return self.from_face_indices(face_indices)

    def from_edge_indices(self, indices):

        grid = self.uxgrid

        # faces that saddle nodes given in 'indices'
        face_indices = np.unique(
            grid.edge_face_connectivity.values[indices].ravel())
        face_indices = face_indices[face_indices != INT_FILL_VALUE]

        return self.from_face_indices(face_indices)

    def from_face_indices(self, indices):

        from uxarray.grid import Grid
        grid = self.uxgrid
        ds = grid._ds

        face_indices = indices

        # nodes of each face (inclusive)
        node_indices = np.unique(
            grid.face_node_connectivity.values[face_indices].ravel())
        node_indices = node_indices[node_indices != INT_FILL_VALUE]

        # edges of each face (inclusive)
        edge_indices = np.unique(
            grid.face_edge_connectivity.values[face_indices].ravel())
        edge_indices = edge_indices[edge_indices != INT_FILL_VALUE]

        # index original dataset to obtain a 'subgrid'
        ds = ds.isel(n_node=node_indices)
        ds = ds.isel(n_face=face_indices)
        ds = ds.isel(n_edge=edge_indices)

        ds['subgrid_node_indices'] = xr.DataArray(node_indices, dims=['n_node'])
        ds['subgrid_face_indices'] = xr.DataArray(face_indices, dims=['n_face'])
        ds['subgrid_edge_indices'] = xr.DataArray(edge_indices, dims=['n_edge'])

        # mapping to update existing connectivity
        node_indices_dict = {
            key: val
            for key, val in zip(node_indices, np.arange(0, len(node_indices)))
        }
        node_indices_dict[INT_FILL_VALUE] = INT_FILL_VALUE

        for conn_name in grid._ds.data_vars:

            if "_node_connectivity" in conn_name:
                # update connectivity vars that index into nodes
                ds[conn_name] = xr.DataArray(np.vectorize(
                    node_indices_dict.__getitem__)(ds[conn_name].values),
                                             dims=ds[conn_name].dims)

            elif "_connectivity" in conn_name:
                # drop any conn that would require re-computation
                ds = ds.drop_vars(conn_name)

        return Grid.from_dataset(ds,
                                 source_grid_spec="nearest neighbor subgrid")
