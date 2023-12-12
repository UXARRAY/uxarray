from __future__ import annotations

import numpy as np
import xarray as xr
from uxarray.constants import INT_FILL_VALUE

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.grid import Grid


class SubgridAccessor:

    def __init__(self, uxgrid: Grid) -> None:
        self.uxgrid = uxgrid

    def __call__(self):
        pass

    def bounding_box(self):
        pass

    def bounding_sphere(self):
        pass

    def nearest_neighbor(self):
        pass

    def from_node_indices(self, indices, method='inclusive', minimal=True):

        from uxarray.grid import Grid

        grid = self.uxgrid
        ds = grid._ds

        if method == "inclusive":
            face_indices = np.unique(
                grid.node_face_connectivity.values[indices].ravel())

            node_indices = grid.face_node_connectivity.values[
                face_indices].ravel()
            node_indices = node_indices[node_indices != INT_FILL_VALUE]

            edge_indices = grid.face_edge_connectivity.values[
                face_indices].ravel()
            edge_indices = edge_indices[edge_indices != INT_FILL_VALUE]

            ds = ds.isel(n_node=node_indices)

            ds = ds.isel(n_face=face_indices)

            ds = ds.isel(n_edge=edge_indices)

            node_indices_dict = {
                key: val for key, val in zip(node_indices,
                                             np.arange(0, len(node_indices)))
            }
            node_indices_dict[INT_FILL_VALUE] = INT_FILL_VALUE

            face_indices_dict = {
                key: val for key, val in zip(face_indices,
                                             np.arange(0, len(face_indices)))
            }
            face_indices_dict[INT_FILL_VALUE] = INT_FILL_VALUE

            for conn_name in grid._ds.data_vars:

                if "_node_connectivity" in conn_name:
                    ds[conn_name] = xr.DataArray(np.vectorize(
                        node_indices_dict.__getitem__)(ds[conn_name].values),
                                                 dims=ds[conn_name].dims)

                if "_face_connectivity" in conn_name:
                    ds[conn_name] = xr.DataArray(np.vectorize(
                        face_indices_dict.__getitem__)(ds[conn_name].values),
                                                 dims=ds[conn_name].dims)

        elif method == "exclusive":
            pass

        else:
            raise ValueError(
                f"Invalid 'method', expected one of 'exclusive' or 'inclusive', but received {method}"
            )

        return Grid.from_dataset(ds,
                                 source_grid_spec="nearest neighbor subgrid")

    def from_edge_indices(self):
        pass

    def from_face_indices(self):
        pass


def _subset_from_node_indices(grid, indices, method='inclusive'):
    ds = grid._ds

    if method == "inclusive":
        face_indices = np.unique(
            grid.node_face_connectivity.values[indices].ravel())

        node_indices = grid.face_node_connectivity.values[face_indices].ravel()
        node_indices = node_indices[node_indices != INT_FILL_VALUE]

        ds = ds.isel(n_node=node_indices)

        ds = ds.isel(n_face=face_indices)

        node_indices_dict = {
            key: val
            for key, val in zip(node_indices, np.arange(0, len(node_indices)))
        }
        node_indices_dict[INT_FILL_VALUE] = INT_FILL_VALUE

        face_indices_dict = {
            key: val
            for key, val in zip(face_indices, np.arange(0, len(face_indices)))
        }
        face_indices[INT_FILL_VALUE] = INT_FILL_VALUE

        ds['face_node_connectivity'] = xr.DataArray(
            np.vectorize(node_indices_dict.__getitem__)(
                ds.face_node_connectivity.values),
            dims=['n_face', 'n_max_face_nodes'])

    elif method == "exclusive":
        pass

    else:
        raise ValueError(
            f"Invalid 'method', expected one of 'exclusive' or 'inclusive', but received {method}"
        )

    return Grid.from_dataset(ds, source_grid_spec="nearest neighbor subgrid")
