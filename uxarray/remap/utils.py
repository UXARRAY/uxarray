from copy import deepcopy
from typing import Set

import numpy as np

import uxarray.core.dataset

KDTREE_DIM_MAP: dict[str, str] = {
    "corner nodes": "node",
    "edge centers": "edge",
    "face centers": "face",
    "nodes": "node",
    "faces": "face",
    "edges": "edge",
    "node": "node",
    "edge": "edge",
    "face": "face",
    "n_face": "face",
    "n_edge": "edge",
    "n_node": "node",
}

LABEL_TO_COORD: dict[str, str] = {
    "corner nodes": "n_node",
    "edge centers": "n_edge",
    "face centers": "n_face",
    "nodes": "n_node",
    "faces": "n_face",
    "edges": "n_edge",
    "node": "n_node",
    "edge": "n_edge",
    "face": "n_face",
    "n_face": "n_face",
    "n_edge": "n_edge",
    "n_node": "n_node",
}

SPATIAL_DIMS = {"n_node", "n_edge", "n_face"}


def _assert_dimension(dim):
    if dim not in LABEL_TO_COORD:
        raise ValueError


def _construct_remapped_ds(source, remapped_vars, destination_grid, destination_dim):
    # preserve only non-spatial coordinates
    destination_coords = deepcopy(source.coords)
    if destination_dim in destination_coords:
        del destination_coords[destination_dim]

    ds_remapped = uxarray.core.dataset.UxDataset(
        data_vars=remapped_vars,
        uxgrid=destination_grid,
        coords=destination_coords,
    )

    return ds_remapped


def _to_dataset(source):
    if isinstance(source, uxarray.core.dataarray.UxDataArray):
        is_da = True
        name = source.name if source.name is not None else "nearest_neighbor_remap"
        ds = source.to_dataset(name=name) if is_da else source
    else:
        is_da = False
        ds = source
        name = None

    return ds, is_da, name


def _get_remap_dims(ds):
    """TODO:"""
    dims_to_remap: Set[str] = set()
    for da in ds.data_vars.values():
        dims_to_remap |= set(da.dims) & SPATIAL_DIMS

    if not dims_to_remap:
        raise ValueError(
            "No spatial dimensions (n_node, n_edge, or n_face) found in source to remap."
        )

    return dims_to_remap


def _prepare_points(grid, element_dim):
    element_dim = KDTREE_DIM_MAP[element_dim]
    return np.vstack(
        [
            getattr(grid, f"{element_dim}_x").values,
            getattr(grid, f"{element_dim}_y").values,
            getattr(grid, f"{element_dim}_z").values,
        ]
    ).T
