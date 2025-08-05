from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numba import njit, prange

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray
    from uxarray.core.dataset import UxDataset

from uxarray.constants import ERROR_TOLERANCE
from uxarray.grid import Grid
from uxarray.grid.geometry import barycentric_coordinates_cartesian

from .utils import (
    KDTREE_DIM_MAP,
    LABEL_TO_COORD,
    SPATIAL_DIMS,
    _assert_dimension,
    _construct_remapped_ds,
    _get_remap_dims,
    _prepare_points,
    _to_dataset,
)


def _bilinear(
    source: UxDataArray | UxDataset,
    destination_grid: Grid,
    destination_dim: str = "n_face",
) -> np.ndarray:
    """Bilinear Remapping between two grids, mapping data that resides on the
    corner nodes, edge centers, or face centers on the source grid to the
    corner nodes, edge centers, or face centers of the destination grid.

    Parameters
    ---------
    source_uxda : UxDataArray
        Source UxDataArray
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes", "edge centers", or "face centers"

    Returns
    -------
    destination_data : np.ndarray
        Data mapped to destination grid
    """

    # ensure array is a np.ndarray
    _assert_dimension(destination_dim)

    # Ensure the destination grid is normalized
    destination_grid.normalize_cartesian_coordinates()

    # Perform remapping on a UxDataset
    ds, is_da, name = _to_dataset(source)

    # Determine which spatial dimensions we need to remap
    dims_to_remap = _get_remap_dims(ds)

    for src_dim in dims_to_remap:
        if src_dim != "n_face":
            raise ValueError(
                "Bilinear remapping is not supported for non-face centered variables"
            )

        # Construct dual for searching
        dual = source.uxgrid.get_dual()

        # get destination coordinate pairs
        point_xyz = _prepare_points(destination_grid, destination_dim)

        weights, indices = _barycentric_weights(
            point_xyz=point_xyz,
            dual=dual,
            data_size=getattr(destination_grid, f"n_{KDTREE_DIM_MAP[destination_dim]}"),
            source_grid=ds.uxgrid,
        )

    remapped_vars = {}
    for name, da in ds.data_vars.items():
        spatial = set(da.dims) & SPATIAL_DIMS
        if spatial:
            source_dim = spatial.pop()
            inds, w = indices, weights

            # pack indices & weights into tiny DataArrays:
            indexer = xr.DataArray(inds, dims=[LABEL_TO_COORD[destination_dim], "k"])
            weight_da = xr.DataArray(w, dims=[LABEL_TO_COORD[destination_dim], "k"])

            # gather the k neighbor values:
            da_k = da.isel({source_dim: indexer}, ignore_grid=True)

            # weighted sum over the small "k" axis:
            da_idw = (da_k * weight_da).sum(dim="k")

            # attach the new grid
            da_idw.uxgrid = destination_grid
            remapped_vars[name] = da_idw
        else:
            remapped_vars[name] = da

    ds_remapped = _construct_remapped_ds(
        source, remapped_vars, destination_grid, destination_dim
    )

    return ds_remapped[name] if is_da else ds_remapped


def _barycentric_weights(point_xyz, dual, data_size, source_grid):
    """Get barycentric weights and source face indices for each destination point."""
    all_weights = np.zeros((data_size, 4), dtype=np.float64)
    all_indices = np.zeros((data_size, 4), dtype=int)

    # Query dual grid
    face_indices, hits = dual.get_faces_containing_point(points=point_xyz)

    # Handle fallback cases (hits == 0)
    fallback_mask = hits == 0
    fallback_idxs = np.where(fallback_mask)[0]
    for i in fallback_idxs:
        cur_inds, counts = source_grid.get_faces_containing_point(points=point_xyz[i])
        if counts == 0:
            continue
        all_weights[i, 0] = 1.0
        all_indices[i, 0] = int(cur_inds[0])

    # Prepare args for the Numba function
    valid_idxs = np.where(hits != 0)[0]

    # Call numba function to find weights
    _calculate_weights(
        valid_idxs,
        point_xyz,
        face_indices,
        dual.node_x.values,
        dual.node_y.values,
        dual.node_z.values,
        dual.face_node_connectivity.values,
        dual.n_nodes_per_face.values,
        dual.n_face,
        all_weights,
        all_indices,
    )

    return all_weights, all_indices


@njit(cache=True, parallel=True)
def _calculate_weights(
    valid_idxs,
    point_xyz,
    face_indices,
    x,
    y,
    z,
    face_node_conn,
    n_nodes_per_face,
    n_faces,
    all_weights,
    all_indices,
):
    for idx in prange(len(valid_idxs)):
        fidx = int(face_indices[valid_idxs[idx], 0])
        # bounds check: ensure face index is within valid range (0 to n_faces-1)
        if fidx < 0 or fidx >= n_faces:
            continue
        nverts = int(n_nodes_per_face[fidx])
        polygon_xyz = np.zeros((nverts, 3), dtype=np.float64)
        polygon_face_indices = np.empty(nverts, dtype=np.int32)
        for j in range(nverts):
            node = face_node_conn[fidx, j]
            polygon_xyz[j, 0] = x[node]
            polygon_xyz[j, 1] = y[node]
            polygon_xyz[j, 2] = z[node]
            polygon_face_indices[j] = node

        # snap check
        match = _find_matching_node_index(polygon_xyz, point_xyz[valid_idxs[idx]])
        if match[0] != -1:
            all_weights[valid_idxs[idx], 0] = 1.0
            all_indices[valid_idxs[idx], 0] = polygon_face_indices[match[0]]
            continue

        weights, node_idxs = barycentric_coordinates_cartesian(
            polygon_xyz, point_xyz[valid_idxs[idx]]
        )
        for k in range(len(weights)):
            all_weights[valid_idxs[idx], k] = weights[k]
            all_indices[valid_idxs[idx], k] = polygon_face_indices[node_idxs[k]]


@njit(cache=True)
def _find_matching_node_index(nodes, point, tolerance=ERROR_TOLERANCE):
    for i in range(nodes.shape[0]):
        match = True
        for j in range(3):  # Compare each coordinate
            diff = abs(nodes[i, j] - point[j])
            if diff > tolerance + tolerance * abs(point[j]):
                match = False
                break
        if match:
            return np.array([i])  # Return first matching index
    return np.array([-1])  # Not found
