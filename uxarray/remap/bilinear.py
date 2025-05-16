from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numba import njit

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray
    from uxarray.core.dataset import UxDataset

from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
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

        weights, indices = _get_values(
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


def _get_values(point_xyz, dual, data_size, source_grid):
    """Get barycentric weights and source face indices for each destination point."""
    all_weights = np.zeros((data_size, 4), dtype=np.float64)
    all_indices = np.zeros((data_size, 4), dtype=int)

    for i in range(data_size):
        # Get dual face(s) that contain the point
        face_ind = dual.get_faces_containing_point(point_xyz=point_xyz[i])

        # If not found in dual, fallback to source grid
        if len(face_ind) == 0:
            face_ind = source_grid.get_faces_containing_point(point_xyz=point_xyz[i])
            if len(face_ind) == 0:
                # No face found, weights remain as 0
                continue
            else:
                # Point is in one face, weight 1 to that face center
                all_weights[i, 0] = 1.0
                all_indices[i, 0] = face_ind[0]
                continue

        # Proceed with barycentric interpolation from dual face
        face_id = face_ind[0]
        node_inds = dual.face_node_connectivity[face_id].values
        nodes_per_face = dual.n_nodes_per_face[face_id].values

        polygon_xyz = np.zeros((nodes_per_face, 3), dtype=np.float64)
        polygon_face_indices = np.full(nodes_per_face, INT_FILL_VALUE, dtype=int)

        for j, node in enumerate(node_inds[:nodes_per_face]):
            if node == INT_FILL_VALUE:
                break
            x = dual.node_x.values[node]
            y = dual.node_y.values[node]
            z = dual.node_z.values[node]
            polygon_xyz[j] = [x, y, z]
            polygon_face_indices[j] = node

        # Trim the arrays to get only the valid entries
        valid = polygon_face_indices != INT_FILL_VALUE
        polygon_xyz = polygon_xyz[valid]
        polygon_face_indices = polygon_face_indices[valid]

        # Check if point matches a node
        matching_index = _find_matching_node_index(
            polygon_xyz, point_xyz[i], tolerance=ERROR_TOLERANCE
        )

        if matching_index[0] != -1:
            # If the point lies on a node (face center), assign weight 1 to that node
            all_weights[i, 0] = 1.0
            all_indices[i, 0] = polygon_face_indices[matching_index[0]]
        else:
            # Otherwise, use barycentric coordinates to get weights and face indices
            weights, node_indices = barycentric_coordinates_cartesian(
                polygon_xyz=polygon_xyz,
                point_xyz=point_xyz[i],
            )

            # Map the node indices to the corresponding face center indices in the source mesh
            contributing_face_indices = [polygon_face_indices[j] for j in node_indices]

            # Assign the calculated weights and indices
            all_weights[i, : len(weights)] = weights
            all_indices[i, : len(contributing_face_indices)] = contributing_face_indices

    return all_weights, all_indices


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
