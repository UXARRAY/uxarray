from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray

import numpy as np

import uxarray.core.dataarray
import uxarray.core.dataset
from uxarray.grid import Grid
import warnings

from copy import deepcopy

from uxarray.remap.utils import _remap_grid_parse


def _inverse_distance_weighted_remap(
    source_grid: Grid,
    destination_grid: Grid,
    source_data: np.ndarray,
    remap_to: str = "face centers",
    coord_type: str = "spherical",
    power=2,
    k=8,
) -> np.ndarray:
    """Inverse Distance Weighted Remapping between two grids.

    Parameters:
    -----------
    source_grid : Grid
        Source grid that data is mapped from.
    destination_grid : Grid
        Destination grid to remap data to.
    source_data : np.ndarray
        Data variable to remap.
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes", "edge centers", or "face centers".
    coord_type: str, default="spherical"
        Coordinate type to use for nearest neighbor query, either "spherical" or "Cartesian".
    power : int, default=2
        Power parameter for inverse distance weighting. This controls how local or global the remapping is, a higher
        power causes points that are further away to have less influence
    k : int, default=8
        Number of nearest neighbors to consider in the weighted calculation.

    Returns:
    --------
    destination_data : np.ndarray
        Data mapped to the destination grid.
    """

    if power > 5:
        warnings.warn("It is recommended not to exceed a power of 5.0.", UserWarning)
    if k > source_grid.n_node:
        raise ValueError(
            f"Number of nearest neighbors to be used in the calculation is {k}, but should not exceed the "
            f"number of nodes in the source grid of {source_grid.n_node}"
        )
    if k <= 1:
        raise ValueError(
            f"Number of nearest neighbors to be used in the calculation is {k}, but should be greater than 1"
        )

    # ensure array is a np.ndarray
    source_data = np.asarray(source_data)

    _, distances, nearest_neighbor_indices = _remap_grid_parse(
        source_data,
        source_grid,
        destination_grid,
        coord_type,
        remap_to,
        k=k,
        query=True,
    )

    weights = 1 / (distances**power + 1e-6)
    weights /= np.sum(weights, axis=1, keepdims=True)

    destination_data = np.sum(
        source_data[..., nearest_neighbor_indices] * weights, axis=-1
    )

    return destination_data


def _inverse_distance_weighted_remap_uxda(
    source_uxda: UxDataArray,
    destination_grid: Grid,
    remap_to: str = "face centers",
    coord_type: str = "spherical",
    power=2,
    k=8,
):
    """Inverse Distance Weighted Remapping implementation for ``UxDataArray``.

    Parameters
    ---------
    source_uxda : UxDataArray
        Source UxDataArray for remapping
    destination_grid : Grid
        Destination grid for remapping
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes", "edge centers", or "face centers"
    coord_type : str, default="spherical"
        Indicates whether to remap using on Spherical or Cartesian coordinates for the computations when
        remapping.
    power : int, default=2
        Power parameter for inverse distance weighting. This controls how local or global the remapping is, a higher
        power causes points that are further away to have less influence
    k : int, default=8
        Number of nearest neighbors to consider in the weighted calculation.
    """

    # check dimensions remapped to and from
    if (
        (source_uxda._node_centered() and remap_to != "nodes")
        or (source_uxda._face_centered() and remap_to != "face centers")
        or (source_uxda._edge_centered() and remap_to != "edge centers")
    ):
        warnings.warn(
            f"Your data is stored on {source_uxda.dims[-1]}, but you are remapping to {remap_to}"
        )

    # prepare dimensions
    if remap_to == "nodes":
        destination_dim = "n_node"
    elif remap_to == "face centers":
        destination_dim = "n_face"
    else:
        destination_dim = "n_edge"

    destination_dims = list(source_uxda.dims)
    destination_dims[-1] = destination_dim

    # perform remapping
    destination_data = _inverse_distance_weighted_remap(
        source_uxda.uxgrid,
        destination_grid,
        source_uxda.data,
        remap_to,
        coord_type,
        power,
        k,
    )

    # preserve only non-spatial coordinates
    destination_coords = deepcopy(source_uxda.coords)
    if destination_dim in destination_coords:
        del destination_coords[destination_dim]

    # construct data array for remapping variable
    uxda_remap = uxarray.core.dataarray.UxDataArray(
        data=destination_data,
        name=source_uxda.name,
        dims=destination_dims,
        uxgrid=destination_grid,
        coords=destination_coords,
    )

    # return UxDataArray with remapped variable
    return uxda_remap


def _inverse_distance_weighted_remap_uxds(
    source_uxds: UxDataset,
    destination_grid: Grid,
    remap_to: str = "face centers",
    coord_type: str = "spherical",
    power=2,
    k=8,
):
    """Inverse Distance Weighted implementation for ``UxDataset``.

    Parameters
    ---------
    source_uxds : UxDataset
        Source UxDataset for remapping
    destination_grid : Grid
        Destination for remapping
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes", "edge centers", or "face centers"
    coord_type : str, default="spherical"
        Indicates whether to remap using on Spherical or Cartesian coordinates
    power : int, default=2
        Power parameter for inverse distance weighting. This controls how local or global the remapping is, a higher
        power causes points that are further away to have less influence
    k : int, default=8
        Number of nearest neighbors to consider in the weighted calculation.
    """
    destination_uxds = uxarray.UxDataset(uxgrid=destination_grid)
    for var_name in source_uxds.data_vars:
        destination_uxds[var_name] = _inverse_distance_weighted_remap_uxda(
            source_uxds[var_name],
            destination_grid,
            remap_to,
            coord_type,
            power,
            k,
        )

    return destination_uxds
