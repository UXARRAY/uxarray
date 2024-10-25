from __future__ import annotations
from typing import TYPE_CHECKING

from uxarray.remap.utils import _remap_grid_parse

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray

import numpy as np

import uxarray.core.dataarray
import uxarray.core.dataset
from uxarray.grid import Grid

from copy import deepcopy


def _nearest_neighbor(
    source_grid: Grid,
    destination_grid: Grid,
    source_data: np.ndarray,
    remap_to: str = "face centers",
    coord_type: str = "spherical",
) -> np.ndarray:
    """Nearest Neighbor Remapping between two grids, mapping data that resides
    on the corner nodes, edge centers, or face centers on the source grid to
    the corner nodes, edge centers, or face centers of the destination grid.

    Parameters
    ---------
    source_grid : Grid
        Source grid that data is mapped to
    destination_grid : Grid
        Destination grid to remap data to
    source_data : np.ndarray
        Data variable to remaps
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes", "edge centers", or "face centers"
    coord_type: str, default="spherical"
        Coordinate type to use for nearest neighbor query, either "spherical" or "Cartesian"

    Returns
    -------
    destination_data : np.ndarray
        Data mapped to destination grid
    """

    # ensure array is a np.ndarray
    source_data = np.asarray(source_data)

    _, _, nearest_neighbor_indices = _remap_grid_parse(
        source_data,
        source_grid,
        destination_grid,
        coord_type,
        remap_to,
        k=1,
        query=True,
    )

    # support arbitrary dimension data using Ellipsis "..."
    destination_data = source_data[..., nearest_neighbor_indices]

    # case for 1D slice of data
    if source_data.ndim == 1:
        destination_data = destination_data.squeeze()

    return destination_data


def _nearest_neighbor_uxda(
    source_uxda: UxDataArray,
    destination_grid: Grid,
    remap_to: str = "face centers",
    coord_type: str = "spherical",
):
    """Nearest Neighbor Remapping implementation for ``UxDataArray``.

    Parameters
    ---------
    source_uxda : UxDataArray
        Source UxDataArray for remapping
    destination_grid : Grid
        Destination for remapping
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes", "edge centers", or "face centers"
    coord_type : str, default="spherical"
        Indicates whether to remap using on Spherical or Cartesian coordinates for nearest neighbor computations when
        remapping.
    """

    # prepare dimensions
    if remap_to == "nodes":
        destination_dim = "n_node"
    elif remap_to == "edge centers":
        destination_dim = "n_edge"
    else:
        destination_dim = "n_face"

    destination_dims = list(source_uxda.dims)
    destination_dims[-1] = destination_dim

    # perform remapping
    destination_data = _nearest_neighbor(
        source_uxda.uxgrid, destination_grid, source_uxda.data, remap_to, coord_type
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
    return uxda_remap


def _nearest_neighbor_uxds(
    source_uxds: UxDataset,
    destination_grid: Grid,
    remap_to: str = "face centers",
    coord_type: str = "spherical",
):
    """Nearest Neighbor Remapping implementation for ``UxDataset``.

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
    """
    destination_uxds = uxarray.UxDataset(uxgrid=destination_grid)
    for var_name in source_uxds.data_vars:
        destination_uxds[var_name] = _nearest_neighbor_uxda(
            source_uxds[var_name], destination_grid, remap_to, coord_type
        )

    return destination_uxds
