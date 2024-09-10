from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray

import numpy as np

import uxarray.core.dataarray
import uxarray.core.dataset
from uxarray.grid import Grid


def _bilinear(
    source_uxda: UxDataArray,
    destination_grid: Grid,
    remap_to: str = "face centers",
    coord_type: str = "spherical",
) -> np.ndarray:
    """Bilinear Remapping between two grids, mapping data that resides
    on the corner nodes, edge centers, or face centers on the source grid to
    the corner nodes, edge centers, or face centers of the destination grid.

    Parameters
    ---------
    source_uxda : UxDataArray
        Source UxDataArray
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes", "edge centers", or "face centers"
    coord_type: str, default="spherical"
        Coordinate type to use for bilinear query, either "spherical" or "Cartesian"

    Returns
    -------
    destination_data : np.ndarray
        Data mapped to destination grid
    """

    # ensure array is a np.ndarray
    source_data = np.asarray(source_uxda.data)
    source_grid = source_uxda.uxgrid

    n_elements = source_data.shape[-1]

    # Construct dual for searching
    dual = source_uxda.get_dual()

    if n_elements == source_grid.n_node:
        source_data_mapping = "nodes"
    elif n_elements == source_grid.n_edge:
        source_data_mapping = "edge centers"
    elif n_elements == source_grid.n_face:
        source_data_mapping = "face centers"
    else:
        raise ValueError(
            f"Invalid source_data shape. The final dimension should be either match the number of corner "
            f"nodes ({source_grid.n_node}), edge centers ({source_grid.n_edge}), or face centers ({source_grid.n_face}) in the"
            f" source grid, but received: {source_data.shape}"
        )

    if coord_type == "spherical":
        # get destination coordinate pairs
        if remap_to == "nodes":
            lon, lat = (
                destination_grid.node_lon.values,
                destination_grid.node_lat.values,
            )
        elif remap_to == "edge centers":
            lon, lat = (
                destination_grid.edge_lon.values,
                destination_grid.edge_lat.values,
            )
        elif remap_to == "face centers":
            lon, lat = (
                destination_grid.face_lon.values,
                destination_grid.face_lat.values,
            )
        else:
            raise ValueError(
                f"Invalid remap_to. Expected 'nodes', 'edge centers', or 'face centers', "
                f"but received: {remap_to}"
            )
        # TODO: Find dual polygon that contains point

        # TODO: Get bilinear weights

    elif coord_type == "cartesian":
        # get destination coordinates
        if remap_to == "nodes":
            cart_x, cart_y, cart_z = (
                destination_grid.node_x.values,
                destination_grid.node_y.values,
                destination_grid.node_z.values,
            )
        elif remap_to == "edge centers":
            cart_x, cart_y, cart_z = (
                destination_grid.edge_x.values,
                destination_grid.edge_y.values,
                destination_grid.edge_z.values,
            )
        elif remap_to == "face centers":
            cart_x, cart_y, cart_z = (
                destination_grid.face_x.values,
                destination_grid.face_y.values,
                destination_grid.face_z.values,
            )
        else:
            raise ValueError(
                f"Invalid remap_to. Expected 'nodes', 'edge centers', or 'face centers', "
                f"but received: {remap_to}"
            )
        # TODO: Find dual polygon that contains point

        # TODO: Get bilinear weights

    else:
        raise ValueError(
            f"Invalid coord_type. Expected either 'spherical' or 'cartesian', but received {coord_type}"
        )

    # TODO: Apply bilinear weights to destination data
    destination_data = None

    return destination_data


def _bilinear_uxda(
    source_uxda: UxDataArray,
    destination_grid: Grid,
    remap_to: str = "face centers",
    coord_type: str = "spherical",
):
    """Bilinear Remapping implementation for ``UxDataArray``.

    Parameters
    ---------
    source_uxda : UxDataArray
        Source UxDataArray for remapping
    destination_grid : Grid
        Destination Grid for remapping
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes", "edge centers", or "face centers"
    coord_type : str, default="spherical"
        Indicates whether to remap using on Spherical or Cartesian coordinates for computations when
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

    if not isinstance(destination_grid, Grid):
        raise ValueError("TODO: Invalid Input")

    # perform remapping
    destination_data = _bilinear(
        source_uxda, destination_grid, remap_to, coord_type
    )
    # construct data array for remapping variable
    uxda_remap = uxarray.core.dataarray.UxDataArray(
        data=destination_data,
        name=source_uxda.name,
        coords=source_uxda.coords,
        dims=destination_dims,
        uxgrid=destination_grid,
    )

    return uxda_remap


def _bilinear_uxds(
    source_uxds: UxDataset,
    destination_grid: Grid,
    remap_to: str = "face centers",
    coord_type: str = "spherical",
):
    """Bilinear Remapping implementation for ``UxDataset``.

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

    if not isinstance(destination_grid, Grid):
        raise ValueError
    else:
        destination_uxds = uxarray.core.dataset.UxDataset(uxgrid=destination_grid)

    for var_name in source_uxds.data_vars:
        destination_uxds = _bilinear_uxda(
            source_uxds[var_name], destination_grid, remap_to, coord_type
        )

    return destination_uxds
