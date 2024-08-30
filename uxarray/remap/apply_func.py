from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray

import numpy as np

import uxarray.core.dataarray
import uxarray.core.dataset
from uxarray.grid import Grid
import warnings


def _apply_func_remap(
    source_grid: Grid,
    destination_grid: Grid,
    source_data: np.ndarray,
    remap_to: str = "face centers",
    coord_type: str = "spherical",
    func: Callable = np.mean,
    r: float = 1.0,
) -> np.array:
    """Apply neighborhood function Remapping between two grids.

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
    r : float, default=1.
        radius of neighborhoodFor spherical coordinates, the radius is in units of degrees,
        and for cartesian coordinates, the radius is in meters.

    Returns:
    --------
    destination_data : np.ndarray
        Data mapped to the destination grid.
    """

    source_data = np.asarray(source_data)
    n_elements = source_data.shape[-1]

    if n_elements == source_grid.n_node:
        source_data_mapping = "nodes"
    elif n_elements == source_grid.n_face:
        source_data_mapping = "face centers"
    elif n_elements == source_grid.n_edge:
        source_data_mapping = "edge centers"
    else:
        raise ValueError(
            f"Invalid source_data shape. The final dimension should match the number of corner "
            f"nodes ({source_grid.n_node}), edge nodes ({source_grid.n_edge}), or face centers ({source_grid.n_face}) "
            f"in the source grid, but received: {source_data.shape}"
        )

    if coord_type == "spherical":
        if remap_to == "nodes":
            lon, lat = (
                destination_grid.node_lon.values,
                destination_grid.node_lat.values,
            )
        elif remap_to == "face centers":
            lon, lat = (
                destination_grid.face_lon.values,
                destination_grid.face_lat.values,
            )
        elif remap_to == "edge centers":
            lon, lat = (
                destination_grid.edge_lon.values,
                destination_grid.edge_lat.values,
            )
        else:
            raise ValueError(
                f"Invalid remap_to. Expected 'nodes', 'edge centers', or 'face centers', "
                f"but received: {remap_to}"
            )

        _source_tree = source_grid.get_ball_tree(coordinates=source_data_mapping)

        dest_coords = np.vstack([lon, lat]).T

        neighbor_indices = _source_tree.query_radius(dest_coords, r=r)

    elif coord_type == "cartesian":
        if remap_to == "nodes":
            x, y, z = (
                destination_grid.node_x.values,
                destination_grid.node_y.values,
                destination_grid.node_z.values,
            )
        elif remap_to == "face centers":
            x, y, z = (
                destination_grid.face_x.values,
                destination_grid.face_y.values,
                destination_grid.face_z.values,
            )
        elif remap_to == "edge centers":
            x, y, z = (
                destination_grid.edge_x.values,
                destination_grid.edge_y.values,
                destination_grid.edge_z.values,
            )
        else:
            raise ValueError(
                f"Invalid remap_to. Expected 'nodes', 'edge centers', or 'face centers', "
                f"but received: {remap_to}"
            )

        _source_tree = source_grid.get_ball_tree(
            coordinates=source_data_mapping,
            coordinate_system="cartesian",
            distance_metric="minkowski",
        )

        dest_coords = np.vstack([x, y, z]).T

        neighbor_indices = _source_tree.query_radius(dest_coords, r=r)

    else:
        raise ValueError(
            f"Invalid coord_type. Expected either 'spherical' or 'cartesian', but received {coord_type}"
        )

    # make destination_shape a list instead of immutable tuple
    destination_shape = list(source_data.shape)
    # last dimension has same number of elements as neighbor_indices list
    destination_shape[-1] = len(neighbor_indices)
    destination_data = np.empty(destination_shape)
    # Apply function to indices on last axis.
    for i, idx in enumerate(neighbor_indices):
        if len(idx):
            destination_data[..., i] = func(source_data[..., idx])

    return destination_data


def _apply_func_remap_uxda(
    source_uxda: UxDataArray,
    destination_grid: Grid,
    remap_to: str = "face centers",
    coord_type: str = "spherical",
    func: Callable = np.mean,
    r=1.0,
):
    """Neighborhood function Remapping implementation for ``UxDataArray``.

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
    r : float, default=1.
        Radius of neighborhood.
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
    destination_data = _apply_func_remap(
        source_uxda.uxgrid,
        destination_grid,
        source_uxda.data,
        remap_to,
        coord_type,
        func,
        r,
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


def _apply_func_remap_uxds(
    source_uxds: UxDataset,
    destination_grid: Grid,
    remap_to: str = "face centers",
    coord_type: str = "spherical",
    func: Callable = np.mean,
    r: float = 1.0,
):
    """Neighboohood function implementation for ``UxDataset``.

    Parameters
    ---------
    source_uxds : UxDataset
        Source UxDataset for remapping
    destination_grid : Grid
        Destination grid for remapping
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes", "edge centers", or "face centers"
    coord_type : str, default="spherical"
        Indicates whether to remap using on Spherical or Cartesian coordinates
    func : Callable = np.mean
        function to apply to neighborhood
    r : float, default=1.
        Radius of neighborhood in deg
    """

    destination_uxds = uxarray.core.dataset.UxDataset(uxgrid=destination_grid)
    for var_name in source_uxds.data_vars:
        destination_uxds[var_name] = _apply_func_remap_uxda(
            source_uxds[var_name],
            destination_uxds,
            remap_to,
            coord_type,
            func,
            r,
        )

    return destination_uxds
