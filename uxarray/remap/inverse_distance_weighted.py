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


def _inverse_distance_weighted_remap(
    source_grid,
    destination_grid,
    source_data,
    remap_to="face centers",
    coord_type="spherical",
    power=2,
    k=8,
):
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
        warnings.warn("It is recommended not to exceed a power of 5.0.")
    if k > source_grid.n_node:
        raise ValueError(
            f"Number of nearest neighbors to be used in the calculation is {k}, but should not exceed the "
            f"number of nodes in the source grid of {source_grid.n_node}"
        )
    if k <= 1:
        raise ValueError(
            f"Number of nearest neighbors to be used in the calculation is {k}, but should be greater than 1"
        )

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

        distances, nearest_neighbor_indices = _source_tree.query(dest_coords, k=k)

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

        distances, nearest_neighbor_indices = _source_tree.query(dest_coords, k=k)

    else:
        raise ValueError(
            f"Invalid coord_type. Expected either 'spherical' or 'cartesian', but received {coord_type}"
        )

    if nearest_neighbor_indices.ndim > 1:
        nearest_neighbor_indices = nearest_neighbor_indices.squeeze()

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
    # construct data array for remapping variable
    uxda_remap = uxarray.core.dataarray.UxDataArray(
        data=destination_data,
        name=source_uxda.name,
        coords=source_uxda.coords,
        dims=destination_dims,
        uxgrid=destination_grid,
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
