from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from uxarray.grid import Grid
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray

import numpy as np

import uxarray.core.dataarray
import uxarray.core.dataset
from uxarray.grid import Grid


def _nearest_neighbor(source_grid: Grid,
                      destination_grid: Grid,
                      source_data: np.ndarray,
                      remap_to: str = "nodes",
                      coord_type: str = "lonlat") -> np.ndarray:
    """Nearest Neighbor Remapping between two grids, mapping data that resides
    on either the corner nodes or face centers on the source grid to the corner
    nodes or face centers of the destination grid..

    Parameters
    ---------
    source_grid : Grid
        Source grid that data is mapped to
    destination_grid : Grid
        Destination grid to remap data to
    source_data : np.ndarray
        Data variable to remaps
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes" or "face centers"

    Returns
    -------
    destination_data : np.ndarray
        Data mapped to destination grid
    """

    # TODO: implementation in latlon, consider cartesiain once KDtree is implemented

    # ensure array is an np.ndarray
    source_data = np.asarray(source_data)

    n_elements = source_data.shape[-1]

    if n_elements == source_grid.n_node:
        source_data_mapping = "nodes"
    elif n_elements == source_grid.n_face:
        source_data_mapping = "face centers"
    else:
        raise ValueError(
            f"Invalid source_data shape. The final dimension should be either match the number of corner "
            f"nodes ({source_grid.n_node}) or face centers ({source_grid.n_face}) in the "
            f"source grid, but received: {source_data.shape}")

    if coord_type == "lonlat":
        # get destination coordinate pairs
        if remap_to == "nodes":
            lon, lat = destination_grid.node_lon.values, destination_grid.node_lat.values

        elif remap_to == "face centers":
            lon, lat = destination_grid.face_lon.values, destination_grid.face_lat.values
        else:
            raise ValueError(
                f"Invalid remap_to. Expected 'nodes' or 'face centers', "
                f"but received: {remap_to}")

        # specify whether to query on the corner nodes or face centers based on source grid
        _source_tree = source_grid.get_ball_tree(tree_type=source_data_mapping)

        # prepare coordinates for query
        lonlat = np.vstack([lon, lat]).T

        _, nearest_neighbor_indices = _source_tree.query(lonlat, k=1)

        # data values from source data to destination data using nearest neighbor indices
        if nearest_neighbor_indices.ndim > 1:
            nearest_neighbor_indices = nearest_neighbor_indices.squeeze()

        # support arbitrary dimension data using Ellipsis "..."
        destination_data = source_data[..., nearest_neighbor_indices]

        # case for 1D slice of data
        if source_data.ndim == 1:
            destination_data = destination_data.squeeze()

        return destination_data

    elif coord_type == "cartesian":
        # TODO: once a cartesian balltree/kdtree is implemented, implement this
        raise ValueError(
            f"Nearest Neighbor Remapping using Cartesian coordinates is not yet supported"
        )

    else:
        raise ValueError(
            f"Invalid coord_type. Expected either 'lonlat' or 'Cartesian', but received {coord_type}"
        )


def _nearest_neighbor_uxda(source_uxda: UxDataArray,
                           destination_obj: Union[Grid, UxDataArray, UxDataset],
                           remap_to: str = "nodes",
                           coord_type: str = "lonlat"):
    """Nearest Neighbor Remapping implementation for ``UxDataArray``.

    Parameters
    ---------
    source_uxda : UxDataArray
        Source UxDataArray for remapping
    destination_obj : Grid, UxDataArray, UxDataset
        Destination for remapping
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes" or "face centers"
    coord_type : str, default="lonlat"
        Indicates whether to remap using on latlon or Cartesian coordinates for nearest neighbor computations when
        remapping.
    """

    # prepare dimensions
    if remap_to == "nodes":
        destination_dim = "n_node"
    else:
        destination_dim = "n_face"

    destination_dims = list(source_uxda.dims)
    destination_dims[-1] = destination_dim

    if isinstance(destination_obj, Grid):
        destination_grid = destination_obj
    elif isinstance(
            destination_obj,
        (uxarray.core.dataarray.UxDataArray, uxarray.core.dataset.UxDataset)):
        destination_grid = destination_obj.uxgrid
    else:
        raise ValueError("TODO: Invalid Input")

    # perform remapping
    destination_data = _nearest_neighbor(source_uxda.uxgrid, destination_grid,
                                         source_uxda.data, remap_to, coord_type)
    # construct data array for remapping variable
    uxda_remap = uxarray.core.dataarray.UxDataArray(data=destination_data,
                                                    name=source_uxda.name,
                                                    dims=destination_dims,
                                                    uxgrid=destination_grid)
    # add remapped variable to existing UxDataset
    if isinstance(destination_obj, uxarray.core.dataset.UxDataset):
        destination_obj[source_uxda.name] = uxda_remap
        return destination_obj

    # construct a UxDataset from remapped variable and existing variable
    elif isinstance(destination_obj, uxarray.core.dataset.UxDataArray):
        uxds = destination_obj.to_dataset()
        uxds[source_uxda.name] = uxda_remap
        return uxds

    # return UxDataArray with remapped variable
    else:
        return uxda_remap


def _nearest_neighbor_uxds(source_uxds: UxDataset,
                           destination_obj: Union[Grid, UxDataArray, UxDataset],
                           remap_to: str = "nodes",
                           coord_type: str = "lonlat"):
    """Nearest Neighbor Remapping implementation for ``UxDataset``.

    Parameters
    ---------
    source_uxds : UxDataset
        Source UxDataset for remapping
    destination_obj : Grid, UxDataArray, UxDataset
        Destination for remapping
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes" or "face centers"
    coord_type : str, default="lonlat"
        Indicates whether to remap using on latlon or cartesiain coordinates
    """

    if isinstance(destination_obj, Grid):
        destination_uxds = uxarray.core.dataset.UxDataset(
            uxgrid=destination_obj)
    elif isinstance(destination_obj, uxarray.core.dataset.UxDataArray):
        destination_uxds = destination_obj.to_dataset()
    elif isinstance(destination_obj, uxarray.core.dataset.UxDataset):
        destination_uxds = destination_obj
    else:
        raise ValueError

    for var_name in source_uxds.data_vars:
        destination_uxds = _nearest_neighbor_uxda(source_uxds[var_name],
                                                  destination_uxds, remap_to,
                                                  coord_type)

    return destination_uxds
