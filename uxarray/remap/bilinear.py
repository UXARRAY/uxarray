from __future__ import annotations
from typing import TYPE_CHECKING

from uxarray.grid.geometry import barycentric_coordinates_cartesian

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray

import numpy as np
import uxarray.core.dataarray
import uxarray.core.dataset
from uxarray.grid import Grid

from uxarray.constants import INT_FILL_VALUE


def _bilinear(
    source_uxda: UxDataArray,
    destination_grid: Grid,
    remap_to: str = "face centers",
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
    source_data = np.asarray(source_uxda.data)
    source_grid = source_uxda.uxgrid

    n_elements = source_data.shape[-1]

    # Find where the source data is located
    if n_elements == source_grid.n_node:
        source_data_mapping = "nodes"
    elif n_elements == source_grid.n_face:
        source_data_mapping = "face centers"
    elif n_elements == source_grid.n_edge:
        # Since currently `topological_mean` is not supported for edge centers, raise a `ValueError`
        raise ValueError(
            "'edges' is currently an unsupported source data dimension for bilinear remapping"
        )
    else:
        raise ValueError(
            f"Invalid source_data shape. The final dimension should be either match the number of corner "
            f"nodes ({source_grid.n_node}), edge centers ({source_grid.n_edge}), or face centers ({source_grid.n_face}) in the"
            f" source grid, but received: {source_data.shape}"
        )

    # If the data isn't face centered, take a `topological_mean` so the data will be face centered for the dual
    if source_data_mapping != "face centers":
        source_uxda = source_uxda.topological_mean(destination="face")

    # Ensure the destination grid is normalized
    destination_grid.normalize_cartesian_coordinates()

    # Reload the data array after topological mean
    source_data = np.asarray(source_uxda.data)
    # Construct dual for searching
    dual = source_uxda.get_dual()

    # get destination coordinate pairs
    if remap_to == "face centers":
        lon, lat = (
            destination_grid.face_lon.values,
            destination_grid.face_lat.values,
        )
        cart_x, cart_y, cart_z = (
            destination_grid.face_x.values,
            destination_grid.face_y.values,
            destination_grid.face_z.values,
        )
        data_size = destination_grid.n_face
    elif remap_to == "nodes":
        lon, lat = (
            destination_grid.node_lon.values,
            destination_grid.node_lat.values,
        )
        cart_x, cart_y, cart_z = (
            destination_grid.node_x.values,
            destination_grid.node_y.values,
            destination_grid.node_z.values,
        )
        data_size = destination_grid.n_node
    # TODO: Uncomment when edge support is added
    # elif remap_to == "edge centers":
    #     lon, lat = (
    #         destination_grid.edge_lon.values,
    #         destination_grid.edge_lat.values,
    #     )
    #     cart_x, cart_y, cart_z = (
    #         destination_grid.edge_x.values,
    #         destination_grid.edge_y.values,
    #         destination_grid.edge_z.values,
    #     )
    #     data_size = destination_grid.n_edge
    else:
        raise ValueError(
            f"Invalid remap_to. Expected 'nodes', 'edge centers', or 'face centers', "
            f"but received: {remap_to}"
        )
    # Get the point coordinates to use for search
    point_lonlat = np.column_stack([lon, lat])
    point_xyz = np.column_stack([cart_x, cart_y, cart_z])

    values = _get_values(
        point_xyz=point_xyz,
        point_lonlat=point_lonlat,
        dual=dual,
        source_data=source_data,
        data_size=data_size,
        source_grid=source_grid,
    )

    return values


def _bilinear_uxda(
    source_uxda: UxDataArray,
    destination_grid: Grid,
    remap_to: str = "face centers",
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
    destination_data = _bilinear(source_uxda, destination_grid, remap_to)

    # construct a data array for remapping variable
    uxda_remap = uxarray.core.dataarray.UxDataArray(
        data=destination_data,
        name=source_uxda.name,
        coords=source_uxda.coords,
        dims=destination_dims,
        uxgrid=destination_grid,
    )

    return uxda_remap


def _bilinear_uxds(
    source_uxds: UxDataset, destination_grid: Grid, remap_to: str = "face centers"
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
    """

    destination_uxds = uxarray.UxDataset(uxgrid=destination_grid)

    for var_name in source_uxds.data_vars:
        destination_uxds[var_name] = _bilinear_uxda(
            source_uxds[var_name], destination_grid, remap_to
        )

    return destination_uxds


def _get_values(point_xyz, point_lonlat, dual, source_data, data_size, source_grid):
    """Get the values for each point being remapped to by calculating and applying the weights"""
    values = np.zeros(data_size)

    for i in range(data_size):
        # Find the index of the polygon containing the point
        face_ind = dual.uxgrid.get_faces_containing_point(
            point_xyz=point_xyz[i], point_lonlat=point_lonlat[i]
        )

        number_of_faces = len(face_ind)

        if number_of_faces == 0:
            # Check to see if the point lies within the source grid instead
            face_ind = source_grid.get_faces_containing_point(
                point_xyz=point_xyz[i], point_lonlat=point_lonlat[i]
            )

            if len(face_ind) == 0:
                values[i] = 0
            else:
                values[i] = source_data[face_ind[0]]
        else:
            # Get the index of the face that holds the point
            node_ind = dual.uxgrid.face_node_connectivity[face_ind[0]].values

            # Create the polygon from the `face_node_connectivity`
            nodes_per_face = dual.uxgrid.n_nodes_per_face[face_ind[0]].values
            polygon = np.empty([nodes_per_face, 3])
            data = np.empty([nodes_per_face])
            for ind, node in enumerate(node_ind):
                if node == INT_FILL_VALUE:
                    break
                polygon[ind] = [
                    dual.uxgrid.node_x.values[node],
                    dual.uxgrid.node_y.values[node],
                    dual.uxgrid.node_z.values[node],
                ]

                # Create the data array on the polygon
                data[ind] = source_data[node]

            weights = barycentric_coordinates_cartesian(polygon, point_xyz[i])
            values[i] = np.sum(weights * data, axis=-1)

    return values
