from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray

import numpy as np

import uxarray.core.dataarray
import uxarray.core.dataset
from uxarray.grid import Grid
from uxarray.grid.area import calculate_face_area

from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _normalize_xyz, _xyz_to_lonlat_rad


def _bilinear(
        source_uxda: UxDataArray,
        destination_grid: Grid,
        remap_to: str = "face centers",
        coord_type: str = "spherical",
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
    coord_type: str, default="spherical"
        Coordinate type to use for bilinear query, either "spherical" or "Cartesian"

    Returns
    -------
    destination_data : np.ndarray
        Data mapped to destination grid
    """

    # TODO: The source data has to be on the face centers, if it is not a topological aggregation needs to be done

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
            data_size = destination_grid.n_node
        elif remap_to == "edge centers":
            lon, lat = (
                destination_grid.edge_lon.values,
                destination_grid.edge_lat.values,
            )
            data_size = destination_grid.n_edge
        elif remap_to == "face centers":
            lon, lat = (
                destination_grid.face_lon.values,
                destination_grid.face_lat.values,
            )
            data_size = destination_grid.n_face
        else:
            raise ValueError(
                f"Invalid remap_to. Expected 'nodes', 'edge centers', or 'face centers', "
                f"but received: {remap_to}"
            )
        # TODO: Find dual polygon that contains point
        # TODO: Create `calculate_bilinear_weights`
        values = np.ndarray(data_size)
        if source_grid.hole_edge_indices.size == 0:
            for i in range(len(lon)):
                point = [lon[i], lat[i]]
                # Find a subset of polygons that contain the point
                polygons_subset = find_polygons_subset(dual, point)
                print(polygons_subset)
                weights = calculate_bilinear_weights(polygons_subset, point)
                values[i] = np.sum(weights * polygons_subset.values, axis=-1)
                # Search the subset to find which one contains the point
                # for polygon in polygons_subset:
                #     if point_in_polygon(polygon, point):
                #         # TODO: Get indices of the nodes of the polygon
                #         polygon_ind = None
                #         weights = calculate_bilinear_weights(polygon, point)
                #         values[i] = np.sum(weights * source_data[..., polygon_ind], axis=-1)
                #         break

    elif coord_type == "cartesian":
        # get destination coordinates
        if remap_to == "nodes":
            cart_x, cart_y, cart_z = (
                destination_grid.node_x.values,
                destination_grid.node_y.values,
                destination_grid.node_z.values,
            )
            data_size = destination_grid.n_node
        elif remap_to == "edge centers":
            cart_x, cart_y, cart_z = (
                destination_grid.edge_x.values,
                destination_grid.edge_y.values,
                destination_grid.edge_z.values,
            )
            data_size = destination_grid.n_edge
        elif remap_to == "face centers":
            cart_x, cart_y, cart_z = (
                destination_grid.face_x.values,
                destination_grid.face_y.values,
                destination_grid.face_z.values,
            )
            data_size = destination_grid.n_face
        else:
            raise ValueError(
                f"Invalid remap_to. Expected 'nodes', 'edge centers', or 'face centers', "
                f"but received: {remap_to}"
            )
        # TODO: Find dual polygon that contains point

        # TODO: Get bilinear weights
        # TODO: Find dual polygon that contains point
        # TODO: Create `calculate_bilinear_weights`
        values = np.ndarray(data_size)
        if source_grid.hole_edge_indices.size == 0:
            for i in range(len(cart_x)):
                point = [cart_x[i], cart_y[i], cart_z[i]]

                # Find a subset of polygons that contain the point
                polygons_subset = find_polygons_subset(dual, point)
                # print(polygons_subset.uxgrid.node_lon.values, polygons_subset.uxgrid.node_lat.values,
                #       point)

                weights = calculate_bilinear_weights(polygons_subset, point)
                values[i] = np.sum(weights * polygons_subset.values, axis=-1)
                # Search the subset to find which one contains the point
                # for polygon in polygons_subset:
                #     if point_in_polygon(polygon, point):
                #         # TODO: Get indices of the nodes of the polygon
                #         polygon_ind = None
                #         weights = calculate_bilinear_weights(polygon, point)
                #         values[i] = np.sum(weights * source_data[..., polygon_ind], axis=-1)
                #         break

    else:
        raise ValueError(
            f"Invalid coord_type. Expected either 'spherical' or 'cartesian', but received {coord_type}"
        )

    # TODO: Apply bilinear weights to destination data

    return values


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

    # perform remapping
    destination_data = _bilinear(source_uxda, destination_grid, remap_to, coord_type)
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

    destination_uxds = uxarray.core.dataset.UxDataset(uxgrid=destination_grid)

    for var_name in source_uxds.data_vars:
        destination_uxds = _bilinear_uxda(
            source_uxds[var_name], destination_grid, remap_to, coord_type
        )

    return destination_uxds


def calculate_bilinear_weights(polygon, point):
    """Calculates the bilinear weights for a point inside a triangle.

    Returns an array with 3 weights for each node of the triangle
    """
    if len(point) == 2:
        point = _lonlat_rad_to_xyz(point[1], point[0])

    # Find the area of the whole triangle
    x = np.array(
        [polygon.uxgrid.node_x.values[0], polygon.uxgrid.node_x.values[1], polygon.uxgrid.node_x.values[2]]
    )
    y = np.array(
        [polygon.uxgrid.node_y.values[0], polygon.uxgrid.node_y.values[1], polygon.uxgrid.node_y.values[2]]
    )
    z = np.array(
        [polygon.uxgrid.node_z.values[0], polygon.uxgrid.node_z.values[1], polygon.uxgrid.node_z.values[2]]
    )
    area = calculate_face_area(x, y, z)

    # Find the area of sub triangle: point, vertex b, vertex c
    x_pbc = np.array([point[0], polygon.uxgrid.node_x.values[1], polygon.uxgrid.node_x.values[2]])
    y_pbc = np.array([point[1], polygon.uxgrid.node_y.values[1], polygon.uxgrid.node_y.values[2]])
    z_pbc = np.array([point[2], polygon.uxgrid.node_z.values[1], polygon.uxgrid.node_z.values[2]])
    area_pbc = calculate_face_area(x_pbc, y_pbc, z_pbc)

    # Find the area of sub triangle: vertex a, point, vertex c
    x_apc = np.array([polygon.uxgrid.node_x.values[0], point[0], polygon.uxgrid.node_x.values[2]])
    y_apc = np.array([polygon.uxgrid.node_y.values[0], point[1], polygon.uxgrid.node_y.values[2]])
    z_apc = np.array([polygon.uxgrid.node_z.values[0], point[2], polygon.uxgrid.node_z.values[2]])
    area_apc = calculate_face_area(x_apc, y_apc, z_apc)

    # Find the area of sub triangle: vertex a, vertex b, point
    x_abp = np.array([polygon.uxgrid.node_x.values[0], polygon.uxgrid.node_x.values[1], point[0]])
    y_abp = np.array([polygon.uxgrid.node_y.values[0], polygon.uxgrid.node_y.values[1], point[1]])
    z_abp = np.array([polygon.uxgrid.node_z.values[0], polygon.uxgrid.node_z.values[1], point[2]])
    area_abp = calculate_face_area(x_abp, y_abp, z_abp)

    weight_a = area_pbc[0] / area[0]
    weight_b = area_apc[0] / area[0]
    weight_c = area_abp[0] / area[0]
    print(f"{weight_a + weight_b + weight_c} weight")
    return np.array([weight_a, weight_b, weight_c])


def find_polygons_subset(dual, point):
    """Find a subset of polygons to be searched for the polygon containing the point"""

    point = _xyz_to_lonlat_rad(point[0], point[1], point[2])

    subset = dual.subset.nearest_neighbor(point, k=1,
                                          element='face centers')
    return subset
