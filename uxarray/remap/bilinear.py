from __future__ import annotations
from typing import TYPE_CHECKING

from numba import njit

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray

import numpy as np
from uxarray.constants import INT_FILL_VALUE, INT_DTYPE
import uxarray.core.dataarray
import uxarray.core.dataset
from uxarray.grid import Grid
from uxarray.grid.area import calculate_face_area
from uxarray.grid.coordinates import _xyz_to_lonlat_deg


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
        # Since currently `topological_mean` is not supported for edge centers raise a `ValueError`
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

    # Reload the data array after topological mean
    source_data = np.asarray(source_uxda.data)
    # Construct dual for searching
    dual = source_uxda.get_dual()

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
        # TODO: Find subset of potential polygons that contains point
        values = np.ndarray(data_size)
        if source_grid.hole_edge_indices.size == 0:
            for i in range(len(lon)):
                point = [lon[i], lat[i]]
                # Find a subset of polygons that contain the point
                polygons_subset = find_polygons_subset(dual, point)

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

        # TODO: Find subset of potential polygons that contains point
        values = np.ndarray(data_size)
        if source_grid.hole_edge_indices.size == 0:
            for i in range(len(cart_x)):
                # Convert xyz to lat lon to use in subset and weights calculation
                point = _xyz_to_lonlat_deg(cart_x[i], cart_y[i], cart_z[i])

                # Find a subset of polygons that contain the point
                weights, data = find_polygon_containing_point(point, dual, source_data)

                values[i] = np.sum(weights * data, axis=-1)

                # Search the subset to find which one contains the point
                # for polygon in polygons_subset:
                #     if point_in_polygon(polygon, point):
                #         if polygon.n_nodes == 3:
                #             # TODO: Get indices of the nodes of the polygon
                #             polygon_ind = None
                #             weights = calculate_bilinear_weights(polygon, point)
                #             values[i] = np.sum(weights * source_data[..., polygon_ind], axis=-1)
                #             break
                #         elif:
                #             polygon_triangle_split(polygon)

    else:
        raise ValueError(
            f"Invalid coord_type. Expected either 'spherical' or 'cartesian', but received {coord_type}"
        )

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


def calculate_bilinear_weights(point, triangle):
    """Calculates the bilinear weights for a point inside a triangle.

    Returns an array with 3 weights for each node of the triangle
    """

    # Find the area of the whole triangle
    x = np.array(
        [
            triangle[1][0],
            triangle[1][1],
            triangle[1][2],
        ],
        dtype=np.float64,
    )
    y = np.array(
        [
            triangle[0][0],
            triangle[0][1],
            triangle[0][2],
        ],
        dtype=np.float64,
    )

    z = x * 0
    area = calculate_face_area(x, y, z)

    # Find the area of sub triangle: point, vertex b, vertex c
    x_pbc = np.array([point[0], triangle[1][1], triangle[1][2]], dtype=np.float64)
    y_pbc = np.array([point[1], triangle[0][1], triangle[0][2]], dtype=np.float64)

    area_pbc = calculate_face_area(x_pbc, y_pbc, z)

    # Find the area of sub triangle: vertex a, point, vertex c
    x_apc = np.array([triangle[1][0], point[0], triangle[1][2]], dtype=np.float64)
    y_apc = np.array([triangle[0][0], point[1], triangle[0][2]], dtype=np.float64)

    area_apc = calculate_face_area(x_apc, y_apc, z)

    # Find the area of sub triangle: vertex a, vertex b, point
    x_abp = np.array([triangle[1][0], triangle[1][1], point[0]], dtype=np.float64)
    y_abp = np.array([triangle[0][0], triangle[0][1], point[1]], dtype=np.float64)

    area_abp = calculate_face_area(x_abp, y_abp, z)

    weight_a = area_pbc[0] / area[0]
    weight_b = area_apc[0] / area[0]
    weight_c = area_abp[0] / area[0]

    return np.array([weight_a, weight_b, weight_c], dtype=np.float64)


def find_polygons_subset(dual, point):
    """Find a subset of polygons to be searched for the polygon containing the
    point."""

    subset = dual.subset.nearest_neighbor(point, k=1, element="face centers")
    return subset


def polygon_triangle_split(polygon, point):
    """For a given polygon, split into triangles and find the one containing
    the point."""
    triangles = polygon.n_nodes - 2
    x = polygon.node_x.values
    y = polygon.node_y.values
    z = polygon.node_z.values
    values = polygon.values

    for j in range(0, triangles):
        node1 = np.array([x[0], y[0], z[0]], dtype=x.dtype)
        node2 = np.array([x[j + 1], y[j + 1], z[j + 1]], dtype=x.dtype)
        node3 = np.array([x[j + 2], y[j + 2], z[j + 2]], dtype=x.dtype)

        # TODO: Create point_inside_polygon() function
        # if point_inside_triangle([node1, node2, node3], point):
        #     return [node1, node2, node3], [values[0], values[j + 1], values[j + 2]]


def point_in_triangle(point, triangle, tolerance=1e-2):
    weights = calculate_bilinear_weights(point=point, triangle=triangle)

    # Check if weights are greater than or equal to -tolerance (i.e., close to zero)
    is_inside = (
        weights[0] >= -tolerance
        and weights[1] >= -tolerance
        and weights[2] >= -tolerance
    )

    return is_inside, weights


def find_polygon_containing_point(point, mesh, source_data):
    """Finds the polygon that contains a point."""

    # Get ball_tree
    tree = mesh.uxgrid.get_ball_tree(coordinates="face centers")

    # Create arrays to hold the lat/lon of first face
    lat = np.array(
        [INT_FILL_VALUE for _ in range(mesh.uxgrid.n_max_face_edges)], dtype=np.float64
    )
    lon = np.array(
        [INT_FILL_VALUE for _ in range(mesh.uxgrid.n_max_face_edges)], dtype=np.float64
    )

    # If the mesh is not partial
    if mesh.uxgrid.hole_edge_indices.size == 0:
        # First check the nearest face
        ind = [tree.query(point, k=1, return_distance=False)]
        data = []
        for j, node in enumerate(mesh.uxgrid.face_node_connectivity[ind[0]]):
            if node != INT_FILL_VALUE:
                lat[j] = mesh.uxgrid.node_lat[node.values].values
                lon[j] = mesh.uxgrid.node_lon[node.values].values

                data.append(source_data[node])

        triangle = np.array((lat, lon), dtype=np.float64)
        point_found, weights = point_in_triangle(point, triangle=triangle)

        # If found in first face, return weights
        if point_found:
            return weights, data

        # If the nearest face doesn't contain the point, continue to check nearest faces
        for i in range(2, mesh.uxgrid.n_face):
            # Create arrays to hold the lat/lon of the face
            lat = np.array(
                [INT_FILL_VALUE for _ in range(mesh.uxgrid.n_max_face_edges)],
                dtype=INT_DTYPE,
            )
            lon = np.array(
                [INT_FILL_VALUE for _ in range(mesh.uxgrid.n_max_face_edges)],
                dtype=INT_DTYPE,
            )

            # Query the tree for increasingly more neighbors if the polygon isn't found
            ind = tree.query(point, k=i, return_distance=False, sort_results=True)
            data = []
            # Get the lat/lon for the face
            for j, node in enumerate(mesh.uxgrid.face_node_connectivity[ind[i]]):
                if node != INT_FILL_VALUE:
                    lat[j] = mesh.uxgrid.node_lat[node]
                    lon[j] = mesh.uxgrid.node_lon[node]
                    data.append(source_data[node])

            # Check if the point is inside the polygon
            point_found, weights = point_in_triangle(point, [lat, lon])

            if point_found:
                return weights, data

    # If the mesh is partial, limit the search to the distance of the largest face radius of the mesh
    else:
        # First check the nearest face
        ind = [tree.query(point, k=1, return_distance=False)]

        for j, node in enumerate(mesh.uxgrid.face_node_connectivity[ind[0]]):
            if node != INT_FILL_VALUE:
                lat[j] = mesh.uxgrid.node_lat[node]
                lon[j] = mesh.uxgrid.node_lon[node]

        point_found, weights = point_in_triangle(point, [lat, lon])

        # If found in the first face, return weights
        if point_found:
            return weights

        # Find the largest face radius
        max_distance = get_max_face_radius(mesh)

        for i in range(2, mesh.uxgrid.n_face):
            # Query the tree for increasingly more neighbors
            d, ind = tree.query(point, k=i, return_distance=True, sort_results=True)

            # If the distance is outside the max distance the point could be in, the point is outside the partial grid
            if d > max_distance:
                return INT_FILL_VALUE, INT_FILL_VALUE

            lat = np.array(
                [INT_FILL_VALUE for _ in range(mesh.uxgrid.n_max_face_nodes)],
                dtype=INT_DTYPE,
            )
            lon = np.array(
                [INT_FILL_VALUE for _ in range(mesh.uxgrid.n_max_face_nodes)],
                dtype=INT_DTYPE,
            )

            for j, node in enumerate(mesh.uxgrid.face_node_connectivity[ind[i]]):
                if node != INT_FILL_VALUE:
                    lat[j] = mesh.uxgrid.node_lat[node]
                    lon[j] = mesh.uxgrid.node_lon[node]

            point_found, weights = point_in_triangle(point, [lat, lon])

            if point_found:
                return weights


def get_max_face_radius(mesh):
    # Parse all variables needed for `njit` functions
    face_node_connectivity = mesh.uxgrid.face_node_connectivity.values
    node_lats_rad = np.radians(mesh.uxgrid.node_lat.values)
    node_lons_rad = np.radians(mesh.uxgrid.node_lon.values)
    face_lats_rad = np.radians(mesh.uxgrid.face_lat.values)
    face_lons_rad = np.radians(mesh.uxgrid.face_lon.values)

    # Get the max distance
    max_distance = calculate_max_face_radius(
        face_node_connectivity,
        node_lats_rad,
        node_lons_rad,
        face_lats_rad,
        face_lons_rad,
    )

    return max_distance


@njit(cache=True)
def calculate_max_face_radius(
    face_node_connectivity, node_lats_rad, node_lons_rad, face_lats_rad, face_lons_rad
):
    """Finds the max face radius in the mesh."""

    # Array to store all distance of each face to it's furthest node.
    end_distances = np.zeros(len(face_node_connectivity))

    # Loop over each face and its nodes
    for ind, face in enumerate(face_node_connectivity):
        # Get the face lat/lon of this face
        face_lat = face_lats_rad[ind]
        face_lon = face_lons_rad[ind]

        # Get the node lat/lon of this face
        node_lat_rads = node_lats_rad[face]
        node_lon_rads = node_lons_rad[face]

        # Calculate Haversine distances for all nodes in this face
        distances = haversine_distance(node_lat_rads, node_lon_rads, face_lat, face_lon)

        # Store the max distance for this face
        end_distances[ind] = np.max(distances)

    # Return the maximum distance found across all faces
    return np.max(end_distances)


@njit(cache=True)
def haversine_distance(node_lats, node_lons, face_lat, face_lon):
    """Calculates the haversine distance."""
    dlat = node_lats - face_lat
    dlon = node_lons - face_lon
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(face_lat) * np.cos(node_lats) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c
