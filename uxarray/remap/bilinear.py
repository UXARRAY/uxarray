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

        tree = dual.uxgrid.get_ball_tree(
            coordinates="face centers", coordinate_system="spherical"
        )

        for i in range(len(lon)):
            # Find polygon containing point
            weights, data = find_polygon_containing_point(
                [lon[i], lat[i]], dual, source_data, tree
            )

            values[i] = np.sum(weights * data, axis=-1)

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
        tree = dual.uxgrid.get_ball_tree(
            coordinates="face centers", coordinate_system="spherical"
        )
        for i in range(len(cart_x)):
            # Convert xyz to lat lon to use in subset and weights calculation
            point = _xyz_to_lonlat_deg(cart_x[i], cart_y[i], cart_z[i])

            # Find a subset of polygons that contain the point
            weights, data = find_polygon_containing_point(
                point, dual, source_data, tree
            )

            values[i] = np.sum(weights * data, axis=-1)

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


@njit(cache=True)
def calculate_bilinear_weights(point, triangle):
    """Calculates the barycentric weights for a point inside a triangle.

    Args:
        point: A 2D point (x, y) inside the triangle.
        triangle: A 2D triangle with three vertices as [(x0, y0), (x1, y1), (x2, y2)].

    Returns:
        An array with 3 weights for each node of the triangle.
    """
    x1, y1 = triangle[0][1], triangle[0][0]
    x2, y2 = triangle[1][1], triangle[1][0]
    x3, y3 = triangle[2][1], triangle[2][0]

    px, py = point

    # Compute the denominator (2 * the signed area of the full triangle)
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

    if np.abs(denom) < 1e-10:
        raise ValueError("The triangle points are too close to being collinear.")

    # Compute barycentric weights (dA, dB, dC)
    weight_a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    weight_b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    weight_c = 1.0 - weight_a - weight_b  # Third barycentric coordinate

    return np.array([weight_a, weight_b, weight_c], dtype=np.float64)


def find_polygon_containing_point(point, mesh, source_data, tree):
    """Finds the polygon that contains a point."""

    # Create arrays to hold the lat/lon of first face
    triangle = np.zeros(
        (3, 2), dtype=np.float64
    )  # Array to store 3 vertices (lat, lon)

    # If the mesh is not partial
    if mesh.uxgrid.hole_edge_indices.size == 0:
        # First check the nearest face
        ind = [tree.query(point, k=1, return_distance=False)]
        data = []

        for j, node in enumerate(mesh.uxgrid.face_node_connectivity[ind[0]]):
            if node != INT_FILL_VALUE:
                lat = mesh.uxgrid.node_lat[node.values].values  # Latitude for the node
                lon = mesh.uxgrid.node_lon[node.values].values  # Longitude for the node

                tolerance = 1e-0
                if abs(lat - point[1]) <= tolerance and abs(lon - point[0]) <= tolerance:
                    return 1, source_data[node]

                triangle[j] = [lat, lon]  # Store the (lat, lon) pair in the triangle
                data.append(source_data[node])

        projection_center = [mesh.uxgrid.face_lat[ind[0]].values, mesh.uxgrid.face_lon[ind[0]].values]


        # Now, triangle contains 3 vertices with (lat, lon) pairs
        point_found = point_in_triangle_projected(
            [point[1], point[0]],
            triangle=triangle,
            projection_center=projection_center,
        )
        #print(triangle)
        # If found in first face, return weights
        if point_found:
            return calculate_bilinear_weights(point=point, triangle=triangle), data
        else:
            print(triangle, projection_center, [point[1], point[0]])
            return INT_FILL_VALUE, 0

        # Find the largest face radius
        max_distance = get_max_face_radius(mesh)

        # If the nearest face doesn't contain the point, continue to check nearest faces
        for i in range(2, mesh.uxgrid.n_face):
            triangle = np.zeros(
                (3, 2), dtype=np.float64
            )  # Array to store 3 vertices (lat, lon)

            # Query the tree for increasingly more neighbors if the polygon isn't found
            d, ind = tree.query(point, k=i, return_distance=True, sort_results=True)
            data = []

            # If the distance is outside the max distance the point could be in, the point is outside the partial grid
            if d[i - 1] > max_distance:
                return INT_FILL_VALUE, 0
            # Get the lat/lon for the face
            for j, node in enumerate(mesh.uxgrid.face_node_connectivity[ind[0]]):
                if node != INT_FILL_VALUE:
                    lat = mesh.uxgrid.node_lat[
                        node.values
                    ].values  # Latitude for the node
                    lon = mesh.uxgrid.node_lon[
                        node.values
                    ].values  # Longitude for the node

                    triangle[j] = [
                        lat,
                        lon,
                    ]  # Store the (lat, lon) pair in the triangle
                    data.append(source_data[node])

            point_found = point_in_triangle_projected(
                [point[1], point[0]],
                triangle=triangle,
                projection_center=[
                    mesh.uxgrid.face_lat[ind[0]].values,
                    mesh.uxgrid.face_lon[ind[0]].values,
                ],
            )

            if point_found:
                return calculate_bilinear_weights(point=point, triangle=triangle), data

    # If the mesh is partial, limit the search to the distance of the largest face radius of the mesh
    else:
        # First check the nearest face
        ind = [tree.query(point, k=1, return_distance=False)]
        data = []
        for j, node in enumerate(mesh.uxgrid.face_node_connectivity[ind[0]]):
            if node != INT_FILL_VALUE:
                triangle[j] = (
                    mesh.uxgrid.node_lat[node.values].values,
                    mesh.uxgrid.node_lon[node.values].values,
                )

                data.append(source_data[node])

        point_found = point_in_triangle_projected(
            [point[1], point[0]],
            triangle=triangle,
            projection_center=[
                mesh.uxgrid.face_lat[ind[0]].values,
                mesh.uxgrid.face_lon[ind[0]].values,
            ],
        )
        # If found in first face, return weights
        if point_found:
            return calculate_bilinear_weights(point=point, triangle=triangle), data
        else:
            print(triangle,
                  [
                mesh.uxgrid.face_lat[ind[0]].values,
                mesh.uxgrid.face_lon[ind[0]].values,
                ])
            return INT_FILL_VALUE, 0

        # Find the largest face radius
        max_distance = get_max_face_radius(mesh)

        for i in range(2, mesh.uxgrid.n_face):
            # Query the tree for increasingly more neighbors
            d, ind = tree.query(point, k=i, return_distance=True, sort_results=True)

            # If the distance is outside the max distance the point could be in, the point is outside the partial grid
            if d[i - 1] > max_distance:
                return INT_FILL_VALUE, 0

            lat = np.array(
                [INT_FILL_VALUE for _ in range(mesh.uxgrid.n_max_face_nodes)],
                dtype=INT_DTYPE,
            )
            lon = np.array(
                [INT_FILL_VALUE for _ in range(mesh.uxgrid.n_max_face_nodes)],
                dtype=INT_DTYPE,
            )
            data = []
            for j, node in enumerate(mesh.uxgrid.face_node_connectivity[ind[i]]):
                if node != INT_FILL_VALUE:
                    lat[j] = mesh.uxgrid.node_lat[node.values].values
                    lon[j] = mesh.uxgrid.node_lon[node.values].values

                    data.append(source_data[node])
            point_found = point_in_triangle_projected(
                [point[1], point[0]],
                triangle=triangle,
                projection_center=[
                    mesh.uxgrid.face_lat[ind[0]].values,
                    mesh.uxgrid.face_lon[ind[0]].values,
                ],
            )

            if point_found:
                return calculate_bilinear_weights(point=point, triangle=triangle), data
    return [0, 0, 0], 0


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


# Gnomonic projection function
@njit(cache=True)
def gnomonic_projection(lat, lon, lat0, lon0):
    """Project a point (lat, lon) onto a 2D plane using gnomonic projection
    from (lat0, lon0)."""

    # Convert degrees to radians
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)

    # Precompute cosines and sines
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lat0 = np.cos(lat0)
    sin_lat0 = np.sin(lat0)

    delta_lon = lon - lon0

    # Gnomonic projection equations
    denom = cos_lat0 * cos_lat * np.cos(delta_lon) + sin_lat0 * sin_lat
    x = cos_lat * np.sin(delta_lon) / denom
    y = (cos_lat0 * sin_lat - sin_lat0 * cos_lat * np.cos(delta_lon)) / denom

    return x, y


@njit(cache=True)
def project_triangle_to_plane(triangle, projection_center):
    """Project a triangle's vertices to a 2D plane using gnomonic
    projection."""
    projected_triangle = np.empty((3, 2), dtype=np.float64)

    for i in range(3):
        lat, lon = triangle[i]
        projected_triangle[i] = gnomonic_projection(
            lat, lon, projection_center[0], projection_center[1]
        )

    return projected_triangle


# Ray-casting function for 2D point-in-triangle test
@njit(cache=True)
def point_in_triangle_ray_casting(point, triangle):
    px, py = point
    intersection_count = 0

    # Triangle edges
    for i in range(3):
        x1, y1 = triangle[i]
        x2, y2 = triangle[(i + 1) % 3]

        # Check if the ray crosses the edge
        if (y1 > py) != (y2 > py):
            x_intersection = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x_intersection > px:
                intersection_count += 1

    return intersection_count % 2 == 1


@njit(cache=True)
def point_in_triangle_projected(point, triangle, projection_center):
    """Check if a point is inside a triangle on the sphere by projecting to a
    plane."""
    # Project the point and the triangle to the 2D plane
    projected_point = gnomonic_projection(
        point[0], point[1], projection_center[0], projection_center[1]
    )
    projected_triangle = project_triangle_to_plane(triangle, projection_center)

    # Apply the 2D ray-casting algorithm
    return point_in_triangle_ray_casting(projected_point, projected_triangle)
