from __future__ import annotations
from typing import TYPE_CHECKING

from numba import njit

from uxarray.grid.geometry import _point_in_polygon, point_in_polygon

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray

import numpy as np
from uxarray.constants import INT_FILL_VALUE, INT_DTYPE
import uxarray.core.dataarray
import uxarray.core.dataset
from uxarray.grid import Grid
from uxarray.grid.coordinates import _xyz_to_lonlat_deg, _lonlat_rad_to_xyz


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


# @njit(cache=True)
def barycentric_coordinates(point, triangle):
    """Calculates the barycentric weights for a point inside a triangle.

    Args:
        point: A 2D point (x, y) inside the triangle.
        triangle: A 2D triangle with three vertices as [(x0, y0), (x1, y1), (x2, y2)].

    Returns:
        An array with 3 weights for each node of the triangle.
    """
    x1, y1 = triangle[0][0], triangle[0][1]
    x2, y2 = triangle[1][0], triangle[1][1]
    x3, y3 = triangle[2][0], triangle[2][1]

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


def find_polygon_containing_point(point, dual, source_data, tree):
    """Finds the polygon that contains a point."""

    # Create arrays to hold the lat/lon of first face
    triangle = np.zeros(
        (3, 2), dtype=np.float64
    )  # Array to store 3 vertices (lat, lon)

    xyz = np.zeros(
        (3, 3), dtype=np.float64
    )  # Array to store 3 vertices (lat, lon)

    # If the mesh is not partial
    if dual.uxgrid.boundary_edge_indices.size == 0:
        # First check the nearest face
        ind = tree.query(point, k=1, return_distance=False)
        data = []
        polygon = np.zeros([len(dual.uxgrid.face_node_connectivity[ind].values), 3])
        for j, node in enumerate(dual.uxgrid.face_node_connectivity[ind]):
            if node != INT_FILL_VALUE:
                lon = dual.uxgrid.node_lon[node.values].values  # Longitude for the node
                lat = dual.uxgrid.node_lat[node.values].values  # Latitude for the node
                polygon[j] = [dual.uxgrid.node_x[node.values].values,
                              dual.uxgrid.node_y[node.values].values,
                              dual.uxgrid.node_z[node.values].values]

                tolerance = 1e-0
                if abs(lat - point[1]) <= tolerance and abs(lon - point[0]) <= tolerance:
                    return 1, source_data[node]

                triangle[j] = [lon, lat]  # Store the (lon, lat) pair in the triangle
                data.append(source_data[node])
        polygon2 = [
            _lonlat_rad_to_xyz(np.deg2rad(vertex[0]), np.deg2rad(vertex[1]))
            for vertex in triangle
        ]

        face_ind = (ind + 2) % len(dual.uxgrid.node_x.values)
        ref_point = np.array([0, 0, 1])
        point_cart = np.array(_lonlat_rad_to_xyz(np.deg2rad(point[0]), np.deg2rad(point[1])))

        point_found = point_in_polygon(polygon2, point_cart, ref_point)

        # If found in first face, return weights
        if point_found:
            return barycentric_coordinates(point=point, triangle=triangle), data
        else:

            # Find the largest face radius
            max_distance = get_max_face_radius(dual)

            # If the nearest face doesn't contain the point, continue to check nearest faces
            for i in range(2, dual.uxgrid.n_face):
                triangle = np.zeros(
                    (3, 2), dtype=np.float64
                )  # Array to store 3 vertices (lat, lon)

                # Query the tree for increasingly more neighbors if the polygon isn't found
                d, ind = tree.query(point, k=i, return_distance=True, sort_results=True)
                data = []

                # If the distance is outside the max distance the point could be in, the point is outside the partial
                # grid
                if d[i - 1] > max_distance:
                    return INT_FILL_VALUE, 0
                
                # Get the lat/lon for the face
                for j, node in enumerate(dual.uxgrid.face_node_connectivity[ind[0]]):
                    if node != INT_FILL_VALUE:
                        lat = dual.uxgrid.node_lat[node.values].values  # Latitude for the node
                        lon = dual.uxgrid.node_lon[node.values].values  # Longitude for the node
                        x = dual.uxgrid.node_x[node.values].values
                        y = dual.uxgrid.node_y[node.values].values
                        z = dual.uxgrid.node_z[node.values].values
                        tolerance = 1e-0
                        if abs(lat - point[1]) <= tolerance and abs(lon - point[0]) <= tolerance:
                            return 1, source_data[node]

                        triangle[j] = [lat, lon]  # Store the (lat, lon) pair in the triangle
                        xyz[j] = [x, y, z]
                        data.append(source_data[node])

                polygon2 = [
                    _lonlat_rad_to_xyz(np.deg2rad(vertex[0]), np.deg2rad(vertex[1]))
                    for vertex in triangle
                ]

                ref_point = np.array([0, 0, 1])
                point_cart = np.array(_lonlat_rad_to_xyz(np.deg2rad(point[0]), np.deg2rad(point[1])))

                point_found = point_in_polygon(polygon2, point_cart, ref_point)

                # If found in first face, return weights
                if point_found:
                    return barycentric_coordinates(point=point, triangle=triangle), data
        return 0, 0
    # If the mesh is partial, limit the search to the distance of the largest face radius of the mesh


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

