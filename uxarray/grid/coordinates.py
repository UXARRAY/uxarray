import xarray as xr
import numpy as np
import warnings
import math

from uxarray.conventions import ugrid

from numba import njit
from uxarray.constants import ERROR_TOLERANCE
from typing import Union

from uxarray.grid.utils import _angle_of_2_vectors


@njit(cache=True)
def _lonlat_rad_to_xyz(
    lon: Union[np.ndarray, float],
    lat: Union[np.ndarray, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts Spherical latitude and longitude coordinates into Cartesian x,
    y, z coordinates."""
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)

    return x, y, z


@njit(cache=True)
def _xyz_to_lonlat_rad_no_norm(
    x: Union[np.ndarray, float],
    y: Union[np.ndarray, float],
    z: Union[np.ndarray, float],
):
    """Converts a Cartesian x,y,z coordinates into Spherical latitude and
    longitude without normalization, decorated with Numba.

    Parameters
    ----------
    x : float
        Cartesian x coordinate
    y: float
        Cartesiain y coordinate
    z: float
        Cartesian z coordinate


    Returns
    -------
    lon : float
        Longitude in radians
    lat: float
        Latitude in radians
    """

    lon = np.atan2(y, x)
    lat = np.asin(z)

    # set longitude range to [0, pi]
    lon = np.mod(lon, 2 * np.pi)

    z_mask = np.abs(z) > 1.0 - ERROR_TOLERANCE

    lat = np.where(z_mask, np.sign(z) * np.pi / 2, lat)
    lon = np.where(z_mask, 0.0, lon)

    return lon, lat


@njit(cache=True)
def _xyz_to_lonlat_rad_scalar(x, y, z, normalize=True):
    if normalize:
        x, y, z = _normalize_xyz_scalar(x, y, z)
        denom = abs(x * x + y * y + z * z)
        x /= denom
        y /= denom
        z /= denom

    lon = np.atan2(y, x)
    lat = np.asin(z)

    # Set longitude range to [0, 2*pi]
    lon = lon % (2 * math.pi)

    z_abs = abs(z)
    if z_abs > (1.0 - ERROR_TOLERANCE):
        lat = math.copysign(math.pi / 2, z)
        lon = 0.0

    lonlat = np.empty(2)
    lonlat[0] = lon
    lonlat[1] = lat

    return lonlat


def _xyz_to_lonlat_rad(
    x: Union[np.ndarray, float],
    y: Union[np.ndarray, float],
    z: Union[np.ndarray, float],
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts Cartesian x, y, z coordinates in Spherical latitude and
    longitude coordinates in degrees.

    Parameters
    ----------
    x : Union[np.ndarray, float]
        Cartesian x coordinates
    y: Union[np.ndarray, float]
        Cartesiain y coordinates
    z: Union[np.ndarray, float]
        Cartesian z coordinates
    normalize: bool
        Flag to select whether to normalize the coordinates

    Returns
    -------
    lon : Union[np.ndarray, float]
        Longitude in radians
    lat: Union[np.ndarray, float]
        Latitude in radians
    """

    if normalize:
        x, y, z = _normalize_xyz(x, y, z)
        denom = np.abs(x * x + y * y + z * z)
        x /= denom
        y /= denom
        z /= denom

    lon = np.arctan2(y, x)
    lat = np.arcsin(z)

    # set longitude range to [0, pi]
    lon = np.mod(lon, 2 * np.pi)

    z_mask = np.abs(z) > 1.0 - ERROR_TOLERANCE

    lat = np.where(z_mask, np.sign(z) * np.pi / 2, lat)
    lon = np.where(z_mask, 0.0, lon)

    return lon, lat


def _xyz_to_lonlat_deg(
    x: Union[np.ndarray, float],
    y: Union[np.ndarray, float],
    z: Union[np.ndarray, float],
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts Cartesian x, y, z coordinates in Spherical latitude and
    longitude coordinates in degrees.

    Parameters
    ----------
    x : Union[np.ndarray, float]
        Cartesian x coordinates
    y: Union[np.ndarray, float]
        Cartesiain y coordinates
    z: Union[np.ndarray, float]
        Cartesian z coordinates
    normalize: bool
        Flag to select whether to normalize the coordinates

    Returns
    -------
    lon : Union[np.ndarray, float]
        Longitude in degrees
    lat: Union[np.ndarray, float]
        Latitude in degrees
    """
    lon_rad, lat_rad = _xyz_to_lonlat_rad(x, y, z, normalize=normalize)

    lon = np.rad2deg(lon_rad)
    lat = np.rad2deg(lat_rad)

    lon = (lon + 180) % 360 - 180
    return lon, lat


def _normalize_xyz(
    x: Union[np.ndarray, float],
    y: Union[np.ndarray, float],
    z: Union[np.ndarray, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalizes a set of Cartesiain coordinates."""
    denom = np.linalg.norm(
        np.asarray(np.array([x, y, z]), dtype=np.float64), ord=2, axis=0
    )

    x_norm = x / denom
    y_norm = y / denom
    z_norm = z / denom
    return x_norm, y_norm, z_norm


@njit(cache=True)
def _normalize_xyz_scalar(x: float, y: float, z: float):
    denom = np.linalg.norm(np.asarray(np.array([x, y, z]), dtype=np.float64), ord=2)
    x_norm = x / denom
    y_norm = y / denom
    z_norm = z / denom
    return x_norm, y_norm, z_norm


def _populate_node_latlon(grid) -> None:
    """Populates the lon and lat coordinates of a Grid (`node_lon`,
    `node_lat`)"""
    lon_rad, lat_rad = _xyz_to_lonlat_rad(
        grid.node_x.values, grid.node_y.values, grid.node_z.values
    )

    lon = np.rad2deg(lon_rad)
    lat = np.rad2deg(lat_rad)

    grid._ds["node_lon"] = xr.DataArray(
        data=lon, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LON_ATTRS
    )
    grid._ds["node_lat"] = xr.DataArray(
        data=lat, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LAT_ATTRS
    )


def _populate_node_xyz(grid) -> None:
    """Populates the Cartesiain node coordinates of a Grid (`node_x`, `node_y`
    and `node_z`)"""

    node_lon_rad = np.deg2rad(grid.node_lon.values)
    node_lat_rad = np.deg2rad(grid.node_lat.values)
    x, y, z = _lonlat_rad_to_xyz(node_lon_rad, node_lat_rad)

    grid._ds["node_x"] = xr.DataArray(
        data=x, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_X_ATTRS
    )
    grid._ds["node_y"] = xr.DataArray(
        data=y, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Y_ATTRS
    )
    grid._ds["node_z"] = xr.DataArray(
        data=z, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Z_ATTRS
    )


def _populate_face_centroids(grid, repopulate=False):
    """Finds the centroids of faces using cartesian averaging based off the
    vertices. The centroid is defined as the average of the x, y, z
    coordinates, normalized. This cannot be guaranteed to work on concave
    polygons.

    Parameters
    ----------
    repopulate : bool, optional
        Bool used to turn on/off repopulating the face coordinates of the centroids
    """
    warnings.warn("This cannot be guaranteed to work correctly on concave polygons")

    node_x = grid.node_x.values
    node_y = grid.node_y.values
    node_z = grid.node_z.values
    face_nodes = grid.face_node_connectivity.values
    n_nodes_per_face = grid.n_nodes_per_face.values

    if "face_lon" not in grid._ds or repopulate:
        # Construct the centroids if there are none stored
        if "face_x" not in grid._ds:
            centroid_x, centroid_y, centroid_z = _construct_face_centroids(
                node_x, node_y, node_z, face_nodes, n_nodes_per_face
            )

        else:
            # If there are cartesian centroids already use those instead
            centroid_x, centroid_y, centroid_z = grid.face_x, grid.face_y, grid.face_z

        # Convert from xyz to latlon TODO
        centroid_lon, centroid_lat = _xyz_to_lonlat_deg(
            centroid_x, centroid_y, centroid_z, normalize=False
        )
    else:
        # Convert to xyz if there are latlon centroids already stored
        centroid_lon, centroid_lat = grid.face_lon.values, grid.face_lat.values
        centroid_x, centroid_y, centroid_z = _lonlat_rad_to_xyz(
            centroid_lon, centroid_lat
        )

    # Populate the centroids
    if "face_lon" not in grid._ds or repopulate:
        grid._ds["face_lon"] = xr.DataArray(
            centroid_lon, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_LON_ATTRS
        )
        grid._ds["face_lat"] = xr.DataArray(
            centroid_lat, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_LAT_ATTRS
        )

    if "face_x" not in grid._ds or repopulate:
        grid._ds["face_x"] = xr.DataArray(
            centroid_x, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_X_ATTRS
        )

        grid._ds["face_y"] = xr.DataArray(
            centroid_y, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_Y_ATTRS
        )

        grid._ds["face_z"] = xr.DataArray(
            centroid_z, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_Z_ATTRS
        )


def _construct_face_centroids(node_x, node_y, node_z, face_nodes, n_nodes_per_face):
    """Constructs the xyz centroid coordinate for each face using Cartesian
    Averaging.

    Parameters
    ----------
    node_x : numpy.ndarray
        X coordinates of the nodes.
    node_y : numpy.ndarray
        Y coordinates of the nodes.
    node_z : numpy.ndarray
        Z coordinates of the nodes.
    face_nodes : numpy.ndarray
        Indices of nodes per face.
    n_nodes_per_face : numpy.ndarray
        Number of nodes per face.

    Returns
    -------
    tuple
        The x, y, and z coordinates of the centroids.
    """
    centroid_x = np.zeros((face_nodes.shape[0]), dtype=np.float64)
    centroid_y = np.zeros((face_nodes.shape[0]), dtype=np.float64)
    centroid_z = np.zeros((face_nodes.shape[0]), dtype=np.float64)

    for face_idx, n_max_nodes in enumerate(n_nodes_per_face):
        # Compute Cartesian Average
        centroid_x[face_idx] = np.mean(node_x[face_nodes[face_idx, 0:n_max_nodes]])
        centroid_y[face_idx] = np.mean(node_y[face_nodes[face_idx, 0:n_max_nodes]])
        centroid_z[face_idx] = np.mean(node_z[face_nodes[face_idx, 0:n_max_nodes]])

    return _normalize_xyz(centroid_x, centroid_y, centroid_z)


def _welzl_recursive(points, boundary, R):
    """Recursive helper function for Welzl's algorithm to find the smallest
    enclosing circle.

    Parameters
    ----------
    points : numpy.ndarray
        The set of points to consider.
    boundary : numpy.ndarray
        The current boundary points of the minimal enclosing circle.
    R : tuple
        The current minimal enclosing circle.

    Returns
    -------
    tuple
        The smallest enclosing circle as a tuple of center and radius.
    """
    # Base case: no points or boundary has 3 points
    if len(points) == 0 or len(boundary) == 3:
        # Construct the minimal circle based on the number of boundary points
        if len(boundary) == 0:
            # Return a default circle if no boundary points are available
            return ((0.0, 0.0), 0.0)
        elif len(boundary) == 1:
            return _circle_from_two_points(boundary[0], boundary[0])
        elif len(boundary) == 2:
            return _circle_from_two_points(boundary[0], boundary[1])
        elif len(boundary) == 3:
            return _circle_from_three_points(boundary[0], boundary[1], boundary[2])

    p = points[-1]
    temp_points = points[:-1]
    circle = _welzl_recursive(temp_points, boundary, R)

    # Check if the chosen point is inside the current circle
    if circle and _is_inside_circle(circle, p):
        return circle
    # If not, the point must be on the boundary of the minimal enclosing circle
    else:
        new_boundary = np.empty(
            (boundary.shape[0] + 1, boundary.shape[1]), dtype=boundary.dtype
        )
        new_boundary[:-1] = boundary
        new_boundary[-1] = p
        return _welzl_recursive(temp_points, new_boundary, R)


def _smallest_enclosing_circle(points):
    """Find the smallest circle that encloses all given points on a unit sphere
    using Welzl's algorithm.

    Parameters
    ----------
    points : numpy.ndarray
        An array of points as tuples of (lon, lat).

    Returns
    -------
    tuple
        The smallest enclosing circle as a tuple of center and radius.
    """
    np.random.shuffle(
        points
    )  # Randomize the input to increase the chance of an optimal solution
    return _welzl_recursive(points, np.empty((0, 2)), None)


@njit(cache=True)
def _circle_from_two_points(p1, p2):
    """Calculate the smallest circle that encloses two points on a unit sphere.

    Parameters
    ----------
    p1 : tuple
        The first point as a tuple of (latitude, longitude).
    p2 : tuple
        The second point as a tuple of (latitude, longitude).

    Returns
    -------
    tuple
        A tuple containing the center (as a tuple of lon and lat) and the radius of the circle.
    """
    center_lon = (p1[0] + p2[0]) / 2
    center_lat = (p1[1] + p2[1]) / 2
    center = (center_lon, center_lat)

    v1 = np.array(_lonlat_rad_to_xyz(np.radians(p1[0]), np.radians(p1[1])))
    v2 = np.array(_lonlat_rad_to_xyz(np.radians(p2[0]), np.radians(p2[1])))

    distance = _angle_of_2_vectors(v1, v2)
    radius = distance / 2

    return center, radius


@njit(cache=True)
def _circle_from_three_points(p1, p2, p3):
    """Calculate the smallest circle that encloses three points on a unit
    sphere. This is a placeholder implementation.

    Parameters
    ----------
    p1 : tuple
        The first point.
    p2 : tuple
        The second point.
    p3 : tuple
        The third point.

    Returns
    -------
    tuple
        A tuple containing the center (as a tuple of lon and lat) and the radius of the circle.
    """
    # Placeholder implementation for three-point circle calculation
    center_lon = (p1[0] + p2[0] + p3[0]) / 3
    center_lat = (p1[1] + p2[1] + p3[1]) / 3
    center = (center_lon, center_lat)

    v1 = np.array(_lonlat_rad_to_xyz(np.radians(p1[0]), np.radians(p1[1])))
    v2 = np.array(_lonlat_rad_to_xyz(np.radians(p2[0]), np.radians(p2[1])))
    v3 = np.array(_lonlat_rad_to_xyz(np.radians(p3[0]), np.radians(p3[1])))

    radius = (
        max(
            _angle_of_2_vectors(v1, v2),
            _angle_of_2_vectors(v1, v3),
            _angle_of_2_vectors(v2, v3),
        )
        / 2
    )

    return center, radius


@njit(cache=True)
def _is_inside_circle(circle, point):
    """Check if a point is inside a given circle on a unit sphere.

    Parameters
    ----------
    circle : tuple
        A tuple containing the center (as a tuple of lon and lat) and the radius of the circle.
    point : tuple
        The point to check, as a tuple of (lon, lat).

    Returns
    -------
    bool
        True if the point is inside the circle, False otherwise.
    """
    center, radius = circle
    v1 = np.array(_lonlat_rad_to_xyz(np.radians(center[0]), np.radians(center[1])))
    v2 = np.array(_lonlat_rad_to_xyz(np.radians(point[0]), np.radians(point[1])))
    distance = _angle_of_2_vectors(v1, v2)
    return distance <= radius


def _populate_face_centerpoints(grid, repopulate=False):
    """Calculates the face centerpoints using Welzl's algorithm. It is a
    randomized algorithm for finding the center and radius of the smallest
    circle that encloses a set of points. It is here adapted to work on a unit
    sphere. Also, this algorithm cannot be guaranteed to work on concave
    polygons.

    Parameters
    ----------
    grid : Grid
        The grid containing the nodes and faces.
    repopulate : bool, optional
        Bool used to turn on/off repopulating the face coordinates of the centerpoints, default is False.

    Returns
    -------
    None, populates the grid with the face centerpoints: face_lon, face_lat
    """
    # warnings.warn("This cannot be guaranteed to work correctly on concave polygons")

    node_lon = grid.node_lon.values
    node_lat = grid.node_lat.values

    centerpoint_lat = []
    centerpoint_lon = []

    face_nodes = grid.face_node_connectivity.values
    n_nodes_per_face = grid.n_nodes_per_face.values

    # Check if the centerpoints are already populated
    if "face_lon" not in grid._ds or repopulate:
        centerpoint_lon, centerpoint_lat = _construct_face_centerpoints(
            node_lon, node_lat, face_nodes, n_nodes_per_face
        )
    # get the cartesian coordinates of the centerpoints
    ctrpt_x, ctrpt_y, ctrpt_z = _lonlat_rad_to_xyz(centerpoint_lon, centerpoint_lat)

    # set the grid variables for centerpoints
    if "face_lon" not in grid._ds or repopulate:
        grid._ds["face_lon"] = xr.DataArray(
            centerpoint_lon, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_LON_ATTRS
        )
        grid._ds["face_lat"] = xr.DataArray(
            centerpoint_lat, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_LAT_ATTRS
        )

    if "face_x" not in grid._ds or repopulate:
        grid._ds["face_x"] = xr.DataArray(
            ctrpt_x, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_X_ATTRS
        )

        grid._ds["face_y"] = xr.DataArray(
            ctrpt_y, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_Y_ATTRS
        )

        grid._ds["face_z"] = xr.DataArray(
            ctrpt_z, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_Z_ATTRS
        )


def _construct_face_centerpoints(node_lon, node_lat, face_nodes, n_nodes_per_face):
    """Constructs the face centerpoint using Welzl's algorithm.

    Parameters
    ----------
    node_lon : array_like
        Longitudes of the nodes.
    node_lat : array_like
        Latitudes of the nodes.
    face_nodes : array_like
        Indices of nodes per face.
    n_nodes_per_face : array_like
        Number of nodes per face.

    Returns
    -------
    tuple of numpy.ndarray
        Two arrays containing the longitudes and latitudes of the centerpoints.
    """
    num_faces = face_nodes.shape[0]
    ctrpt_lon = np.zeros(num_faces, dtype=np.float64)
    ctrpt_lat = np.zeros(num_faces, dtype=np.float64)

    # Pre-compute all points arrays
    points_arrays = [
        np.column_stack(
            (
                node_lon[face_nodes[face_idx, :n_max_nodes]],
                node_lat[face_nodes[face_idx, :n_max_nodes]],
            )
        )
        for face_idx, n_max_nodes in enumerate(n_nodes_per_face)
    ]

    # Compute circles for all faces
    circles = [_smallest_enclosing_circle(points) for points in points_arrays]

    # Extract centerpoints
    ctrpt_lon, ctrpt_lat = zip(*[circle[0] for circle in circles])

    return np.array(ctrpt_lon), np.array(ctrpt_lat)


def _populate_edge_centroids(grid, repopulate=False):
    """Finds the centroids using cartesian averaging of the edges based off the
    vertices. The centroid is defined as the average of the x, y, z
    coordinates, normalized.

    Parameters
    ----------
    repopulate : bool, optional
        Bool used to turn on/off repopulating the edge coordinates of the centroids
    """

    node_x = grid.node_x.values
    node_y = grid.node_y.values
    node_z = grid.node_z.values
    edge_nodes_con = grid.edge_node_connectivity.values

    if "edge_lon" not in grid._ds or repopulate:
        # Construct the centroids if there are none stored
        if "edge_x" not in grid._ds:
            centroid_x, centroid_y, centroid_z = _construct_edge_centroids(
                node_x, node_y, node_z, edge_nodes_con
            )

        else:
            # If there are cartesian centroids already use those instead
            centroid_x, centroid_y, centroid_z = grid.edge_x, grid.edge_y, grid.edge_z

        # Convert from xyz to latlon
        centroid_lon, centroid_lat = _xyz_to_lonlat_deg(
            centroid_x, centroid_y, centroid_z, normalize=False
        )
    else:
        # Convert to xyz if there are latlon centroids already stored
        centroid_lon, centroid_lat = grid.edge_lon.values, grid.edge_lat.values
        centroid_x, centroid_y, centroid_z = _lonlat_rad_to_xyz(
            centroid_lon, centroid_lat
        )

    # Populate the centroids
    if "edge_lon" not in grid._ds or repopulate:
        grid._ds["edge_lon"] = xr.DataArray(
            centroid_lon, dims=[ugrid.EDGE_DIM], attrs=ugrid.EDGE_LON_ATTRS
        )
        grid._ds["edge_lat"] = xr.DataArray(
            centroid_lat,
            dims=[ugrid.EDGE_DIM],
            attrs=ugrid.EDGE_LAT_ATTRS,
        )

    if "edge_x" not in grid._ds or repopulate:
        grid._ds["edge_x"] = xr.DataArray(
            centroid_x,
            dims=[ugrid.EDGE_DIM],
            attrs=ugrid.EDGE_X_ATTRS,
        )

        grid._ds["edge_y"] = xr.DataArray(
            centroid_y,
            dims=[ugrid.EDGE_DIM],
            attrs=ugrid.EDGE_Y_ATTRS,
        )

        grid._ds["edge_z"] = xr.DataArray(
            centroid_z,
            dims=[ugrid.EDGE_DIM],
            attrs=ugrid.EDGE_Z_ATTRS,
        )


def _construct_edge_centroids(node_x, node_y, node_z, edge_node_conn):
    """Constructs the xyz centroid coordinate for each edge using Cartesian
    Averaging."""

    centroid_x = np.mean(node_x[edge_node_conn], axis=1)
    centroid_y = np.mean(node_y[edge_node_conn], axis=1)
    centroid_z = np.mean(node_z[edge_node_conn], axis=1)

    return _normalize_xyz(centroid_x, centroid_y, centroid_z)


def _set_desired_longitude_range(ds):
    """Sets the longitude range to [-180, 180] for all longitude variables."""

    for lon_name in ["node_lon", "edge_lon", "face_lon"]:
        if lon_name in ds:
            if ds[lon_name].max() > 180:
                ds[lon_name].data = (ds[lon_name].data + 180) % 360 - 180


def _xyz_to_lonlat_rad(
    x: Union[np.ndarray, float],
    y: Union[np.ndarray, float],
    z: Union[np.ndarray, float],
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts Cartesian x, y, z coordinates in Spherical latitude and
    longitude coordinates in degrees.

    Parameters
    ----------
    x : Union[np.ndarray, float]
        Cartesian x coordinates
    y: Union[np.ndarray, float]
        Cartesiain y coordinates
    z: Union[np.ndarray, float]
        Cartesian z coordinates
    normalize: bool
        Flag to select whether to normalize the coordinates

    Returns
    -------
    lon : Union[np.ndarray, float]
        Longitude in radians
    lat: Union[np.ndarray, float]
        Latitude in radians
    """

    if normalize:
        x, y, z = _normalize_xyz(x, y, z)
        denom = np.abs(x * x + y * y + z * z)
        x /= denom
        y /= denom
        z /= denom

    lon = np.arctan2(y, x, dtype=np.float64)
    lat = np.arcsin(z, dtype=np.float64)

    # set longitude range to [0, pi]
    lon = np.mod(lon, 2 * np.pi)

    z_mask = np.abs(z) > 1.0 - ERROR_TOLERANCE

    lat = np.where(z_mask, np.sign(z) * np.pi / 2, lat)
    lon = np.where(z_mask, 0.0, lon)

    return lon, lat


@njit(cache=True)
def _xyz_to_lonlat_rad_no_norm(
    x: Union[np.ndarray, float],
    y: Union[np.ndarray, float],
    z: Union[np.ndarray, float],
):
    """Converts a Cartesian x,y,z coordinates into Spherical latitude and
    longitude without normalization, decorated with Numba.

    Parameters
    ----------
    x : float
        Cartesian x coordinate
    y: float
        Cartesiain y coordinate
    z: float
        Cartesian z coordinate


    Returns
    -------
    lon : float
        Longitude in radians
    lat: float
        Latitude in radians
    """

    lon = np.atan2(y, x)
    lat = np.asin(z)

    # set longitude range to [0, pi]
    lon = np.mod(lon, 2 * np.pi)

    z_mask = np.abs(z) > 1.0 - ERROR_TOLERANCE

    lat = np.where(z_mask, np.sign(z) * np.pi / 2, lat)
    lon = np.where(z_mask, 0.0, lon)

    return lon, lat


def _normalize_xyz(
    x: Union[np.ndarray, float],
    y: Union[np.ndarray, float],
    z: Union[np.ndarray, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalizes a set of Cartesiain coordinates."""
    denom = np.linalg.norm(
        np.asarray(np.array([x, y, z]), dtype=np.float64), ord=2, axis=0
    )

    x_norm = x / denom
    y_norm = y / denom
    z_norm = z / denom
    return x_norm, y_norm, z_norm


@njit(cache=True)
def _lonlat_rad_to_xyz(
    lon: Union[np.ndarray, float],
    lat: Union[np.ndarray, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts Spherical lon and lat coordinates into Cartesian x, y, z
    coordinates."""
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)

    return x, y, z


def _xyz_to_lonlat_deg(
    x: Union[np.ndarray, float],
    y: Union[np.ndarray, float],
    z: Union[np.ndarray, float],
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts Cartesian x, y, z coordinates in Spherical latitude and
    longitude coordinates in degrees.

    Parameters
    ----------
    x : Union[np.ndarray, float]
        Cartesian x coordinates
    y: Union[np.ndarray, float]
        Cartesiain y coordinates
    z: Union[np.ndarray, float]
        Cartesian z coordinates
    normalize: bool
        Flag to select whether to normalize the coordinates

    Returns
    -------
    lon : Union[np.ndarray, float]
        Longitude in degrees
    lat: Union[np.ndarray, float]
        Latitude in degrees
    """
    lon_rad, lat_rad = _xyz_to_lonlat_rad(x, y, z, normalize=normalize)

    lon = np.rad2deg(lon_rad)
    lat = np.rad2deg(lat_rad)

    lon = (lon + 180) % 360 - 180
    return lon, lat


@njit(cache=True)
def _normalize_xyz_scalar(x: float, y: float, z: float):
    denom = np.linalg.norm(np.asarray(np.array([x, y, z]), dtype=np.float64), ord=2)
    x_norm = x / denom
    y_norm = y / denom
    z_norm = z / denom
    return x_norm, y_norm, z_norm


def prepare_points(points, normalize):
    """Prepares points for use with ``Grid.from_points()``"""
    if len(points) == 2:
        lon_deg, lat_deg = points[0], points[1]
        lon_rad = np.deg2rad(lon_deg)
        lat_rad = np.deg2rad(lat_deg)
        x, y, z = _lonlat_rad_to_xyz(lon_rad, lat_rad)
        x, y, z = _normalize_xyz(x, y, z)
    elif len(points) == 3:
        x, y, z = points[0], points[1], points[2]
        if normalize:
            x, y, z = _normalize_xyz(x, y, z)
    else:
        raise ValueError(
            "Points must be a sequence of length 2 (longitude, latitude) or 3 (x, y, z coordinates)."
        )

    return np.vstack([x, y, z]).T
