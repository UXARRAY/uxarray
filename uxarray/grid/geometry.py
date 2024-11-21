import antimeridian
import cartopy.crs as ccrs
import geopandas
from matplotlib.collections import LineCollection, PolyCollection
from numba import njit, prange
import numpy as np
import pandas as pd
import shapely
from shapely import Polygon
from shapely import polygons as Polygons
import spatialpandas
from spatialpandas.geometry import MultiPolygonArray, PolygonArray
import xarray as xr
import math

from uxarray.constants import (
    ERROR_TOLERANCE,
    INT_DTYPE,
    INT_FILL_VALUE,
)
from uxarray.grid.arcs import extreme_gca_latitude, point_within_gca
from uxarray.grid.coordinates import _xyz_to_lonlat_rad
from uxarray.grid.intersections import gca_gca_intersection
from uxarray.grid.utils import (
    _get_cartesian_face_edge_nodes,
    _get_lonlat_rad_face_edge_nodes,
)
from uxarray.utils.computing import allclose, isclose

POLE_POINTS_XYZ = {
    "North": np.array([0.0, 0.0, 1.0]),
    "South": np.array([0.0, 0.0, -1.0]),
}
POLE_POINTS_LONLAT = {
    "North": np.array([0.0, np.pi / 2]),
    "South": np.array([0.0, -np.pi / 2]),
}

# number of faces/polygons before raising a warning for performance
GDF_POLYGON_THRESHOLD = 100000

REFERENCE_POINT_EQUATOR_XYZ = np.array([1.0, 0.0, 0.0])
REFERENCE_POINT_EQUATOR_LONLAT = np.array([0.0, 0.0])  # TODO ?

POLE_NAME_TO_INT = {"North": 1, "Equator": 0, "South": -1}


@njit(cache=True)
def error_radius(p1, p2):
    """Calculate the error radius between two points in 3D space."""
    numerator = math.sqrt(
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    )
    denominator = math.sqrt(p2[0] ** 2 + p2[1] ** 2 + p2[2] ** 2)
    return numerator / denominator


@njit(cache=True)
def _unique_points(points, tolerance=ERROR_TOLERANCE):
    """Identify unique intersection points from a list of points, considering floating point precision errors.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (n_points, 3) containing the intersection points.
    tolerance : float
        The distance threshold within which two points are considered identical.

    Returns
    -------
    np.ndarray
        An array of unique points in Cartesian coordinates.
    """
    n_points = points.shape[0]
    unique_points = np.empty((n_points, 3), dtype=points.dtype)
    unique_count = 0

    for i in range(n_points):
        point = points[i]
        is_unique = True
        for j in range(unique_count):
            unique_point = unique_points[j]
            if error_radius(point, unique_point) < tolerance:
                is_unique = False
                break
        if is_unique:
            unique_points[unique_count] = point
            unique_count += 1

    return unique_points[:unique_count]


@njit(cache=True)
def _pad_closed_face_nodes(
    face_node_connectivity, n_face, n_max_face_nodes, n_nodes_per_face
):
    """Pads a closed array of face nodes by inserting the first element at any
    point a fill value is encountered.

    Ensures each resulting polygon has the same number of vertices.
    """

    closed = np.ones((n_face, n_max_face_nodes + 1), dtype=INT_DTYPE)

    # set final value to the original
    closed[:, :-1] = face_node_connectivity.copy()

    for i, final_node_idx in enumerate(n_nodes_per_face):
        closed[i, final_node_idx:] = closed[i, 0]

    return closed


def _build_polygon_shells(
    node_lon,
    node_lat,
    face_node_connectivity,
    n_face,
    n_max_face_nodes,
    n_nodes_per_face,
    projection=None,
    central_longitude=0.0,
):
    """Builds an array of polygon shells, which can be used with Shapely to
    construct polygons."""

    closed_face_nodes = _pad_closed_face_nodes(
        face_node_connectivity, n_face, n_max_face_nodes, n_nodes_per_face
    )

    if projection:
        lonlat_proj = projection.transform_points(
            ccrs.PlateCarree(central_longitude=central_longitude), node_lon, node_lat
        )

        node_lon = lonlat_proj[:, 0]
        node_lat = lonlat_proj[:, 1]

    polygon_shells = (
        np.array(
            [node_lon[closed_face_nodes], node_lat[closed_face_nodes]], dtype=np.float32
        )
        .swapaxes(0, 1)
        .swapaxes(1, 2)
    )

    return polygon_shells


def _correct_central_longitude(node_lon, node_lat, projection):
    """Shifts the central longitude of an unstructured grid, which moves the
    antimeridian when visualizing, which is used when projections have a
    central longitude other than 0.0."""
    if projection:
        central_longitude = projection.proj4_params["lon_0"]
        if central_longitude != 0.0:
            _source_projection = ccrs.PlateCarree(central_longitude=0.0)
            _destination_projection = ccrs.PlateCarree(
                central_longitude=projection.proj4_params["lon_0"]
            )

            lonlat_proj = _destination_projection.transform_points(
                _source_projection, node_lon, node_lat
            )

            node_lon = lonlat_proj[:, 0]
    else:
        central_longitude = 0.0

    return node_lon, node_lat, central_longitude


def _grid_to_polygon_geodataframe(grid, periodic_elements, projection, project, engine):
    """Converts the faces of a ``Grid`` into a ``spatialpandas.GeoDataFrame``
    or ``geopandas.GeoDataFrame`` with a geometry column of polygons."""

    node_lon, node_lat, central_longitude = _correct_central_longitude(
        grid.node_lon.values, grid.node_lat.values, projection
    )
    polygon_shells = _build_polygon_shells(
        node_lon,
        node_lat,
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_nodes,
        grid.n_nodes_per_face.values,
        projection=None,
        central_longitude=central_longitude,
    )

    if projection is not None and project:
        projected_polygon_shells = _build_polygon_shells(
            node_lon,
            node_lat,
            grid.face_node_connectivity.values,
            grid.n_face,
            grid.n_max_face_nodes,
            grid.n_nodes_per_face.values,
            projection=projection,
            central_longitude=central_longitude,
        )
    else:
        projected_polygon_shells = None

    antimeridian_face_indices = _build_antimeridian_face_indices(
        polygon_shells[:, :, 0]
    )

    non_nan_polygon_indices = None
    if projection is not None and project:
        shells_d = np.delete(
            projected_polygon_shells, antimeridian_face_indices, axis=0
        )

        # Check for NaN in each sub-array and invert the condition
        does_not_contain_nan = ~np.isnan(shells_d).any(axis=1)

        # Get the indices where NaN is NOT present
        non_nan_polygon_indices = np.where(does_not_contain_nan)[0]

    grid._gdf_cached_parameters["antimeridian_face_indices"] = antimeridian_face_indices

    if periodic_elements == "split":
        gdf = _build_geodataframe_with_antimeridian(
            polygon_shells,
            projected_polygon_shells,
            antimeridian_face_indices,
            engine=engine,
        )
    elif periodic_elements == "ignore":
        if engine == "geopandas":
            # create a geopandas.GeoDataFrame
            if projected_polygon_shells is not None:
                geometry = projected_polygon_shells
            else:
                geometry = polygon_shells

            gdf = geopandas.GeoDataFrame({"geometry": shapely.polygons(geometry)})
        else:
            # create a spatialpandas.GeoDataFrame
            if projected_polygon_shells is not None:
                geometry = PolygonArray.from_exterior_coords(projected_polygon_shells)
            else:
                geometry = PolygonArray.from_exterior_coords(polygon_shells)
            gdf = spatialpandas.GeoDataFrame({"geometry": geometry})

    else:
        gdf = _build_geodataframe_without_antimeridian(
            polygon_shells,
            projected_polygon_shells,
            antimeridian_face_indices,
            engine=engine,
        )
    return gdf, non_nan_polygon_indices


def _build_geodataframe_without_antimeridian(
    polygon_shells, projected_polygon_shells, antimeridian_face_indices, engine
):
    """Builds a ``spatialpandas.GeoDataFrame`` or
    ``geopandas.GeoDataFrame``excluding any faces that cross the
    antimeridian."""
    if projected_polygon_shells is not None:
        # use projected shells if a projection is applied
        shells_without_antimeridian = np.delete(
            projected_polygon_shells, antimeridian_face_indices, axis=0
        )
    else:
        shells_without_antimeridian = np.delete(
            polygon_shells, antimeridian_face_indices, axis=0
        )

    if engine == "geopandas":
        # create a geopandas.GeoDataFrame
        gdf = geopandas.GeoDataFrame(
            {"geometry": shapely.polygons(shells_without_antimeridian)}
        )
    else:
        # create a spatialpandas.GeoDataFrame
        geometry = PolygonArray.from_exterior_coords(shells_without_antimeridian)
        gdf = spatialpandas.GeoDataFrame({"geometry": geometry})

    return gdf


def _build_geodataframe_with_antimeridian(
    polygon_shells,
    projected_polygon_shells,
    antimeridian_face_indices,
    engine,
):
    """Builds a ``spatialpandas.GeoDataFrame`` or ``geopandas.GeoDataFrame``
    including any faces that cross the antimeridian."""
    polygons = _build_corrected_shapely_polygons(
        polygon_shells, projected_polygon_shells, antimeridian_face_indices
    )
    if engine == "geopandas":
        # Create a geopandas.GeoDataFrame
        gdf = geopandas.GeoDataFrame({"geometry": polygons})
    else:
        # Create a spatialpandas.GeoDataFrame
        geometry = MultiPolygonArray(polygons)
        gdf = spatialpandas.GeoDataFrame({"geometry": geometry})

    return gdf


def _build_corrected_shapely_polygons(
    polygon_shells,
    projected_polygon_shells,
    antimeridian_face_indices,
):
    if projected_polygon_shells is not None:
        # use projected shells if a projection is applied
        shells = projected_polygon_shells
    else:
        shells = polygon_shells

    # list of shapely Polygons representing each face in our grid
    polygons = Polygons(shells)

    # construct antimeridian polygons
    antimeridian_polygons = Polygons(polygon_shells[antimeridian_face_indices])

    # correct each antimeridian polygon
    corrected_polygons = [antimeridian.fix_polygon(P) for P in antimeridian_polygons]

    # insert correct polygon back into original array
    for i in reversed(antimeridian_face_indices):
        polygons[i] = corrected_polygons.pop()

    return polygons


def _build_antimeridian_face_indices(shells_x, projection=None):
    """Identifies any face that has an edge that crosses the antimeridian."""

    x_mag = np.abs(np.diff(shells_x))
    x_mag_cross = np.any(x_mag >= 180, axis=1)
    x_cross_indices = np.argwhere(x_mag_cross)

    if x_cross_indices.ndim == 2:
        if x_cross_indices.shape[1] == 1:
            return x_cross_indices[:, 0]
        else:
            return x_cross_indices.squeeze()
    elif x_cross_indices.ndim == 0:
        return np.array([], dtype=INT_DTYPE)
    else:
        return x_cross_indices


def _populate_antimeridian_face_indices(grid):
    """Populates ``Grid.antimeridian_face_indices``"""
    polygon_shells = _build_polygon_shells(
        grid.node_lon.values,
        grid.node_lat.values,
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_nodes,
        grid.n_nodes_per_face.values,
    )

    antimeridian_face_indices = _build_antimeridian_face_indices(
        polygon_shells[:, :, 0]
    )

    return antimeridian_face_indices


def _build_corrected_polygon_shells(polygon_shells):
    """Constructs ``corrected_polygon_shells`` and
    ``Grid.original_to_corrected), representing the polygon shells, with
    antimeridian polygons split.

     Parameters
    ----------
    grid : uxarray.Grid
        Grid Object

    Returns
    -------
    corrected_polygon_shells : np.ndarray
        Array containing polygon shells, with antimeridian polygons split
    _corrected_shells_to_original_faces : np.ndarray
        Original indices used to map the corrected polygon shells to their entries in face nodes
    """

    # import optional dependencies
    import antimeridian

    # list of shapely Polygons representing each Face in our grid
    polygons = [Polygon(shell) for shell in polygon_shells]

    # List of Polygons (non-split) and MultiPolygons (split across antimeridian)
    corrected_polygons = [
        antimeridian.fix_polygon(P, fix_winding=False) for P in polygons
    ]

    _corrected_shells_to_original_faces = []
    corrected_polygon_shells = []

    for i, polygon in enumerate(corrected_polygons):
        # Convert MultiPolygons into individual Polygon Vertices
        if polygon.geom_type == "MultiPolygon":
            for individual_polygon in polygon.geoms:
                corrected_polygon_shells.append(
                    np.array(
                        [
                            individual_polygon.exterior.coords.xy[0],
                            individual_polygon.exterior.coords.xy[1],
                        ]
                    ).T
                )
                _corrected_shells_to_original_faces.append(i)

        # Convert Shapely Polygon into Polygon Vertices
        else:
            corrected_polygon_shells.append(
                np.array(
                    [polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1]]
                ).T
            )
            _corrected_shells_to_original_faces.append(i)

    return corrected_polygon_shells, _corrected_shells_to_original_faces


def _grid_to_matplotlib_polycollection(
    grid, periodic_elements, projection=None, **kwargs
):
    """Constructs and returns a ``matplotlib.collections.PolyCollection``"""

    # Handle unsupported configuration: splitting periodic elements with projection
    if periodic_elements == "split" and projection is not None:
        raise ValueError(
            "Explicitly projecting lines is not supported. Please pass in your projection"
            "using the 'transform' parameter"
        )

    # Correct the central longitude and build polygon shells
    node_lon, node_lat, central_longitude = _correct_central_longitude(
        grid.node_lon.values, grid.node_lat.values, projection
    )

    if "transform" not in kwargs:
        if projection is None:
            kwargs["transform"] = ccrs.PlateCarree(central_longitude=central_longitude)
        else:
            kwargs["transform"] = projection

    polygon_shells = _build_polygon_shells(
        node_lon,
        node_lat,
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_nodes,
        grid.n_nodes_per_face.values,
        projection=None,
        central_longitude=central_longitude,
    )

    # Projected polygon shells if a projection is specified
    if projection is not None:
        projected_polygon_shells = _build_polygon_shells(
            node_lon,
            node_lat,
            grid.face_node_connectivity.values,
            grid.n_face,
            grid.n_max_face_nodes,
            grid.n_nodes_per_face.values,
            projection=projection,
            central_longitude=central_longitude,
        )
    else:
        projected_polygon_shells = None

    # Determine indices of polygons crossing the antimeridian
    antimeridian_face_indices = _build_antimeridian_face_indices(
        polygon_shells[:, :, 0]
    )

    # Filter out NaN-containing polygons if projection is applied
    non_nan_polygon_indices = None
    if projected_polygon_shells is not None:
        # Delete polygons at the antimeridian
        shells_d = np.delete(
            projected_polygon_shells, antimeridian_face_indices, axis=0
        )

        # Get the indices of polygons that do not contain NaNs
        does_not_contain_nan = ~np.isnan(shells_d).any(axis=(1, 2))
        non_nan_polygon_indices = np.where(does_not_contain_nan)[0]

    grid._poly_collection_cached_parameters["non_nan_polygon_indices"] = (
        non_nan_polygon_indices
    )
    grid._poly_collection_cached_parameters["antimeridian_face_indices"] = (
        antimeridian_face_indices
    )

    # Select which shells to use: projected or original
    if projected_polygon_shells is not None:
        shells_to_use = projected_polygon_shells
    else:
        shells_to_use = polygon_shells

    # Handle periodic elements: exclude or split antimeridian polygons
    if periodic_elements == "exclude":
        # Remove antimeridian polygons and keep only non-NaN polygons if available
        shells_without_antimeridian = np.delete(
            shells_to_use, antimeridian_face_indices, axis=0
        )

        # Filter the shells using non-NaN indices
        if non_nan_polygon_indices is not None:
            shells_to_use = shells_without_antimeridian[non_nan_polygon_indices]
        else:
            shells_to_use = shells_without_antimeridian

        # Get the corrected indices of original faces
        corrected_to_original_faces = np.delete(
            np.arange(grid.n_face), antimeridian_face_indices, axis=0
        )

        # Create the PolyCollection using the cleaned shells
        return PolyCollection(shells_to_use, **kwargs), corrected_to_original_faces

    elif periodic_elements == "split":
        # Split polygons at the antimeridian
        (
            corrected_polygon_shells,
            corrected_to_original_faces,
        ) = _build_corrected_polygon_shells(polygon_shells)

        # Create PolyCollection using the corrected shells
        return PolyCollection(
            corrected_polygon_shells, **kwargs
        ), corrected_to_original_faces

    else:
        # Default: use original polygon shells
        return PolyCollection(polygon_shells, **kwargs), []


def _get_polygons(grid, periodic_elements, projection=None, apply_projection=True):
    # Correct the central longitude if projection is provided
    node_lon, node_lat, central_longitude = _correct_central_longitude(
        grid.node_lon.values, grid.node_lat.values, projection
    )

    # Build polygon shells without projection
    polygon_shells = _build_polygon_shells(
        node_lon,
        node_lat,
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_nodes,
        grid.n_nodes_per_face.values,
        projection=None,
        central_longitude=central_longitude,
    )

    # If projection is provided, create the projected polygon shells
    if projection and apply_projection:
        projected_polygon_shells = _build_polygon_shells(
            node_lon,
            node_lat,
            grid.face_node_connectivity.values,
            grid.n_face,
            grid.n_max_face_nodes,
            grid.n_nodes_per_face.values,
            projection=projection,
            central_longitude=central_longitude,
        )
    else:
        projected_polygon_shells = None

    # Determine indices of polygons crossing the antimeridian
    antimeridian_face_indices = _build_antimeridian_face_indices(
        polygon_shells[:, :, 0]
    )

    # Filter out NaN-containing polygons if projection is applied
    non_nan_polygon_indices = None
    if projected_polygon_shells is not None:
        # Delete polygons at the antimeridian
        shells_d = np.delete(
            projected_polygon_shells, antimeridian_face_indices, axis=0
        )

        # Get the indices of polygons that do not contain NaNs
        does_not_contain_nan = ~np.isnan(shells_d).any(axis=(1, 2))
        non_nan_polygon_indices = np.where(does_not_contain_nan)[0]

    # Determine which shells to use
    if projected_polygon_shells is not None:
        shells_to_use = projected_polygon_shells
    else:
        shells_to_use = polygon_shells

    # Exclude or handle periodic elements based on the input parameter
    if periodic_elements == "exclude":
        # Remove antimeridian polygons and keep only non-NaN polygons if available
        shells_without_antimeridian = np.delete(
            shells_to_use, antimeridian_face_indices, axis=0
        )

        # Filter the shells using non-NaN indices
        if non_nan_polygon_indices is not None:
            shells_to_use = shells_without_antimeridian[non_nan_polygon_indices]
        else:
            shells_to_use = shells_without_antimeridian

        polygons = _convert_shells_to_polygons(shells_to_use)
    elif periodic_elements == "split":
        # Correct for antimeridian crossings and split polygons as necessary
        polygons = _build_corrected_shapely_polygons(
            polygon_shells, projected_polygon_shells, antimeridian_face_indices
        )
    else:
        # Default: use original polygon shells
        polygons = _convert_shells_to_polygons(polygon_shells)

    return (
        polygons,
        central_longitude,
        antimeridian_face_indices,
        non_nan_polygon_indices,
    )


def _grid_to_matplotlib_linecollection(
    grid, periodic_elements, projection=None, **kwargs
):
    """Constructs and returns a ``matplotlib.collections.LineCollection``"""

    if periodic_elements == "split" and projection is not None:
        apply_projection = False
    else:
        apply_projection = True

    # do not explicitly project when splitting elements
    polygons, central_longitude, _, _ = _get_polygons(
        grid, periodic_elements, projection, apply_projection
    )

    # Convert polygons to line segments for the LineCollection
    lines = []
    for pol in polygons:
        boundary = pol.boundary
        if boundary.geom_type == "MultiLineString":
            for line in list(boundary.geoms):
                lines.append(np.array(line.coords))
        else:
            lines.append(np.array(boundary.coords))

    if "transform" not in kwargs:
        # Set default transform if one is not provided not provided
        if projection is None or not apply_projection:
            kwargs["transform"] = ccrs.PlateCarree(central_longitude=central_longitude)
        else:
            kwargs["transform"] = projection

    return LineCollection(lines, **kwargs)


def _convert_shells_to_polygons(shells):
    """Convert polygon shells to shapely Polygon or MultiPolygon objects."""
    polygons = []
    for shell in shells:
        # Remove NaN values from each polygon shell
        cleaned_shell = shell[~np.isnan(shell[:, 0])]
        if len(cleaned_shell) > 2:  # A valid polygon needs at least 3 points
            polygons.append(Polygon(cleaned_shell))

    return polygons


def _pole_point_inside_polygon_cartesian(pole, face_edges_xyz):
    if isinstance(pole, str):
        pole = POLE_NAME_TO_INT[pole]

    x = face_edges_xyz[:, :, 0]
    y = face_edges_xyz[:, :, 1]
    z = face_edges_xyz[:, :, 2]

    lon, lat = _xyz_to_lonlat_rad(x, y, z)

    face_edges_lonlat = np.stack((lon, lat), axis=2)

    return pole_point_inside_polygon(pole, face_edges_xyz, face_edges_lonlat)

    pass


@njit(cache=True)
def pole_point_inside_polygon(pole, face_edges_xyz, face_edges_lonlat):
    """Determines if a pole point is inside a polygon."""

    if pole != 1 and pole != -1:
        raise ValueError("Pole must be 1 (North) or -1 (South)")

    # Define constants within the function
    pole_point_xyz = np.empty(3, dtype=np.float64)
    pole_point_xyz[0] = 0.0
    pole_point_xyz[1] = 0.0
    pole_point_xyz[2] = 1.0 * pole

    pole_point_lonlat = np.empty(2, dtype=np.float64)
    pole_point_lonlat[0] = 0.0
    pole_point_lonlat[1] = (math.pi / 2) * pole

    REFERENCE_POINT_EQUATOR_XYZ = np.empty(3, dtype=np.float64)
    REFERENCE_POINT_EQUATOR_XYZ[0] = 1.0
    REFERENCE_POINT_EQUATOR_XYZ[1] = 0.0
    REFERENCE_POINT_EQUATOR_XYZ[2] = 0.0

    REFERENCE_POINT_EQUATOR_LONLAT = np.empty(2, dtype=np.float64)
    REFERENCE_POINT_EQUATOR_LONLAT[0] = 0.0
    REFERENCE_POINT_EQUATOR_LONLAT[1] = 0.0

    # Classify the polygon's location
    location = _classify_polygon_location(face_edges_xyz)

    if (location == 1 and pole == -1) or (location == -1 and pole == 1):
        return False

    elif location == -1 or location == 1:
        # Initialize ref_edge_xyz
        ref_edge_xyz = np.empty((2, 3), dtype=np.float64)
        ref_edge_xyz[0, 0] = pole_point_xyz[0]
        ref_edge_xyz[0, 1] = pole_point_xyz[1]
        ref_edge_xyz[0, 2] = pole_point_xyz[2]
        ref_edge_xyz[1, :] = REFERENCE_POINT_EQUATOR_XYZ

        # Initialize ref_edge_lonlat
        ref_edge_lonlat = np.empty((2, 2), dtype=np.float64)
        ref_edge_lonlat[0, 0] = pole_point_lonlat[0]
        ref_edge_lonlat[0, 1] = pole_point_lonlat[1]
        ref_edge_lonlat[1, :] = REFERENCE_POINT_EQUATOR_LONLAT

        intersection_count = _check_intersection(
            ref_edge_xyz, ref_edge_lonlat, face_edges_xyz, face_edges_lonlat
        )
        return (intersection_count % 2) != 0

    elif location == 0:  # Equator
        # Initialize ref_edge_north_xyz and ref_edge_north_lonlat
        ref_edge_north_xyz = np.empty((2, 3), dtype=np.float64)
        ref_edge_north_xyz[0, 0] = 0.0
        ref_edge_north_xyz[0, 1] = 0.0
        ref_edge_north_xyz[0, 2] = 1.0
        ref_edge_north_xyz[1, :] = REFERENCE_POINT_EQUATOR_XYZ

        ref_edge_north_lonlat = np.empty((2, 2), dtype=np.float64)
        ref_edge_north_lonlat[0, 0] = 0.0
        ref_edge_north_lonlat[0, 1] = math.pi / 2
        ref_edge_north_lonlat[1, :] = REFERENCE_POINT_EQUATOR_LONLAT

        # Initialize ref_edge_south_xyz and ref_edge_south_lonlat
        ref_edge_south_xyz = np.empty((2, 3), dtype=np.float64)
        ref_edge_south_xyz[0, 0] = 0.0
        ref_edge_south_xyz[0, 1] = 0.0
        ref_edge_south_xyz[0, 2] = -1.0
        ref_edge_south_xyz[1, :] = REFERENCE_POINT_EQUATOR_XYZ

        ref_edge_south_lonlat = np.empty((2, 2), dtype=np.float64)
        ref_edge_south_lonlat[0, 0] = 0.0
        ref_edge_south_lonlat[0, 1] = -math.pi / 2
        ref_edge_south_lonlat[1, :] = REFERENCE_POINT_EQUATOR_LONLAT

        # Classify edges based on z-coordinate
        n_edges = face_edges_xyz.shape[0]
        north_edges_xyz = np.empty((n_edges, 2, 3), dtype=np.float64)
        north_edges_lonlat = np.empty((n_edges, 2, 2), dtype=np.float64)
        south_edges_xyz = np.empty((n_edges, 2, 3), dtype=np.float64)
        south_edges_lonlat = np.empty((n_edges, 2, 2), dtype=np.float64)
        north_count = 0
        south_count = 0

        for i in range(n_edges):
            edge_xyz = face_edges_xyz[i]
            edge_lonlat = face_edges_lonlat[i]
            if edge_xyz[0, 2] > 0 or edge_xyz[1, 2] > 0:
                north_edges_xyz[north_count] = edge_xyz
                north_edges_lonlat[north_count] = edge_lonlat
                north_count += 1
            elif edge_xyz[0, 2] < 0 or edge_xyz[1, 2] < 0:
                south_edges_xyz[south_count] = edge_xyz
                south_edges_lonlat[south_count] = edge_lonlat
                south_count += 1
            else:
                # skip edges exactly on the equator
                continue

        if north_count > 0:
            north_edges_xyz = north_edges_xyz[:north_count]
            north_edges_lonlat = north_edges_lonlat[:north_count]
        else:
            north_edges_xyz = np.empty((0, 2, 3), dtype=np.float64)
            north_edges_lonlat = np.empty((0, 2, 2), dtype=np.float64)

        if south_count > 0:
            south_edges_xyz = south_edges_xyz[:south_count]
            south_edges_lonlat = south_edges_lonlat[:south_count]
        else:
            south_edges_xyz = np.empty((0, 2, 3), dtype=np.float64)
            south_edges_lonlat = np.empty((0, 2, 2), dtype=np.float64)

        # Count south intersections
        north_intersections = _check_intersection(
            ref_edge_north_xyz,
            ref_edge_north_lonlat,
            north_edges_xyz,
            north_edges_lonlat,
        )

        # Count south intersections
        south_intersections = _check_intersection(
            ref_edge_south_xyz,
            ref_edge_south_lonlat,
            south_edges_xyz,
            south_edges_lonlat,
        )

        return ((north_intersections + south_intersections) % 2) != 0

    else:
        raise ValueError("Invalid pole point query.")


@njit(cache=True)
def _classify_polygon_location(face_edge_cart):
    """Classify the location of the polygon relative to the hemisphere."""
    z_coords = face_edge_cart[:, :, 2]
    if np.all(z_coords > 0):  # Use strict inequality
        return 1  # North
    elif np.all(z_coords < 0):  # Use strict inequality
        return -1  # South
    else:
        return 0  # Equator


@njit(cache=True)
def _check_intersection(ref_edge_xyz, ref_edge_lonlat, edges_xyz, edges_lonlat):
    """Check the number of intersections of the reference edge with the given edges.

    Parameters
    ----------
    ref_edge_xyz : np.ndarray
        Reference edge to check intersections against. Shape: (2, 3)
    ref_edge_lonlat : np.ndarray
        Reference edge longitude and latitude. Shape: (2, 2)
    edges_xyz : np.ndarray
        Edges to check for intersections. Shape: (n_edges, 2, 3)
    edges_lonlat : np.ndarray
        Longitude and latitude of the edges. Shape: (n_edges, 2, 2)

    Returns
    -------
    int
        Count of intersections.
    """
    pole_point_xyz = ref_edge_xyz[0]
    n_edges = edges_xyz.shape[0]

    # Assuming at most 2 intersections per edge
    max_intersections = n_edges * 2
    intersection_points = np.empty((max_intersections, 3), dtype=np.float64)
    intersection_count = 0

    for i in range(n_edges):
        edge_xyz = edges_xyz[i]
        edge_lonlat = edges_lonlat[i]

        # compute intersection
        intersection_point = gca_gca_intersection(
            ref_edge_xyz, ref_edge_lonlat, edge_xyz, edge_lonlat
        )

        if intersection_point.size != 0:
            if intersection_point.ndim == 1:
                # Only one point
                point = intersection_point
                if allclose(point, pole_point_xyz, atol=ERROR_TOLERANCE):
                    return True
                intersection_points[intersection_count] = point
                intersection_count += 1
            else:
                # Multiple points
                num_points = intersection_point.shape[0]
                for j in range(num_points):
                    point = intersection_point[j]
                    if allclose(point, pole_point_xyz, atol=ERROR_TOLERANCE):
                        return True
                    intersection_points[intersection_count] = point
                    intersection_count += 1

    if intersection_count == 0:
        return 0

    unique_intersection_points = _unique_points(
        intersection_points[:intersection_count], tolerance=ERROR_TOLERANCE
    )
    unique_count = unique_intersection_points.shape[0]

    # If there's only one unique intersection point, check if it matches any edge nodes
    if unique_count == 1:
        intersection_point = unique_intersection_points[0]
        for i in range(n_edges):
            edge_xyz = edges_xyz[i]
            if allclose(
                intersection_point, edge_xyz[0], atol=ERROR_TOLERANCE
            ) or allclose(intersection_point, edge_xyz[1], atol=ERROR_TOLERANCE):
                return 0

    return unique_count


@njit(cache=True)
def _get_latlonbox_width(latlonbox_rad):
    """Calculate the width of a latitude-longitude box in radians. The box
    should be represented by a 2x2 array in radians and lon0 represent the
    "left" side of the box. while lon1 represent the "right" side of the box.

    This function computes the width of a given latitude-longitude box. It
    accounts for periodicity in the longitude direction.

    Non-Periodic Longitude: This is the usual case where longitude values are considered within a fixed range,
            typically between -180 and 180 degrees, or 0 and 360 degrees.
            Here, the longitude does not "wrap around" when it reaches the end of this range.

    Periodic Longitude: In this case, the longitude is considered to wrap around the globe.
            This means that if you have a longitude range from 350 to 10 degrees,
            it is understood to cross the 0-degree meridian and actually represents a 20-degree span
            (350 to 360 degrees, then 0 to 10 degrees).

    Parameters
    ----------
    latlonbox_rad : np.ndarray
        A latitude-longitude box represented by a 2x2 array in radians and lon0 represent the "left" side of the box.
        while lon1 represent the "right" side of the box:
        [[lat_0, lat_1], [lon_0, lon_1]].

    Returns
    -------
    float
        The width of the latitude-longitude box in radians.

    Raises
    ------
    Exception
        If the input longitude range is invalid.

    Warning
        If the input longitude range is flagged as periodic but in the form [lon0, lon1] where lon0 < lon1.
        The function will automatically use the is_lon_periodic=False instead.
    """
    # TODO: ensure this function works properly
    lon0, lon1 = latlonbox_rad[1]

    if lon0 != INT_FILL_VALUE:
        lon0 = np.mod(lon0, 2 * np.pi)
    if lon1 != INT_FILL_VALUE:
        lon1 = np.mod(lon1, 2 * np.pi)
    if (lon0 < 0.0 or lon0 > 2.0 * np.pi) and lon0 != INT_FILL_VALUE:
        # -1 used for exception
        return -1

    if lon0 <= lon1:
        return lon1 - lon0
    else:
        # Adjust for periodicity
        return 2 * np.pi - lon0 + lon1


@njit(cache=True)
def all_elements_nan(arr):
    """Check if all elements in an array are np.nan."""
    for i in range(arr.shape[0]):
        if not np.isnan(arr[i]):
            return False
    return True


@njit(cache=True)
def any_close_lat(lat_pt, atol):
    """Check if the latitude point is close to either the North or South Pole."""
    return isclose(lat_pt, 0.5 * math.pi, atol) or isclose(lat_pt, -0.5 * math.pi, atol)


@njit(cache=True)
def insert_pt_in_latlonbox(old_box, new_pt, is_lon_periodic=True):
    """Update the latitude-longitude box to include a new point in radians.

    Parameters
    ----------
    old_box : np.ndarray
        The original latitude-longitude box in radians, a 2x2 array:
        [[min_lat, max_lat],
         [left_lon, right_lon]].
    new_pt : np.ndarray
        The new latitude-longitude point in radians, an array: [lat, lon].
    is_lon_periodic : bool, optional
        Flag indicating if the latitude-longitude box is periodic in longitude (default is True).

    Returns
    -------
    np.ndarray
        Updated latitude-longitude box including the new point in radians.

    Raises
    ------
    Exception
        If logic errors occur in the calculation process.
    """
    # Check if the new point is a fill value
    all_fill = all_elements_nan(new_pt)
    if all_fill:
        return old_box

    # Create a copy of the old box
    latlon_box = np.copy(old_box)

    # Extract latitude and longitude from the new point
    lat_pt = new_pt[0]
    lon_pt = new_pt[1]

    # Normalize the longitude if it's not a fill value
    if not np.isnan(lon_pt):
        lon_pt = lon_pt % (2.0 * math.pi)

    # Check if the latitude range is uninitialized and update it
    if np.isnan(old_box[0, 0]) and np.isnan(old_box[0, 1]):
        latlon_box[0, 0] = lat_pt
        latlon_box[0, 1] = lat_pt
    else:
        # Update latitude range
        if lat_pt < latlon_box[0, 0]:
            latlon_box[0, 0] = lat_pt
        if lat_pt > latlon_box[0, 1]:
            latlon_box[0, 1] = lat_pt

    # Check if the longitude range is uninitialized and update it
    if np.isnan(old_box[1, 0]) and np.isnan(old_box[1, 1]):
        latlon_box[1, 0] = lon_pt
        latlon_box[1, 1] = lon_pt
    else:
        # Validate longitude point
        if not np.isnan(lon_pt) and (lon_pt < 0.0 or lon_pt > 2.0 * math.pi):
            raise Exception("Longitude point out of range")

        # Check for pole points
        is_pole_point = False
        if np.isnan(lon_pt):
            if any_close_lat(lat_pt, ERROR_TOLERANCE):
                is_pole_point = True

        if is_pole_point:
            # Update latitude for pole points
            if isclose(lat_pt, 0.5 * math.pi, ERROR_TOLERANCE):
                latlon_box[0, 1] = 0.5 * math.pi  # Update max_lat for North Pole
            elif isclose(lat_pt, -0.5 * math.pi, ERROR_TOLERANCE):
                latlon_box[0, 0] = -0.5 * math.pi  # Update min_lat for South Pole
        else:
            # Update longitude range based on periodicity
            if not is_lon_periodic:
                # Non-periodic: straightforward min and max updates
                if lon_pt < latlon_box[1, 0]:
                    latlon_box[1, 0] = lon_pt
                if lon_pt > latlon_box[1, 1]:
                    latlon_box[1, 1] = lon_pt
            else:
                # Periodic longitude handling
                # Determine if the new point extends the current longitude range
                # considering the periodic boundary at 2*pi
                left_lon = latlon_box[1, 0]
                right_lon = latlon_box[1, 1]

                # Check if the current box wraps around
                wraps_around = left_lon > right_lon

                if wraps_around:
                    # If the box wraps around, check if the new point is outside the current range
                    if not (left_lon <= lon_pt or lon_pt <= right_lon):
                        # Decide to extend either the left or the right
                        # Calculate the new width for both possibilities
                        # Option 1: Extend the left boundary to lon_pt
                        box_a = np.copy(latlon_box)
                        box_a[1, 0] = lon_pt
                        d_width_a = _get_latlonbox_width(box_a)

                        # Option 2: Extend the right boundary to lon_pt
                        box_b = np.copy(latlon_box)
                        box_b[1, 1] = lon_pt
                        d_width_b = _get_latlonbox_width(box_b)

                        # Ensure widths are non-negative
                        if (d_width_a < 0.0) or (d_width_b < 0.0):
                            raise Exception(
                                "Logic error in longitude box width calculation"
                            )

                        # Choose the box with the smaller width
                        if d_width_a < d_width_b:
                            latlon_box = box_a
                        else:
                            latlon_box = box_b
                else:
                    # If the box does not wrap around, simply update min or max longitude
                    if lon_pt < left_lon or lon_pt > right_lon:
                        # Calculate the new width for both possibilities
                        # Option 1: Extend the left boundary to lon_pt
                        box_a = np.copy(latlon_box)
                        box_a[1, 0] = lon_pt
                        d_width_a = _get_latlonbox_width(box_a)

                        # Option 2: Extend the right boundary to lon_pt
                        box_b = np.copy(latlon_box)
                        box_b[1, 1] = lon_pt
                        d_width_b = _get_latlonbox_width(box_b)

                        # Ensure widths are non-negative
                        if (d_width_a < 0.0) or (d_width_b < 0.0):
                            raise Exception(
                                "Logic error in longitude box width calculation"
                            )

                        # Choose the box with the smaller width
                        if d_width_a < d_width_b:
                            latlon_box = box_a
                        else:
                            latlon_box = box_b

    return latlon_box


@njit(cache=True)
def _populate_face_latlon_bound(
    face_edges_xyz,
    face_edges_lonlat,
    is_latlonface=False,
    is_GCA_list=None,
):
    # Check if face_edges contains pole points
    has_north_pole = pole_point_inside_polygon(1, face_edges_xyz, face_edges_lonlat)
    has_south_pole = pole_point_inside_polygon(-1, face_edges_xyz, face_edges_lonlat)

    # Initialize face_latlon_array with INT_FILL_VALUE
    face_latlon_array = np.full((2, 2), np.nan, dtype=np.float64)

    if has_north_pole or has_south_pole:
        # Initial assumption that the pole point is inside the face
        is_center_pole = True

        pole_point_xyz = np.zeros(3, dtype=np.float64)
        pole_point_lonlat = np.zeros(2, dtype=np.float64)
        new_pt_latlon = np.zeros(2, dtype=np.float64)

        if has_north_pole:
            pole_point_xyz[2] = 1.0  # [0.0, 0.0, 1.0]
            pole_point_lonlat[1] = math.pi / 2  # [0.0, pi/2]
            new_pt_latlon[0] = math.pi / 2  # [pi/2, INT_FILL_VALUE]
            new_pt_latlon[1] = np.nan
        else:
            pole_point_xyz[2] = -1.0  # [0.0, 0.0, -1.0]
            pole_point_lonlat[1] = -math.pi / 2  # [0.0, -pi/2]
            new_pt_latlon[0] = -math.pi / 2  # [-pi/2, INT_FILL_VALUE]
            new_pt_latlon[1] = np.nan

        for i in range(face_edges_xyz.shape[0]):
            edge_xyz = face_edges_xyz[i]
            edge_lonlat = face_edges_lonlat[i]

            # Skip processing if the edge is marked as a dummy with a fill value
            if np.any(edge_xyz == INT_FILL_VALUE):
                continue

            # Extract cartesian coordinates of the edge's endpoints
            n1_cart = edge_xyz[0]
            n2_cart = edge_xyz[1]
            n1_lonlat = edge_lonlat[0]
            n2_lonlat = edge_lonlat[1]

            # Extract latitudes and longitudes of the nodes
            node1_lon_rad = n1_lonlat[0]
            node1_lat_rad = n1_lonlat[1]

            # Determine if the edge's extreme latitudes need to be considered
            if is_GCA_list is not None:
                is_GCA = is_GCA_list[i]
            else:
                is_GCA = not is_latlonface or n1_cart[2] != n2_cart[2]

            # Check if the node matches the pole point or if the pole point is within the edge
            max_abs_diff = np.max(np.abs(n1_cart - pole_point_xyz))
            if max_abs_diff <= ERROR_TOLERANCE or point_within_gca(
                pole_point_xyz,
                pole_point_lonlat,
                n1_cart,
                n1_lonlat,
                n2_cart,
                n2_lonlat,
                is_directed=False,
            ):
                is_center_pole = False
                face_latlon_array = insert_pt_in_latlonbox(
                    face_latlon_array, new_pt_latlon
                )

            # Insert the current node's lat/lon into the latlonbox
            face_latlon_array = insert_pt_in_latlonbox(
                face_latlon_array, np.array([node1_lat_rad, node1_lon_rad])
            )

            # Create n1n2_cart and n1n2_lonlat arrays using preallocation
            n1n2_cart = np.empty((2, 3), dtype=np.float64)
            n1n2_cart[0, :] = n1_cart
            n1n2_cart[1, :] = n2_cart

            n1n2_lonlat = np.empty((2, 2), dtype=np.float64)
            n1n2_lonlat[0, :] = n1_lonlat
            n1n2_lonlat[1, :] = n2_lonlat

            # Determine extreme latitudes for GCA edges
            if is_GCA:
                lat_max = extreme_gca_latitude(n1n2_cart, n1n2_lonlat, "max")
                lat_min = extreme_gca_latitude(n1n2_cart, n1n2_lonlat, "min")
            else:
                lat_max = node1_lat_rad
                lat_min = node1_lat_rad

            # Insert latitudinal extremes based on pole presence
            if has_north_pole:
                face_latlon_array = insert_pt_in_latlonbox(
                    face_latlon_array, np.array([lat_min, node1_lon_rad])
                )
                face_latlon_array[0, 1] = math.pi / 2  # Upper latitude bound
            else:
                face_latlon_array = insert_pt_in_latlonbox(
                    face_latlon_array, np.array([lat_max, node1_lon_rad])
                )
                face_latlon_array[0, 0] = -math.pi / 2  # Lower latitude bound

        # Adjust longitude bounds globally if the pole is centrally inside the polygon
        if is_center_pole:
            face_latlon_array[1, 0] = 0.0
            face_latlon_array[1, 1] = 2 * math.pi

    else:
        # Normal Face
        for i in range(face_edges_xyz.shape[0]):
            edge_xyz = face_edges_xyz[i]
            edge_lonlat = face_edges_lonlat[i]

            # Skip processing if the edge is marked as a dummy with a fill value
            if np.any(edge_xyz == INT_FILL_VALUE):
                continue

            # Extract cartesian coordinates of the edge's endpoints
            n1_cart = edge_xyz[0]
            n2_cart = edge_xyz[1]
            n1_lonlat = edge_lonlat[0]
            n2_lonlat = edge_lonlat[1]

            # Extract latitudes and longitudes of the nodes
            node1_lon_rad = n1_lonlat[0]
            node1_lat_rad = n1_lonlat[1]
            # node2_lon_rad = n2_lonlat[0]
            node2_lat_rad = n2_lonlat[1]

            # Determine if the edge's extreme latitudes need to be considered
            if is_GCA_list is not None:
                is_GCA = is_GCA_list[i]
            else:
                is_GCA = not is_latlonface or n1_cart[2] != n2_cart[2]

            # Create n1n2_cart and n1n2_lonlat arrays using preallocation
            n1n2_cart = np.empty((2, 3), dtype=np.float64)
            n1n2_cart[0, :] = n1_cart
            n1n2_cart[1, :] = n2_cart

            n1n2_lonlat = np.empty((2, 2), dtype=np.float64)
            n1n2_lonlat[0, :] = n1_lonlat
            n1n2_lonlat[1, :] = n2_lonlat

            if is_GCA:
                lat_max = extreme_gca_latitude(n1n2_cart, n1n2_lonlat, "max")
                lat_min = extreme_gca_latitude(n1n2_cart, n1n2_lonlat, "min")
            else:
                lat_max = node1_lat_rad
                lat_min = node1_lat_rad

            # Insert extreme latitude points into the latlonbox
            if (
                abs(node1_lat_rad - lat_max) > ERROR_TOLERANCE
                and abs(node2_lat_rad - lat_max) > ERROR_TOLERANCE
            ):
                face_latlon_array = insert_pt_in_latlonbox(
                    face_latlon_array, np.array([lat_max, node1_lon_rad])
                )
            elif (
                abs(node1_lat_rad - lat_min) > ERROR_TOLERANCE
                and abs(node2_lat_rad - lat_min) > ERROR_TOLERANCE
            ):
                face_latlon_array = insert_pt_in_latlonbox(
                    face_latlon_array, np.array([lat_min, node1_lon_rad])
                )
            else:
                face_latlon_array = insert_pt_in_latlonbox(
                    face_latlon_array, np.array([node1_lat_rad, node1_lon_rad])
                )

    return face_latlon_array


@njit(cache=True, parallel=True)
def compute_temp_latlon_array(
    face_node_connectivity,
    faces_edges_cartesian,
    faces_edges_lonlat_rad,
    n_nodes_per_face,
    is_latlonface,
    is_face_GCA_list,
    INT_FILL_VALUE,
):
    n_face = face_node_connectivity.shape[0]
    temp_latlon_array = np.full((n_face, 2, 2), INT_FILL_VALUE, dtype=np.float64)
    for face_idx in prange(n_face):
        cur_face_edges_cartesian = faces_edges_cartesian[
            face_idx, 0 : n_nodes_per_face[face_idx]
        ]
        cur_faces_edges_lonlat_rad = faces_edges_lonlat_rad[
            face_idx, 0 : n_nodes_per_face[face_idx]
        ]
        if is_face_GCA_list is not None:
            is_GCA_list = is_face_GCA_list[face_idx]
        else:
            is_GCA_list = None

        temp_latlon_array[face_idx] = _populate_face_latlon_bound(
            cur_face_edges_cartesian,
            cur_faces_edges_lonlat_rad,
            is_latlonface=is_latlonface,
            is_GCA_list=is_GCA_list,
        )
    return temp_latlon_array


def _populate_bounds(
    grid, is_latlonface: bool = False, is_face_GCA_list=None, return_array=False
):
    """Populates the bounds of the grid based on the geometry of its faces,
    taking into account special conditions such as faces crossing the
    antimeridian or containing pole points. This method updates the grid's
    internal representation to include accurate bounds for each face, returned
    as a DataArray with detailed attributes.

    Parameters
    ----------
    is_latlonface : bool, optional
        A global flag that indicates if faces are latlon faces. If True, all faces
        are treated as latlon faces, meaning that all edges are either longitude or
        constant latitude lines. If False, all edges are considered as Great Circle Arcs (GCA).
        Default is False.

    is_face_GCA_list : list or np.ndarray, optional
        A list or an array of boolean values for each face, indicating whether each edge
        in that face is a GCA. The shape of the list or array should be (n_faces, n_edges),
        with each sub-list or sub-array like [True, False, True, False] indicating the
        nature of each edge (GCA or constant latitude line) in a face. This parameter allows
        for mixed face types within the grid by specifying the edge type at the face level.
        If None, all edges are considered as GCA. This parameter, if provided, will overwrite
        the `is_latlonface` attribute for specific faces. Default is None.

    Returns
    -------
    xr.DataArray
        A DataArray containing the latitude and longitude bounds for each face in the grid,
        expressed in radians. The array has dimensions ["n_face", "Two", "Two"], where "Two"
        is a literal dimension name indicating two bounds (min and max) for each of latitude
        and longitude. The DataArray includes attributes detailing its purpose and the mapping
        of latitude intervals to face indices.

        Attributes include:
        - `cf_role`: Describes the role of the DataArray, here indicating face latitude bounds.
        - `_FillValue`: The fill value used in the array, indicating uninitialized or missing data.
        - `long_name`: A descriptive name for the DataArray.
        - `start_index`: The starting index for face indices in the grid.
        - `latitude_intervalsIndex`: An IntervalIndex indicating the latitude intervals.
        - `latitude_intervals_name_map`: A DataFrame mapping the latitude intervals to face indices.

    Example
    -------
    Consider a scenario where you have four faces on a grid, each defined by vertices in longitude and latitude degrees:

        face_1 = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
        face_2 = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
        face_3 = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
        face_4 = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]

    After defining these faces, you can create a grid and populate its bounds by treating all faces as latlon faces:

        grid = ux.Grid.from_face_vertices([face_1, face_2, face_3, face_4], latlon=True)
        bounds_dataarray = grid._populate_bounds(is_latlonface=True)

    This will calculate and store the bounds for each face within the grid, adjusting for any special conditions such as crossing the antimeridian, and return them as a DataArray.
    """

    # Ensure grid's cartesian coordinates are normalized
    grid.normalize_cartesian_coordinates()

    # Prepare data for Numba functions
    faces_edges_cartesian = _get_cartesian_face_edge_nodes(
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_edges,
        grid.node_x.values,
        grid.node_y.values,
        grid.node_z.values,
    )

    faces_edges_lonlat_rad = _get_lonlat_rad_face_edge_nodes(
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_edges,
        grid.node_lon.values,
        grid.node_lat.values,
    )

    n_nodes_per_face = grid.n_nodes_per_face.values

    # Call the Numba-compiled function
    temp_latlon_array = compute_temp_latlon_array(
        grid.face_node_connectivity.values,
        faces_edges_cartesian,
        faces_edges_lonlat_rad,
        n_nodes_per_face,
        is_latlonface,
        is_face_GCA_list,
        INT_FILL_VALUE,
    )

    # Process results outside Numba
    intervals_tuple_list = []
    intervals_name_list = []
    for face_idx in range(grid.n_face):
        assert temp_latlon_array[face_idx][0][0] != temp_latlon_array[face_idx][0][1]
        assert temp_latlon_array[face_idx][1][0] != temp_latlon_array[face_idx][1][1]
        lat_array = temp_latlon_array[face_idx][0]
        intervals_tuple_list.append((lat_array[0], lat_array[1]))
        intervals_name_list.append(face_idx)

    intervalsIndex = pd.IntervalIndex.from_tuples(intervals_tuple_list, closed="both")
    df_intervals_map = pd.DataFrame(
        index=intervalsIndex, data=intervals_name_list, columns=["face_id"]
    )

    bounds = xr.DataArray(
        temp_latlon_array,
        dims=["n_face", "lon_lat", "min_max"],
        attrs={
            "cf_role": "face_latlon_bounds",
            "_FillValue": INT_FILL_VALUE,
            "long_name": "Provides the latitude and longitude bounds for each face in radians.",
            "start_index": INT_DTYPE(0),
            "latitude_intervalsIndex": intervalsIndex,
            "latitude_intervals_name_map": df_intervals_map,
        },
    )

    if return_array:
        return bounds
    else:
        grid._ds["bounds"] = bounds


def _construct_boundary_edge_indices(edge_face_connectivity):
    """Index the missing edges on a partial grid with holes, that is a region
    of the grid that is not covered by any geometry."""

    # If an edge only has one face saddling it than the mesh has holes in it
    edge_with_holes = np.where(edge_face_connectivity[:, 1] == INT_FILL_VALUE)[0]
    return edge_with_holes


def stereographic_projection(lon, lat, central_lon, central_lat):
    """Projects a point on the surface of the sphere to a plane using stereographic projection

    Parameters
    ----------
    lon: np.ndarray
        Longitude coordinates of point
    lat: np.ndarray
        Latitude coordinate of point
    central_lon: np.ndarray
        Central longitude of projection
    central_lat: np.ndarray
        Central latitude of projection
    Returns
    -------
    x: np.ndarray
        2D x coordinate of projected point
    y: np.ndarray
        2D y coordinate of projected point
    """

    # Convert to radians
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    central_lon = np.deg2rad(central_lon)
    central_lat = np.deg2rad(central_lat)

    # Calculate constant used for calculation
    k = 2.0 / (
        1.0
        + np.sin(central_lat) * np.sin(lat)
        + np.cos(central_lat) * np.cos(lat) * np.cos(lon - central_lon)
    )

    # Calculate the x and y coordinates
    x = k * np.cos(lat) * np.sin(lon - central_lon)
    y = k * (
        np.cos(central_lat) * np.sin(lat)
        - np.sin(central_lat) * np.cos(lat) * np.cos(lon - central_lon)
    )

    return x, y


def inverse_stereographic_projection(x, y, central_lon, central_lat):
    """Projects a point on a plane to the surface of the sphere using stereographic projection

    Parameters
    ----------
    x: np.ndarray
        2D x coordinates of point
    y: np.ndarray
        2D y coordinate of point
    central_lon: np.ndarray
        Central longitude of projection
    central_lat: np.ndarray
        Central latitude of projection
    Returns
    -------
    lon: np.ndarray
        Longitude of projected point
    lat: np.ndarray
        Latitude of projected point
    """

    # If x and y are zero, the lon and lat will also be zero

    if x == 0 and y == 0:
        return 0, 0

    # Convert to radians
    central_lat = np.deg2rad(central_lat)

    # Calculate constants used for calculation
    p = np.sqrt(x**2 + y**2)

    c = 2 * np.arctan(p / 2)

    # Calculate the lon and lat of the coordinate
    lon = central_lon + np.arctan2(
        x * np.sin(c),
        p * np.cos(central_lat) * np.cos(c) - y * np.sin(central_lat) * np.sin(c),
    )

    lat = np.arcsin(
        np.cos(c) * np.sin(central_lat) + ((y * np.sin(c) * central_lat) / p)
    )

    return lon, lat
