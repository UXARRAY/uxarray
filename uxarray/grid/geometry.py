import math

import antimeridian
import cartopy.crs as ccrs
import geopandas
import numpy as np
import shapely
import spatialpandas
from matplotlib.collections import LineCollection, PolyCollection
from numba import njit
from shapely import Polygon
from shapely import polygons as Polygons
from spatialpandas.geometry import MultiPolygonArray, PolygonArray

from uxarray.constants import (
    ERROR_TOLERANCE,
    INT_DTYPE,
    INT_FILL_VALUE,
)
from uxarray.grid.arcs import (
    point_within_gca,
)
from uxarray.grid.coordinates import _xyz_to_lonlat_rad
from uxarray.grid.intersections import (
    gca_gca_intersection,
)
from uxarray.utils.computing import allclose

POLE_POINTS_XYZ = {
    "North": np.array([0.0, 0.0, 1.0]),
    "South": np.array([0.0, 0.0, -1.0]),
}

REF_POINT_NORTH_XYZ = np.array([0.01745241, 0.0, 0.9998477], dtype=np.float64)
REF_POINT_SOUTH_XYZ = np.array([0.01745241, 0.0, -0.9998477], dtype=np.float64)

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

        intersection_count = _check_intersection(ref_edge_xyz, face_edges_xyz)
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
            north_edges_xyz,
        )

        # Count south intersections
        south_intersections = _check_intersection(
            ref_edge_south_xyz,
            south_edges_xyz,
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
def _check_intersection(ref_edge_xyz, edges_xyz):
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

        # compute intersection
        intersection_point = gca_gca_intersection(ref_edge_xyz, edge_xyz)

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


@njit(cache=True)
def point_in_face(
    edges_xyz,
    point_xyz,
    inclusive=True,
):
    """Determines if a point lies inside a face.

    Parameters
    ----------
        edges_xyz : numpy.ndarray
            Cartesian coordinates of each point in the face
        point_xyz : numpy.ndarray
            Cartesian coordinate of the point
        inclusive : bool
            Flag to determine whether to include points on the nodes and edges of the face

    Returns
    -------
    bool
        True if point is inside face, False otherwise
    """

    # Validate the inputs
    if len(edges_xyz[0][0]) != 3:
        raise ValueError("`edges_xyz` vertices must be in Cartesian coordinates.")

    if len(point_xyz) != 3:
        raise ValueError("`point_xyz` must be a single [3] Cartesian coordinate.")

    # Initialize the intersection count
    intersection_count = 0

    # Set to hold unique intersections
    unique_intersections = set()

    location = _classify_polygon_location(edges_xyz)

    if location == 1:
        ref_point_xyz = REF_POINT_SOUTH_XYZ
    elif location == -1:
        ref_point_xyz = REF_POINT_NORTH_XYZ
    else:
        ref_point_xyz = REF_POINT_SOUTH_XYZ

    # Initialize the points arc between the point and the reference point
    gca_cart = np.empty((2, 3), dtype=np.float64)
    gca_cart[0] = point_xyz
    gca_cart[1] = ref_point_xyz

    # Loop through the face's edges, checking each one for intersection
    for ind in range(len(edges_xyz)):
        # If the point lies on an edge, return True if inclusive
        if point_within_gca(
            point_xyz,
            edges_xyz[ind][0],
            edges_xyz[ind][1],
        ):
            if inclusive:
                return True
            else:
                return False

        # Get the number of intersections between the edge and the point arc
        intersections = gca_gca_intersection(edges_xyz[ind], gca_cart)

        # Add any unique intersections to the intersection_count
        for intersection in intersections:
            intersection_tuple = (
                intersection[0],
                intersection[1],
                intersection[2],
            )
            if intersection_tuple not in unique_intersections:
                unique_intersections.add(intersection_tuple)
                intersection_count += 1

    # Return True if the number of intersections is odd, False otherwise
    return intersection_count % 2 == 1


@njit(cache=True)
def _find_faces(face_edge_cartesian, point_xyz, inverse_indices):
    """Finds the faces that contain a given point, inside a subset `face_edge_cartesian`
    Parameters
    ----------
        face_edge_cartesian : numpy.ndarray
            Cartesian coordinates of all the faces according to their edges
        point_xyz : numpy.ndarray
            Cartesian coordinate of the point
        inverse_indices : numpy.ndarray
           The original indices of the subsetted grid

    Returns
    -------
    index : array
        The index of the face that contains the point
    """

    index = []

    # Run for each face in the subset
    for i, face in enumerate(inverse_indices):
        # Check to see if the face contains the point
        contains_point = point_in_face(
            face_edge_cartesian[i],
            point_xyz,
            inclusive=True,
        )

        # If the point is found, add it to the index array
        if contains_point:
            index.append(face)

    # Return the index array
    return index


def _populate_max_face_radius(self):
    """Populates `max_face_radius`

    Returns
    -------
    max_distance : np.float64
        The max distance from a node to a face center
    """

    # Parse all variables needed for `njit` functions
    face_node_connectivity = self.face_node_connectivity.values
    node_lats_rad = np.deg2rad(self.node_lat.values)
    node_lons_rad = np.deg2rad(self.node_lon.values)
    face_lats_rad = np.deg2rad(self.face_lat.values)
    face_lons_rad = np.deg2rad(self.face_lon.values)

    # Get the max distance
    max_distance = calculate_max_face_radius(
        face_node_connectivity,
        node_lats_rad,
        node_lons_rad,
        face_lats_rad,
        face_lons_rad,
    )

    # Return the max distance, which is the `max_face_radius`
    return np.rad2deg(max_distance)


@njit(cache=True)
def calculate_max_face_radius(
    face_node_connectivity, node_lats_rad, node_lons_rad, face_lats_rad, face_lons_rad
):
    """Finds the max face radius in the mesh.
    Parameters
    ----------
        face_node_connectivity : numpy.ndarray
            Cartesian coordinates of all the faces according to their edges
        node_lats_rad : numpy.ndarray
            The `Grid.node_lat` array in radians
        node_lons_rad : numpy.ndarray
           The `Grid.node_lon` array in radians
        face_lats_rad : numpy.ndarray
           The `Grid.face_lat` array in radians
        face_lons_rad : numpy.ndarray
           The `Grid.face_lon` array in radians

    Returns
    -------
    The max distance from a node to a face center
    """

    # Array to store all distances of each face to it's furthest node.
    end_distances = np.zeros(len(face_node_connectivity))

    # Loop over each face and its nodes
    for ind, face in enumerate(face_node_connectivity):
        # Filter out INT_FILL_VALUE
        valid_nodes = face[face != INT_FILL_VALUE]

        # Get the face lat/lon of this face
        face_lat = face_lats_rad[ind]
        face_lon = face_lons_rad[ind]

        # Get the node lat/lon of this face
        node_lat_rads = node_lats_rad[valid_nodes]
        node_lon_rads = node_lons_rad[valid_nodes]

        # Calculate Haversine distances for all nodes in this face
        distances = haversine_distance(node_lon_rads, node_lat_rads, face_lon, face_lat)

        # Store the max distance for this face
        end_distances[ind] = np.max(distances)

    # Return the maximum distance found across all faces
    return np.max(end_distances)


@njit(cache=True)
def haversine_distance(lon_a, lat_a, lon_b, lat_b):
    """Calculates the haversine distance between two points.

    Parameters
    ----------
    lon_a : np.float64
        The longitude of the first point
    lat_a : np.float64
        The latitude of the first point
    lon_b : np.float64
        The longitude of the second point
    lat_b : np.float64
        The latitude of the second point

    Returns
    -------
    distance :  np.float64
        The distance between the two points
    """

    # Differences in latitudes and longitudes
    dlat = lat_b - lat_a
    dlon = lon_b - lon_a

    # Haversine formula
    equation_in_sqrt = (np.sin(dlat / 2) ** 2) + np.cos(lat_a) * np.cos(lat_b) * (
        np.sin(dlon / 2) ** 2
    )
    distance = 2 * np.arcsin(np.sqrt(equation_in_sqrt))

    # Return the gotten distance
    return distance
