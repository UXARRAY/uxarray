import numpy as np
from uxarray.constants import INT_DTYPE, ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.intersections import gca_gca_intersection
import warnings

from numba import njit

POLE_POINTS = {"North": np.array([0.0, 0.0, 1.0]), "South": np.array([0.0, 0.0, -1.0])}

# number of faces/polygons before raising a warning for performance
GDF_POLYGON_THRESHOLD = 100000

REFERENCE_POINT_EQUATOR = np.array([1.0, 0.0, 0.0])


# General Helpers for Polygon Viz
# ----------------------------------------------------------------------------------------------------------------------
@njit
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

    # if face_node_connectivity.shape[0] == 1:
    #     closed[0, int(n_nodes_per_face):] = closed[0, 0]
    # else:
    #     for i, final_node_idx in enumerate(n_nodes_per_face):
    #         closed[i, final_node_idx:] = closed[i, 0]

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
):
    """Builds an array of polygon shells, which can be used with Shapely to
    construct polygons."""
    closed_face_nodes = _pad_closed_face_nodes(
        face_node_connectivity, n_face, n_max_face_nodes, n_nodes_per_face
    )

    polygon_shells = (
        np.array(
            [node_lon[closed_face_nodes], node_lat[closed_face_nodes]], dtype=np.float32
        )
        .swapaxes(0, 1)
        .swapaxes(1, 2)
    )

    return polygon_shells


def _grid_to_polygon_geodataframe(grid, exclude_antimeridian):
    """Converts the faces of a ``Grid`` into a ``spatialpandas.GeoDataFrame``
    with a geometry column of polygons."""

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

    if grid.n_face > GDF_POLYGON_THRESHOLD:
        warnings.warn(
            "Converting to a GeoDataFrame with over 1,000,000 faces may take some time."
        )

    if len(grid.antimeridian_face_indices) == 0:
        # if no faces cross antimeridian, no need to correct
        exclude_antimeridian = True

    if exclude_antimeridian:
        # build gdf without antimeridian faces
        gdf = _build_geodataframe_without_antimeridian(
            polygon_shells, antimeridian_face_indices
        )
    else:
        # build with antimeridian faces
        gdf = _build_geodataframe_with_antimeridian(
            polygon_shells, antimeridian_face_indices
        )

    return gdf


# Helpers (NO ANTIMERIDIAN)
# ----------------------------------------------------------------------------------------------------------------------
def _build_geodataframe_without_antimeridian(polygon_shells, antimeridian_face_indices):
    """Builds a ``spatialpandas.GeoDataFrame`` excluding any faces that cross
    the antimeridian."""
    from spatialpandas.geometry import PolygonArray
    from spatialpandas import GeoDataFrame

    shells_without_antimeridian = np.delete(
        polygon_shells, antimeridian_face_indices, axis=0
    )
    geometry = PolygonArray.from_exterior_coords(shells_without_antimeridian)

    gdf = GeoDataFrame({"geometry": geometry})

    return gdf


# Helpers (ANTIMERIDIAN)
# ----------------------------------------------------------------------------------------------------------------------
def _build_geodataframe_with_antimeridian(polygon_shells, antimeridian_face_indices):
    """Builds a ``spatialpandas.GeoDataFrame`` including any faces that cross
    the antimeridian."""
    # import optional dependencies
    from spatialpandas.geometry import MultiPolygonArray
    from spatialpandas import GeoDataFrame

    polygons = _build_corrected_shapely_polygons(
        polygon_shells, antimeridian_face_indices
    )

    geometry = MultiPolygonArray(polygons)

    gdf = GeoDataFrame({"geometry": geometry})

    return gdf


def _build_corrected_shapely_polygons(polygon_shells, antimeridian_face_indices):
    import antimeridian
    from shapely import polygons as Polygons

    # list of shapely Polygons representing each face in our grid
    polygons = Polygons(polygon_shells)

    # obtain each antimeridian polygon
    antimeridian_polygons = polygons[antimeridian_face_indices]

    # correct each antimeridian polygon
    corrected_polygons = [antimeridian.fix_polygon(P) for P in antimeridian_polygons]

    # insert correct polygon back into original array
    for i in reversed(antimeridian_face_indices):
        polygons[i] = corrected_polygons.pop()

    return polygons


def _build_antimeridian_face_indices(shells_x):
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
    from shapely import Polygon

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

    # original_to_corrected = np.array(
    #     _corrected_shells_to_original_faces, dtype=INT_DTYPE
    # )

    return corrected_polygon_shells, _corrected_shells_to_original_faces


def _grid_to_matplotlib_polycollection(grid):
    """Constructs and returns a ``matplotlib.collections.PolyCollection``"""

    # import optional dependencies
    from matplotlib.collections import PolyCollection

    polygon_shells = _build_polygon_shells(
        grid.node_lon.values,
        grid.node_lat.values,
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_nodes,
        grid.n_nodes_per_face.values,
    )

    (
        corrected_polygon_shells,
        corrected_to_original_faces,
    ) = _build_corrected_polygon_shells(polygon_shells)

    return PolyCollection(corrected_polygon_shells), corrected_to_original_faces


def _grid_to_matplotlib_linecollection(grid):
    """Constructs and returns a ``matplotlib.collections.LineCollection``"""

    # import optional dependencies
    from matplotlib.collections import LineCollection

    polygon_shells = _build_polygon_shells(
        grid.node_lon.values,
        grid.node_lat.values,
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_nodes,
        grid.n_nodes_per_face.values,
    )

    # obtain corrected shapely polygons
    polygons = _build_corrected_shapely_polygons(
        polygon_shells, grid.antimeridian_face_indices
    )

    # Convert polygons into lines
    lines = []
    for pol in polygons:
        boundary = pol.boundary
        if boundary.geom_type == "MultiLineString":
            for line in list(boundary.geoms):
                lines.append(np.array(line.coords))
        else:
            lines.append(np.array(boundary.coords))

    # need transform? consider adding it later if needed
    return LineCollection(lines)


def _pole_point_inside_polygon(pole, face_edge_cart):
    """Determines if a pole point is inside a polygon.

    .. note::
        - If the pole point is on the edge of the polygon, it will be considered "inside the polygon".

    Parameters
    ----------
    pole : str
        Either 'North' or 'South'.
    face_edge_cart : np.ndarray
        A face polygon represented by edges in Cartesian coordinates. Shape: (n_edges, 2, 3)

    Returns
    -------
    bool
        True if pole point is inside polygon, False otherwise.

    Raises
    ------
    ValueError
        If the provided pole is neither 'North' nor 'South'.

    Warning
    -------
    UserWarning
        Raised if the face contains both pole points.
    """
    if pole not in POLE_POINTS:
        raise ValueError('Pole point must be either "North" or "South"')

    # Classify the polygon's location
    location = _classify_polygon_location(face_edge_cart)
    pole_point = POLE_POINTS[pole]

    if location == pole:
        ref_edge = np.array([pole_point, REFERENCE_POINT_EQUATOR])
        return _check_intersection(ref_edge, face_edge_cart) % 2 != 0
    elif location == "Equator":
        # smallest offset I can obtain when using the float64 type

        ref_edge_north = np.array([pole_point, REFERENCE_POINT_EQUATOR])
        ref_edge_south = np.array([-pole_point, REFERENCE_POINT_EQUATOR])

        north_edges = face_edge_cart[np.any(face_edge_cart[:, :, 2] > 0, axis=1)]
        south_edges = face_edge_cart[np.any(face_edge_cart[:, :, 2] < 0, axis=1)]

        return (
            _check_intersection(ref_edge_north, north_edges)
            + _check_intersection(ref_edge_south, south_edges)
        ) % 2 != 0
    else:
        warnings.warn(
            "The given face should not contain both pole points.", UserWarning
        )
        return False


def _check_intersection(ref_edge, edges):
    """Check the number of intersections of the reference edge with the given
    edges.

    Parameters
    ----------
    ref_edge : np.ndarray
        Reference edge to check intersections against.
    edges : np.ndarray
        Edges to check for intersections. Shape: (n_edges, 2, 3)

    Returns
    -------
    int
        Count of intersections.
    """
    pole_point, ref_point = ref_edge
    intersection_count = 0

    for edge in edges:
        intersection_point = gca_gca_intersection(ref_edge, edge)

        if intersection_point.size != 0:
            if np.allclose(intersection_point, pole_point, atol=ERROR_TOLERANCE):
                return True
            intersection_count += 1

    return intersection_count


def _classify_polygon_location(face_edge_cart):
    """Classify the location of the polygon relative to the hemisphere.

    Parameters
    ----------
    face_edge_cart : np.ndarray
        A face polygon represented by edges in Cartesian coordinates. Shape: (n_edges, 2, 3)

    Returns
    -------
    str
        Returns either 'North', 'South' or 'Equator' based on the polygon's location.
    """
    z_coords = face_edge_cart[:, :, 2]
    if np.all(z_coords > 0):
        return "North"
    elif np.all(z_coords < 0):
        return "South"
    else:
        return "Equator"


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

    lon0, lon1 = latlonbox_rad[1]

    # Check longitude range validity
    if (lon0 < 0.0 or lon0 > 2.0 * np.pi) and lon0 != INT_FILL_VALUE:
        raise Exception("lon0 out of range ({} not in [0, 2π])".format(lon0))

    if lon0 <= lon1:
        return lon1 - lon0
    else:
        # Adjust for periodicity
        return 2 * np.pi - lon0 + lon1


def _insert_pt_in_latlonbox(old_box, new_pt, is_lon_periodic=True):
    """Update the latitude-longitude box to include a new point in radians.

    This function compares the new point's latitude and longitude with the
    existing latitude-longitude box and updates the box if necessary to include the new point.

    Parameters
    ----------
    old_box : np.ndarray
        The original latitude-longitude box in radian, a 2x2 array: [min_lat, max_lat],[left_lon, right_lon]].
    new_pt : np.ndarray
        The new latitude-longitude point in radian, an array: [lat, lon].
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

    Examples
    --------
    >>> _insert_pt_in_latlonbox(np.array([[1.0, 2.0], [3.0, 4.0]]),np.array([1.5, 3.5]))
    array([[1.0, 2.0], [3.0, 4.0]])
    """
    if np.all(new_pt == INT_FILL_VALUE):
        return old_box

    latlon_box = np.copy(old_box)  # Create a copy of the old box
    latlon_box = np.array(
        latlon_box, dtype=np.float64
    )  # Cast to float64, otherwise the following update might fail

    lat_pt, lon_pt = new_pt

    # Check if the latitude range is uninitialized and update it
    if old_box[0][0] == old_box[0][1] == INT_FILL_VALUE:
        latlon_box[0] = np.array([lat_pt, lat_pt])

    # Check if the longitude range is uninitialized and update it
    if old_box[1][0] == old_box[1][1] == INT_FILL_VALUE:
        latlon_box[1] = np.array([lon_pt, lon_pt])

    if lon_pt != INT_FILL_VALUE and (lon_pt < 0.0 or lon_pt > 2.0 * np.pi):
        raise Exception(f"lon_pt out of range ({lon_pt} not in [0, 2π])")

    # Check for pole points and update latitudes
    is_pole_point = (
        lon_pt == INT_FILL_VALUE
        and np.isclose(
            new_pt[0], [0.5 * np.pi, -0.5 * np.pi], atol=ERROR_TOLERANCE
        ).any()
    )

    if is_pole_point:
        # Check if the new point is close to the North Pole
        if np.isclose(new_pt[0], 0.5 * np.pi, atol=ERROR_TOLERANCE):
            latlon_box[0][1] = 0.5 * np.pi

        # Check if the new point is close to the South Pole
        elif np.isclose(new_pt[0], -0.5 * np.pi, atol=ERROR_TOLERANCE):
            latlon_box[0][0] = -0.5 * np.pi

        return latlon_box
    else:
        latlon_box[0] = [min(latlon_box[0][0], lat_pt), max(latlon_box[0][1], lat_pt)]

    # Update longitude range for non-periodic or periodic cases
    if not is_lon_periodic:
        latlon_box[1] = [min(latlon_box[1][0], lon_pt), max(latlon_box[1][1], lon_pt)]
    else:
        if (
            latlon_box[1][0] > latlon_box[1][1]
            and (lon_pt < latlon_box[1][0] and lon_pt > latlon_box[1][1])
        ) or (
            latlon_box[1][0] <= latlon_box[1][1]
            and not (latlon_box[1][0] <= lon_pt <= latlon_box[1][1])
        ):
            # Calculate and compare new box widths
            box_a, box_b = np.copy(latlon_box), np.copy(latlon_box)
            box_a[1][0], box_b[1][1] = lon_pt, lon_pt
            d_width_a, d_width_b = (
                _get_latlonbox_width(box_a),
                _get_latlonbox_width(box_b),
            )

            # The width should not be negative, if so, raise an exception
            if d_width_a < 0 or d_width_b < 0:
                raise Exception("logic error")

            # Return the arc with the smaller width
            latlon_box = box_a if d_width_a < d_width_b else box_b

    return latlon_box
