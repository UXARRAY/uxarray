import math

import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange

from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.arcs import (
    extreme_gca_latitude,
    point_within_gca,
)
from uxarray.grid.geometry import pole_point_inside_polygon
from uxarray.grid.point_in_face import (
    _LOC_INSIDE,
    _LOC_OUTSIDE,
    _point_in_polygon_sphere,
)
from uxarray.grid.utils import (
    _get_cartesian_face_edge_nodes,
    _get_spherical_face_edge_nodes,
    all_elements_nan,
    any_close_lat,
)

# ---------------------------------------------------------------------------
# Constants for the accurate GCA bounds path.
# ---------------------------------------------------------------------------

# Face location codes used by _face_location_info.
_FACE_LOC_LOCAL = 0
_FACE_LOC_NORTH_POLAR = 1
_FACE_LOC_SOUTH_POLAR = 2

_NORTH_POLE = np.array([0.0, 0.0, 1.0])
_SOUTH_POLE = np.array([0.0, 0.0, -1.0])


# ---------------------------------------------------------------------------
# Per-face GCA bounds helpers (accurate path).
# ---------------------------------------------------------------------------


@njit(cache=True)
def _face_location_info(face_vertices, polar_cap_z):
    """Classify a face and return (label, z_min, z_max).

    Iterates over each great-circle edge, finding the interior z-extremum that
    the arc can reach beyond its endpoints, and compares the overall z range
    against the polar-cap threshold.

    Parameters
    ----------
    face_vertices : np.ndarray, shape (n, 3)
        Unit-vector vertices of the face.
    polar_cap_z : float
        sin(polar_cap_latitude); faces whose z-range crosses ±polar_cap_z are
        classified as polar candidates.

    Returns
    -------
    label : int
        _FACE_LOC_LOCAL, _FACE_LOC_NORTH_POLAR, or _FACE_LOC_SOUTH_POLAR.
    z_min : float
    z_max : float
    """
    n = face_vertices.shape[0]
    z_max = -np.inf
    z_min = np.inf

    for i in range(n):
        j = (i + 1) % n
        x1 = face_vertices[i]
        x2 = face_vertices[j]
        z1 = x1[2]
        z2 = x2[2]
        d = x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2]

        # Parameter along the arc at which z is extremal (matches C++ get_face_location_info).
        denom = (z1 + z2) * (d - 1.0)
        a_raw = (z1 * d - z2) / denom if denom != 0.0 else -1.0
        a = min(max(a_raw, 0.0), 1.0)

        one_a = 1.0 - a
        y0 = one_a * x1[0] + a * x2[0]
        y1 = one_a * x1[1] + a * x2[1]
        y2 = one_a * x1[2] + a * x2[2]
        norm = math.sqrt(y0 * y0 + y1 * y1 + y2 * y2)
        z_ext = y2 / norm

        z_edge_max = z1 if z1 > z2 else z2
        z_edge_min = z1 if z1 < z2 else z2

        if 0.0 < a_raw < 1.0:
            z_max_candidate = z_ext
            z_min_candidate = z_ext
        else:
            z_max_candidate = z_edge_max
            z_min_candidate = z_edge_min

        if z_max_candidate > z_max:
            z_max = z_max_candidate
        if z_min_candidate < z_min:
            z_min = z_min_candidate

    north_pole_candidate = z_max >= polar_cap_z
    south_pole_candidate = z_min <= -polar_cap_z

    label = (
        north_pole_candidate * _FACE_LOC_NORTH_POLAR
        + (1 - north_pole_candidate) * south_pole_candidate * _FACE_LOC_SOUTH_POLAR
    )
    return label, z_min, z_max


@njit(cache=True)
def _lon_bounds_from_vertices(face_vertices):
    """Compute (lon_min, lon_max) in degrees in [0, 360].

    If the face crosses the antimeridian, returns lon_min > lon_max, which is
    the uxarray wrap encoding (lon_min > lon_max signals antimeridian crossing
    throughout the bounds and cross-section APIs). AccuSphGeom uses a union-of-
    intervals convention instead; this function is needed to translate to the
    uxarray encoding and cannot be removed without changing the bounds API.

    The largest-gap algorithm is standard for antimeridian detection on a set
    of vertex longitudes: the gap in sorted longitudes opposite the face
    interior is the one the face does NOT span.
    """
    n = face_vertices.shape[0]
    rad_to_deg = 180.0 / math.pi
    lons = np.empty(n)
    for i in range(n):
        x = face_vertices[i]
        lon = math.atan2(x[1], x[0]) * rad_to_deg
        if lon < 0.0:
            lon += 360.0
        lons[i] = lon

    lons_sorted = np.sort(lons)

    # Find the largest gap (including the wrap gap from last to first + 360).
    best_gap = 360.0 - (lons_sorted[n - 1] - lons_sorted[0])
    best_idx = -1  # -1 means the best gap is the wrap gap
    for i in range(n - 1):
        gap = lons_sorted[i + 1] - lons_sorted[i]
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    if best_idx >= 0:
        # A non-wrap gap beat the wrap gap — the face crosses the antimeridian.
        return lons_sorted[best_idx + 1], lons_sorted[best_idx]
    return lons_sorted[0], lons_sorted[n - 1]


@njit(cache=True)
def _generate_lat_lon_bounds_local(face_vertices, z_min, z_max):
    """Compute (lat_min, lat_max, lon_min, lon_max) in degrees for a non-polar face.

    This is a UXarray-specific post-kernel formatting step, not part of the EFT
    computation. It converts the arc z-extrema from ``_face_location_info``
    (which already accounts for interior arc extrema via the compensated kernel)
    into degree lat/lon bounds in uxarray's encoding.

    Parameters
    ----------
    face_vertices : np.ndarray, shape (n, 3)
    z_min, z_max : float
        Arc z-extrema from ``_face_location_info``.

    Returns
    -------
    lat_min, lat_max, lon_min, lon_max : float
        All in degrees; lon in [0, 360] with lon_min > lon_max for
        antimeridian-crossing faces.
    """
    rad_to_deg = 180.0 / math.pi
    lon_min, lon_max = _lon_bounds_from_vertices(face_vertices)
    lat_max = math.asin(min(z_max, 1.0)) * rad_to_deg
    lat_min = math.asin(max(z_min, -1.0)) * rad_to_deg
    return lat_min, lat_max, lon_min, lon_max


@njit(cache=True)
def _generate_lat_lon_bounds_pole(face_vertices, label, z_min, z_max):
    """Compute bounds for a polar-candidate face.

    Checks whether the relevant pole (north or south) is inside the polygon
    using the SPIP test. If the pole is not enclosed after all, falls back to
    the local path.

    Parameters
    ----------
    face_vertices : np.ndarray, shape (n, 3)
    label : int
        _FACE_LOC_NORTH_POLAR or _FACE_LOC_SOUTH_POLAR.
    z_min, z_max : float

    Returns
    -------
    lat_min, lat_max, lon_min, lon_max : float
        Degrees; lon in [0, 360], antimeridian-crossing indicated by
        lon_min > lon_max.
    wraps : bool
        True when the face spans the full longitude circle (pole inside face).
    """
    rad_to_deg = 180.0 / math.pi

    north_loc = (
        _point_in_polygon_sphere(_NORTH_POLE, face_vertices)
        if label == _FACE_LOC_NORTH_POLAR
        else _LOC_OUTSIDE
    )
    south_loc = (
        _point_in_polygon_sphere(_SOUTH_POLE, face_vertices)
        if label == _FACE_LOC_SOUTH_POLAR
        else _LOC_OUTSIDE
    )

    if north_loc == _LOC_OUTSIDE and south_loc == _LOC_OUTSIDE:
        a, b, c, d = _generate_lat_lon_bounds_local(face_vertices, z_min, z_max)
        return a, b, c, d, False

    lon_min, lon_max = _lon_bounds_from_vertices(face_vertices)
    lat_max = math.asin(min(z_max, 1.0)) * rad_to_deg
    lat_min = math.asin(max(z_min, -1.0)) * rad_to_deg

    if north_loc != _LOC_OUTSIDE:
        if north_loc == _LOC_INSIDE:
            return lat_min, 90.0, 0.0, 360.0, True
        return lat_min, 90.0, lon_min, lon_max, False

    if south_loc == _LOC_INSIDE:
        return -90.0, lat_max, 0.0, 360.0, True
    return -90.0, lat_max, lon_min, lon_max, False


@njit(cache=True, parallel=True)
def _construct_face_bounds_array_gca(
    face_node_connectivity,
    n_nodes_per_face,
    node_x,
    node_y,
    node_z,
    polar_cap_z,
):
    """Parallel GCA bounds computation using the accurate local/polar-cap path.

    Replaces ``_construct_face_bounds_array`` for the common case where all
    edges are great-circle arcs (no ``is_latlonface`` or ``is_face_GCA_list``
    overrides).

    Parameters
    ----------
    face_node_connectivity : np.ndarray, shape (n_face, max_nodes)
    n_nodes_per_face : np.ndarray, shape (n_face,)
    node_x, node_y, node_z : np.ndarray, shape (n_node,)
    polar_cap_z : float
        Precomputed sin(polar_cap_latitude).

    Returns
    -------
    np.ndarray, shape (n_face, 2, 2)
        [[lat_min, lat_max], [lon_min, lon_max]] in radians per face.
    """
    n_face = face_node_connectivity.shape[0]
    bounds_array = np.empty((n_face, 2, 2), dtype=np.float64)
    deg_to_rad = math.pi / 180.0

    for face_idx in prange(n_face):
        k = n_nodes_per_face[face_idx]
        verts = np.empty((k, 3))
        for vi in range(k):
            node = face_node_connectivity[face_idx, vi]
            verts[vi, 0] = node_x[node]
            verts[vi, 1] = node_y[node]
            verts[vi, 2] = node_z[node]

        label, z_min, z_max = _face_location_info(verts, polar_cap_z)

        if label == _FACE_LOC_LOCAL:
            lat_min, lat_max, lon_min, lon_max = _generate_lat_lon_bounds_local(
                verts, z_min, z_max
            )
        else:
            lat_min, lat_max, lon_min, lon_max, _ = _generate_lat_lon_bounds_pole(
                verts, label, z_min, z_max
            )

        bounds_array[face_idx, 0, 0] = lat_min * deg_to_rad
        bounds_array[face_idx, 0, 1] = lat_max * deg_to_rad
        bounds_array[face_idx, 1, 0] = lon_min * deg_to_rad
        bounds_array[face_idx, 1, 1] = lon_max * deg_to_rad

    return bounds_array


def _populate_face_bounds(
    grid,
    is_latlonface: bool = False,
    is_face_GCA_list=None,
    return_array=False,
    process_intervals=False,
):
    """Populates the spherical bounds for each face in a grid.


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
    return_array: bool, optional
        Whether to return the bounds array, instead of populating the grid. Default is False
    process_intervals: bool, optional
        Whether to process the latitude intervals. Default is False

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
        - `latitude_intervalsIndex`: An IntervalIndex indicating the latitude intervals, only if
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
    grid.normalize_cartesian_coordinates()

    if not is_latlonface and is_face_GCA_list is None:
        # Pure GCA grid: use the accurate local/polar-cap path.
        bounds_array = _construct_face_bounds_array_gca(
            grid.face_node_connectivity.values,
            grid.n_nodes_per_face.values,
            grid.node_x.values,
            grid.node_y.values,
            grid.node_z.values,
            math.sin(80.0 * math.pi / 180.0),
        )
    else:
        # Latlon or mixed-edge grids: use the existing path.
        bounds_array = _construct_face_bounds_array(
            grid.face_node_connectivity.values,
            grid.n_nodes_per_face.values,
            grid.node_x.values,
            grid.node_y.values,
            grid.node_z.values,
            grid.node_lon.values,
            grid.node_lat.values,
            is_latlonface,
            is_face_GCA_list,
        )

    bounds_da = xr.DataArray(
        bounds_array,
        dims=["n_face", "lat_lon", "min_max"],
        attrs={
            "cf_role": "face_latlon_bounds",
            "_FillValue": INT_FILL_VALUE,
            "long_name": "Latitude and longitude bounds for each face in radians.",
        },
    )

    if process_intervals:
        intervals_tuple_list = []
        intervals_name_list = []
        for face_idx in range(grid.n_face):
            assert bounds_array[face_idx][0][0] != bounds_array[face_idx][0][1]
            assert bounds_array[face_idx][1][0] != bounds_array[face_idx][1][1]
            lat_array = bounds_array[face_idx][0]
            intervals_tuple_list.append((lat_array[0], lat_array[1]))
            intervals_name_list.append(face_idx)

        intervalsIndex = pd.IntervalIndex.from_tuples(
            intervals_tuple_list, closed="both"
        )
        df_intervals_map = pd.DataFrame(
            index=intervalsIndex, data=intervals_name_list, columns=["face_id"]
        )
        bounds_da.assign_attrs(
            {
                "latitude_intervalsIndex": intervalsIndex,
                "latitude_intervals_name_map": df_intervals_map,
            }
        )

    if return_array:
        return bounds_da
    else:
        grid._ds["bounds"] = bounds_da


@njit(cache=True, parallel=True)
def _construct_face_bounds_array(
    face_node_connectivity,
    n_nodes_per_face,
    node_x,
    node_y,
    node_z,
    node_lon,
    node_lat,
    is_latlonface: bool = False,
    is_face_GCA_list=None,
):
    """Computes an array of face bounds."""
    n_face = face_node_connectivity.shape[0]
    bounds_array = np.empty((n_face, 2, 2), dtype=np.float64)

    node_lon = np.deg2rad(node_lon)
    node_lat = np.deg2rad(node_lat)

    # Iterate over each face in parallel
    for face_idx in prange(n_face):
        # 1) Create the Cartesian Face Edge Nodes for the current face
        cur_face_edge_nodes_cartesian = _get_cartesian_face_edge_nodes(
            face_idx, face_node_connectivity, n_nodes_per_face, node_x, node_y, node_z
        )
        # 2) Create the Spherical Face Edge Nodes for the current face
        cur_face_edge_nodes_spherical = _get_spherical_face_edge_nodes(
            face_idx, face_node_connectivity, n_nodes_per_face, node_lon, node_lat
        )

        # TODO: This isn't currently used for the user API, consider updating in the future
        if is_face_GCA_list is not None:
            is_GCA_list = is_face_GCA_list[face_idx]
        else:
            is_GCA_list = None

        # 3) Populate the bounds for the current face
        bounds_array[face_idx] = _construct_face_bounds(
            cur_face_edge_nodes_cartesian,
            cur_face_edge_nodes_spherical,
            is_latlonface=is_latlonface,
            is_GCA_list=is_GCA_list,
        )

    return bounds_array


@njit(cache=True)
def _construct_face_bounds(
    face_edges_xyz,
    face_edges_lonlat,
    is_latlonface=False,
    is_GCA_list=None,
):
    """Compute the bounds of a single face."""
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
            pole_point_lonlat[1] = np.pi / 2  # [0.0, pi/2]
            new_pt_latlon[0] = np.pi / 2  # [pi/2, nan]
            new_pt_latlon[1] = np.nan
        else:
            pole_point_xyz[2] = -1.0  # [0.0, 0.0, -1.0]
            pole_point_lonlat[1] = -np.pi / 2  # [0.0, -pi/2]
            new_pt_latlon[0] = -np.pi / 2  # [-pi/2, nan]
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
                n1_cart,
                n2_cart,
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
                face_latlon_array[0, 1] = np.pi / 2  # Upper latitude bound
            else:
                face_latlon_array = insert_pt_in_latlonbox(
                    face_latlon_array, np.array([lat_max, node1_lon_rad])
                )
                face_latlon_array[0, 0] = -np.pi / 2  # Lower latitude bound

        # Adjust longitude bounds globally if the pole is centrally inside the polygon
        if is_center_pole:
            face_latlon_array[1, 0] = 0.0
            face_latlon_array[1, 1] = 2 * np.pi

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
        lon_pt = lon_pt % (2.0 * np.pi)

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
        if not np.isnan(lon_pt) and (lon_pt < 0.0 or lon_pt > 2.0 * np.pi):
            raise Exception("Longitude point out of range")

        # Check for pole points
        is_pole_point = False
        if np.isnan(lon_pt):
            if any_close_lat(lat_pt, ERROR_TOLERANCE):
                is_pole_point = True

        if is_pole_point:
            # Update latitude for pole points
            if np.isclose(lat_pt, 0.5 * np.pi, ERROR_TOLERANCE):
                latlon_box[0, 1] = 0.5 * np.pi  # Update max_lat for North Pole
            elif np.isclose(lat_pt, -0.5 * np.pi, ERROR_TOLERANCE):
                latlon_box[0, 0] = -0.5 * np.pi  # Update min_lat for South Pole
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
