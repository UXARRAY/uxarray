import numpy as np
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.intersections import gca_const_lat_intersection
from uxarray.grid.coordinates import _xyz_to_lonlat_rad
import polars as pl

from uxarray.utils.computing import isclose

DUMMY_EDGE_VALUE = [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE]


def _is_edge_gca(is_GCA_list, is_latlonface, edges_z):
    """Determine if each edge is a Great Circle Arc (GCA) or a constant
    latitude line in a vectorized manner.

    Parameters:
    ----------
    is_GCA_list : np.ndarray or None
        An array indicating whether each edge is a GCA (True) or a constant latitude line (False).
        Shape: (n_edges). If None, edge types are determined based on `is_latlonface` and the z-coordinates.
    is_latlonface : bool
        Flag indicating if all edges should be considered as lat-lon faces, which implies all edges
        are either constant latitude or longitude lines.
    edges_z : np.ndarray
        Array containing the z-coordinates for each vertex of the edges. This is used to determine
        whether edges are on the equator or if they are aligned in latitude when `is_GCA_list` is None.
        Shape should be (n_edges, 2).

    Returns:
    -------
    np.ndarray
        A boolean array where each element indicates whether the corresponding edge is considered a GCA.
        True for GCA, False for constant latitude line.
    """
    if is_GCA_list is not None:
        return is_GCA_list
    if is_latlonface:
        return ~np.isclose(edges_z[:, 0], edges_z[:, 1], atol=ERROR_TOLERANCE)
    return ~(
        np.isclose(edges_z[:, 0], 0, atol=ERROR_TOLERANCE)
        & np.isclose(edges_z[:, 1], 0, atol=ERROR_TOLERANCE)
    )


def _get_faces_constLat_intersection_info(
    face_edges_cart, latitude_cart, is_GCA_list, is_latlonface
):
    """Processes each edge of a face polygon in a vectorized manner to
    determine overlaps and calculate the intersections for a given latitude and
    the faces.

    Parameters:
    ----------
    face_edges_cart : np.ndarray
        A face polygon represented by edges in Cartesian coordinates. Shape: (n_edges, 2, 3).
    latitude_cart : float
        The latitude in Cartesian coordinates to which intersections or overlaps are calculated.
    is_GCA_list : np.ndarray or None
        An array indicating whether each edge is a GCA (True) or a constant latitude line (False).
        Shape: (n_edges). If None, the function will determine edge types based on `is_latlonface`.
    is_latlonface : bool
        Flag indicating if all faces are considered as lat-lon faces, meaning all edges are either
        constant latitude or longitude lines. This parameter overwrites the `is_GCA_list` if set to True.


    Returns:
    -------
    tuple
        A tuple containing:
        - intersections_pts_list_cart (list): A list of intersection points where each point is where an edge intersects with the latitude.
        - pt_lon_min (float): The min longnitude of the interseted intercal in radian if any; otherwise, None..
        - pt_lon_max (float): The max longnitude of the interseted intercal in radian, if any; otherwise, None.
    """
    valid_edges_mask = ~(np.any(face_edges_cart == DUMMY_EDGE_VALUE, axis=(1, 2)))

    # Apply mask to filter out dummy edges
    valid_edges = face_edges_cart[valid_edges_mask]

    # Extract Z coordinates for edge determination
    edges_z = valid_edges[:, :, 2]

    # Determine if each edge is GCA or constant latitude
    is_GCA = _is_edge_gca(is_GCA_list, is_latlonface, edges_z)

    # Check overlap with latitude
    overlaps_with_latitude = np.all(
        np.isclose(edges_z, latitude_cart, atol=ERROR_TOLERANCE), axis=1
    )
    overlap_flag = np.any(overlaps_with_latitude & ~is_GCA)

    # Identify overlap edges if needed
    intersections_pts_list_cart = []
    if overlap_flag:
        overlap_index = np.where(overlaps_with_latitude & ~is_GCA)[0][0]
        intersections_pts_list_cart.extend(valid_edges[overlap_index])
    else:
        # Calculate intersections (assuming a batch-capable intersection function)
        for idx, edge in enumerate(valid_edges):
            if is_GCA[idx]:
                intersections = gca_const_lat_intersection(edge, latitude_cart)

                if intersections.size == 0:
                    continue
                elif intersections.shape[0] == 2:
                    intersections_pts_list_cart.extend(intersections)
                else:
                    intersections_pts_list_cart.append(intersections[0])

    # Find the unique intersection points
    unique_intersections = np.unique(intersections_pts_list_cart, axis=0)

    if len(unique_intersections) == 2:
        unique_intersection_lonlat = np.array(
            [_xyz_to_lonlat_rad(pt[0], pt[1], pt[2]) for pt in unique_intersections]
        )

        sorted_lonlat = np.sort(unique_intersection_lonlat, axis=0)
        pt_lon_min, pt_lon_max = sorted_lonlat[:, 0]
        return unique_intersections, pt_lon_min, pt_lon_max
    elif len(unique_intersections) == 1:
        return unique_intersections, None, None
    elif len(unique_intersections) != 0 and len(unique_intersections) != 1:
        # If the unique intersections numbers is larger than n_edges * 2, then it means the face is concave
        if len(unique_intersections) > len(valid_edges) * 2:
            raise ValueError(
                "UXarray doesn't support concave face with intersections points as currently, please modify your grids accordingly"
            )
        else:
            # Now return all the intersections points and the pt_lon_min, pt_lon_max
            unique_intersection_lonlat = np.array(
                [_xyz_to_lonlat_rad(pt[0], pt[1], pt[2]) for pt in unique_intersections]
            )

            sorted_lonlat = np.sort(unique_intersection_lonlat, axis=0)
            # Extract the minimum and maximum longitudes
            pt_lon_min, pt_lon_max = (
                np.min(sorted_lonlat[:, 0]),
                np.max(sorted_lonlat[:, 0]),
            )

            return unique_intersections, pt_lon_min, pt_lon_max
    elif len(unique_intersections) == 0:
        raise ValueError(
            "No intersections are found for the face, please make sure the build_latlon_box generates the correct results"
        )


def _get_zonal_faces_weight_at_constLat(
    faces_edges_cart_candidate: np.ndarray,
    latitude_cart: float,
    face_latlon_bound_candidate: np.ndarray,
    is_latlonface: bool = False,
    is_face_GCA_list: np.ndarray | None = None,
) -> pl.DataFrame:
    """
    Utilize the sweep line algorithm to calculate the weight of each face at
    a constant latitude, returning a Polars DataFrame.

    Parameters
    ----------
    faces_edges_cart_candidate : np.ndarray
        A list of the candidate face polygon represented by edges in Cartesian coordinates.
        Shape: (n_faces(candidate), n_edges, 2, 3)
    latitude_cart : float
        The latitude in Cartesian coordinates (the normalized z coordinate).
    face_latlon_bound_candidate : np.ndarray
        An array with shape (n_faces, 2, 2), each face entry like [[lat_min, lat_max],[lon_min, lon_max]].
    is_latlonface : bool, default=False
        Global flag indicating if faces are lat-lon faces (edges are constant lat or long).
    is_face_GCA_list : np.ndarray | None, default=None
        Boolean array (n_faces, n_edges) indicating which edges are GCAs (True) or constant-lat (False).
        If None, all edges are considered GCA.

    Returns
    -------
    weights_df : pl.DataFrame
        DataFrame with columns ["face_index", "weight"], containing the per-face weights
        (as a fraction of the total length of intersection).
    """

    # Special case: latitude_cart close to +1 or -1 (near poles)
    if isclose(latitude_cart, 1, atol=ERROR_TOLERANCE) or isclose(
        latitude_cart, -1, atol=ERROR_TOLERANCE
    ):
        # Evenly distribute weight among candidate faces
        n_faces = len(faces_edges_cart_candidate)
        weights = {face_index: 1.0 / n_faces for face_index in range(n_faces)}
        # Convert dict to Polars DataFrame
        return pl.DataFrame(list(weights.items()), schema=["face_index", "weight"])

    intervals_list = []

    # Iterate over faces
    for face_index, face_edges in enumerate(faces_edges_cart_candidate):
        # Remove edges that contain INT_FILL_VALUE
        face_edges = face_edges[np.all(face_edges != INT_FILL_VALUE, axis=(1, 2))]

        # Which edges are GCA vs constant-lat?
        if is_face_GCA_list is not None:
            is_GCA_list = is_face_GCA_list[face_index]
        else:
            is_GCA_list = None

        # Retrieve intervals for the current face
        face_interval_df = _get_zonal_face_interval(
            face_edges,
            latitude_cart,
            face_latlon_bound_candidate[face_index],
            is_latlonface=is_latlonface,
            is_GCA_list=is_GCA_list,
        )

        # Check if there are any null values in face_interval_df
        has_null = face_interval_df.select(pl.col("*").is_null().any()).row(0)[0]
        if has_null:
            # Skip this face (only "touched" by the latitude)
            continue

        # Check if all start == 0 and all end == 0
        all_start_zero = face_interval_df.select((pl.col("start") == 0).all()).row(0)[0]
        all_end_zero = face_interval_df.select((pl.col("end") == 0).all()).row(0)[0]
        if all_start_zero and all_end_zero:
            # Skip face being merely touched
            continue

        # Add each interval row to intervals_list
        for row in face_interval_df.iter_rows(named=True):
            intervals_list.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "face_index": face_index,
                }
            )

    # Build a Polars DataFrame from intervals
    intervals_df = pl.DataFrame(intervals_list)

    # Process intervals to get overlap contributions
    try:
        overlap_contributions, total_length = _process_overlapped_intervals(
            intervals_df
        )

        # Build final weights dict
        weights = {}
        n_faces = len(faces_edges_cart_candidate)
        for face_index in range(n_faces):
            # fraction of total for this face
            weights[face_index] = (
                overlap_contributions.get(face_index, 0.0) / total_length
            )

        # Return as Polars DataFrame
        weights_df = pl.DataFrame(
            list(weights.items()), schema=["face_index", "weight"], orient="row"
        )
        return weights_df

    except ValueError:
        # If an exception occurs, you can print debug info here if needed
        raise


def _get_zonal_face_interval(
    face_edges_cart: np.ndarray,
    latitude_cart: float,
    face_latlon_bound: np.ndarray,
    is_latlonface: bool = False,
    is_GCA_list: np.ndarray | None = None,
) -> pl.DataFrame:
    """
    Processes a face polygon represented by edges in Cartesian coordinates
    to find intervals where the face intersects with a given latitude. This
    function handles directed and undirected Great Circle Arcs (GCAs) and edges
    at constant latitude, returning a Polars DataFrame with columns ["start", "end"].

    Requires the face edges to be sorted in counter-clockwise order, and the span of the
    face in longitude should be less than pi. Also, all arcs/edges length should be within pi.

    Users can specify which edges are GCAs and which are constant latitude using `is_GCA_list`.
    However, edges on the equator are always treated as constant latitude edges regardless of
    `is_GCA_list`.

    Parameters
    ----------
    face_edges_cart : np.ndarray
        A face polygon represented by edges in Cartesian coordinates. Shape: (n_edges, 2, 3)
    latitude_cart : float
        The latitude in cartesian, the normalized Z coordinates.
    face_latlon_bound : np.ndarray
        The latitude and longitude bounds of the face. Shape: (2, 2), [[lat_min, lat_max], [lon_min, lon_max]]
    is_latlonface : bool, optional, default=False
        A global flag to indicate if faces are latlon face. If True, then treat all faces as latlon faces. Latlon face means
        That all edge is either a longitude or constant latitude line. If False, then all edges are GCA.
         Default is False. This attribute will overwrite the is_latlonface attribute.
    is_GCA_list : np.ndarray, optional, default=False
        An array indicating if each edge is a GCA (True) or a constant latitude (False). Shape: (n_edges,).
        If None, all edges are considered as GCAs. Default is None.

    Returns
    -------
    Intervals_df : pl.DataFrame
        a Polars DataFrame with columns ["start", "end"]
    """

    face_lon_bound_left, face_lon_bound_right = face_latlon_bound[1]

    try:
        unique_intersections, pt_lon_min, pt_lon_max = (
            _get_faces_constLat_intersection_info(
                face_edges_cart, latitude_cart, is_GCA_list, is_latlonface
            )
        )

        # If there's exactly one intersection, the face is only "touched"
        if len(unique_intersections) == 1:
            return pl.DataFrame({"start": [0.0], "end": [0.0]})

        # Convert intersection points to (lon, lat) in radians
        longitudes = np.array(
            [_xyz_to_lonlat_rad(*pt.tolist())[0] for pt in unique_intersections]
        )

        # Handle special wrap-around cases (crossing anti-meridian, etc.)
        if face_lon_bound_left >= face_lon_bound_right or (
            face_lon_bound_left == 0 and face_lon_bound_right == 2 * np.pi
        ):
            if not (
                (pt_lon_max >= np.pi and pt_lon_min >= np.pi)
                or (0 <= pt_lon_max <= np.pi and 0 <= pt_lon_min <= np.pi)
            ):
                if pt_lon_max != 2 * np.pi and pt_lon_min != 0:
                    # Add wrap-around points
                    longitudes = np.append(longitudes, [0.0, 2 * np.pi])
                elif pt_lon_max >= np.pi and pt_lon_min == 0:
                    # If min is 0, but we really need 2*pi
                    longitudes[longitudes == 0] = 2.0 * np.pi

        # Sort unique longitudes
        longitudes = np.unique(longitudes)
        longitudes.sort()

        # Pair sorted longitudes into intervals
        starts = longitudes[::2]
        ends = longitudes[1::2]

        # Create Polars DataFrame
        intervals_df = pl.DataFrame({"start": starts, "end": ends})
        intervals_df_sorted = intervals_df.sort("start")

        return intervals_df_sorted

    except ValueError as e:
        default_print_options = np.get_printoptions()
        if str(e) == (
            "No intersections are found for the face, please make sure the "
            "build_latlon_box generates the correct results"
        ):
            np.set_printoptions(precision=16, suppress=False)
            print(
                "ValueError: No intersections are found for the face, make sure build_latlon_box is correct"
            )
            print(f"Face edges info:\n{face_edges_cart}")
            print(f"Constant z_0: {latitude_cart}")
            print(f"Face latlon bound:\n{face_latlon_bound}")
            np.set_printoptions(**default_print_options)
            raise
        else:
            np.set_printoptions(precision=17, suppress=False)
            print(f"Face edges info:\n{face_edges_cart}")
            print(f"Constant z_0: {latitude_cart}")
            print(f"Face latlon bound:\n{face_latlon_bound}")
            np.set_printoptions(**default_print_options)
            raise


def _process_overlapped_intervals(intervals_df: pl.DataFrame):
    """Process the overlapped intervals using the sweep line algorithm.

    This function processes multiple intervals per face using a sweep line algorithm,
    calculating both individual face contributions and total length while handling
    overlaps. The algorithm moves through sorted interval events (starts and ends),
    maintaining a set of active faces and distributing overlap lengths equally.

    Parameters
    ----------
    intervals_df : pl.DataFrame
        A Polars DataFrame containing the intervals and corresponding face indices.
        Required columns:
            - start : numeric
                Starting position of each interval
            - end : numeric
                Ending position of each interval
            - face_index : int or str
                Identifier for the face associated with each interval

    Returns
    -------
    tuple[dict, float]
        A tuple containing:
        - dict: Maps face indices to their contributions to the total length,
               where overlapping segments are weighted equally among active faces
        - float: The total length of all intervals considering their overlaps
    """

    events = []
    # Iterate Polars rows as dictionaries
    for row in intervals_df.iter_rows(named=True):
        events.append((row["start"], "start", row["face_index"]))
        events.append((row["end"], "end", row["face_index"]))

    # Sort the events by (position, event_type)
    # so that 'start' comes before 'end' if position ties
    events.sort(key=lambda x: (x[0], x[1]))

    active_faces = set()
    last_position = None
    total_length = 0.0
    overlap_contributions = {}

    for position, event_type, face_idx in events:
        if last_position is not None and active_faces:
            segment_length = position - last_position
            # Each face gets an equal share of this segment
            segment_weight = segment_length / len(active_faces)
            for active_face in active_faces:
                overlap_contributions[active_face] = (
                    overlap_contributions.get(active_face, 0.0) + segment_weight
                )
            total_length += segment_length

        if event_type == "start":
            active_faces.add(face_idx)
        elif event_type == "end":
            if face_idx in active_faces:
                active_faces.remove(face_idx)
            else:
                raise ValueError(
                    f"Error: Trying to remove face_idx {face_idx} not in active_faces"
                )

        last_position = position

    return overlap_contributions, total_length
