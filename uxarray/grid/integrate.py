import numpy as np
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.intersections import gca_const_lat_intersection
from uxarray.grid.coordinates import _xyz_to_lonlat_rad
import pandas as pd

from uxarray.utils.computing import isclose

DUMMY_EDGE_VALUE = [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE]


def _get_zonal_faces_weight_at_constLat(
    faces_edges_cart_candidate,
    latitude_cart,
    face_latlon_bound_candidate,
    is_directed=False,
    is_latlonface=False,
    is_face_GCA_list=None,
):
    """Utilize the sweep line algorithm to calculate the weight of each face at
    a constant latitude.

     Parameters
    ----------
    faces_edges_cart_candidate : np.ndarray
        A list of the candidate face polygon represented by edges in Cartesian coordinates.
        The faces must be selected in the previous step such that they will be intersected by the constant latitude.
        It should have the same shape as the face_latlon_bound_candidate.
        The input should not contain any 'INT_FILL_VALUE'. Shape: (n_faces(candidate), n_edges, 2, 3)

    latitude_cart : float
        The latitude in Cartesian coordinates (The normalized z coordinate)

    face_latlon_bound_candidate : np.ndarray
        The list of latitude and longitude bounds of candidate faces.
        It should have the same shape as the face_edges_cart_candidate.
        Shape: (n_faces(candidate),,2, 2),
        [...,[lat_min, lat_max], [lon_min, lon_max],...]

    is_directed : bool, optional (default=False)
        If True, the GCA is considered to be directed, which means it can only from v0-->v1. If False, the GCA is undirected,
        and we will always assume the small circle (The one less than 180 degree) side is the GCA.

    is_latlonface : bool, optional, default=False
        A global flag to indicate if faces are latlon face. If True, then treat all faces as latlon faces. Latlon face means
        That all edge is either a longitude or constant latitude line. If False, then all edges are GCA.
         Default is False.

    is_face_GCA_list : np.ndarray, optional (default=None)
        A list of boolean values that indicates if the edge in that face is a GCA. Shape: (n_faces,n_edges).
        True means edge face is a GCA.
        False mean this edge is a constant latitude.
        If None, all edges are considered as GCA. This attribute will overwrite the is_latlonface attribute.

    Returns
    -------
    weights_df : pandas.DataFrame
        A DataFrame with the calculated weights of each face. The DataFrame has two columns:
        - 'face_index': The index of the face (integer).
        - 'weight': The calculated weight of the face in radian (float).
        The DataFrame is indexed by the face indices, providing a mapping from each face to its corresponding weight.

    Notes
    -----
    Special handling is implemented for the cases when the latitude_cart is close to 1 or -1,
    which corresponds to the poles (90 and -90 degrees). In these cases, if a pole point is
    inside a face, that face's value is the only value that should be considered. If the pole
    point is not inside any face, it lies on the boundary of surrounding faces, and their weights
    are considered evenly since they only contain points rather than intervals.
    This treatment is hard-coded in the function and should be tested with appropriate test cases.
    """

    # Special case if the latitude_cart is 1 or -1, meaning right at the pole
    # If the latitude_cart is close to 1 or -1 (indicating the pole), handle it separately.
    # The -90 and 90 treatment is hard-coded in the function, based on:
    # If a pole point is inside a face, then this face's value is the only value that should be considered.
    # If the pole point is not inside any face, then it's on the boundary of faces around it, so their weights are even.
    if isclose(latitude_cart, 1, atol=ERROR_TOLERANCE) or isclose(
        latitude_cart, -1, atol=ERROR_TOLERANCE
    ):
        # Now all candidate faces( the faces around the pole) are considered as the same weight
        # If the face encompases the pole, then the weight is 1
        weights = {
            face_index: 1 / len(faces_edges_cart_candidate)
            for face_index in range(len(faces_edges_cart_candidate))
        }
        # Convert weights to DataFrame
        weights_df = pd.DataFrame(
            list(weights.items()), columns=["face_index", "weight"]
        )
        return weights_df

    intervals_list = []

    # Iterate through all faces and their edges
    for face_index, face_edges in enumerate(faces_edges_cart_candidate):
        # Remove the Int_fill_value from the face_edges
        face_edges = face_edges[np.all(face_edges != INT_FILL_VALUE, axis=(1, 2))]
        if is_face_GCA_list is not None:
            is_GCA_list = is_face_GCA_list[face_index]
        else:
            is_GCA_list = None
        face_interval_df = _get_zonal_face_interval(
            face_edges,
            latitude_cart,
            face_latlon_bound_candidate[face_index],
            is_directed=is_directed,
            is_latlonface=is_latlonface,
            is_GCA_list=is_GCA_list,
        )
        # If any end of the interval is NaN
        if face_interval_df.isnull().values.any():
            # Skip this face as it is just being touched by the constant latitude
            continue
        # Check if the DataFrame is empty (start and end are both 0)
        if (face_interval_df["start"] == 0).all() and (
            face_interval_df["end"] == 0
        ).all():
            # Skip this face as it is just being touched by the constant latitude
            continue
        else:
            for _, row in face_interval_df.iterrows():
                intervals_list.append(
                    {"start": row["start"], "end": row["end"], "face_index": face_index}
                )

    intervals_df = pd.DataFrame(intervals_list)
    try:
        overlap_contributions, total_length = _process_overlapped_intervals(
            intervals_df
        )

        # Calculate weights for each face
        weights = {
            face_index: overlap_contributions.get(face_index, 0.0) / total_length
            for face_index in range(len(faces_edges_cart_candidate))
        }

        # Convert weights to DataFrame
        weights_df = pd.DataFrame(
            list(weights.items()), columns=["face_index", "weight"]
        )
        return weights_df

    except ValueError:
        print(f"Face index: {face_index}")
        print(f"Face edges information: {face_edges}")
        print(f"Constant z0: {latitude_cart}")
        print(
            f"Face latlon bound information: {face_latlon_bound_candidate[face_index]}"
        )
        print(f"Face interval information: {face_interval_df}")
        # Handle the exception or propagate it further if necessary
        raise


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
    face_edges_cart, latitude_cart, is_GCA_list, is_latlonface, is_directed
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
    is_directed : bool
        Flag indicating if the GCA should be considered as directed (from v0 to v1). If False,
        the smaller circle (less than 180 degrees) side of the GCA is used.

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
                intersections = gca_const_lat_intersection(
                    edge, latitude_cart, is_directed=is_directed
                )

                if intersections.size == 0:
                    continue
                elif intersections.shape[0] == 2:
                    intersections_pts_list_cart.extend(intersections)
                else:
                    intersections_pts_list_cart.append(intersections[0])

    # Find the unique intersection points
    unique_intersections = np.unique(intersections_pts_list_cart, axis=0)

    if len(unique_intersections) == 2:
        # TODO: vectorize?
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


def _get_zonal_face_interval(
    face_edges_cart,
    latitude_cart,
    face_latlon_bound,
    is_directed=False,
    is_latlonface=False,
    is_GCA_list=None,
):
    """Processes a face polygon represented by edges in Cartesian coordinates
    to find intervals where the face intersects with a given latitude. This
    function handles directed and undirected Great Circle Arcs (GCAs) and edges
    at constant latitude.

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
    is_directed : bool, optional
        If True, the GCA is considered to be directed (from v0 to v1). If False, the GCA is undirected,
        and the smaller circle (less than 180 degrees) side of the GCA is used. Default is False.
    is_latlonface : bool, optional, default=False
        A global flag to indicate if faces are latlon face. If True, then treat all faces as latlon faces. Latlon face means
        That all edge is either a longitude or constant latitude line. If False, then all edges are GCA.
         Default is False. This attribute will overwrite the is_latlonface attribute.
    is_GCA_list : np.ndarray, optional, default=False
        An array indicating if each edge is a GCA (True) or a constant latitude (False). Shape: (n_edges,).
        If None, all edges are considered as GCAs. Default is None.

    Returns
    -------
    Intervals_df : pd.DataFrame
        A DataFrame containing the intervals stored as pandas Intervals for the face.
        The columns of the DataFrame are: ['start', 'end']
    """
    face_lon_bound_left, face_lon_bound_right = face_latlon_bound[1]

    # Call the vectorized function to process all edges
    try:
        # Call the vectorized function to process all edges
        unique_intersections, pt_lon_min, pt_lon_max = (
            _get_faces_constLat_intersection_info(
                face_edges_cart, latitude_cart, is_GCA_list, is_latlonface, is_directed
            )
        )

        # Handle the special case where the unique_intersections is 1, which means the face is just being touched
        if len(unique_intersections) == 1:
            # If the face is just being touched, then just return the empty DataFrame
            return pd.DataFrame({"start": [0.0], "end": [0.0]}, index=[0])

        # Convert intersection points to longitude-latitude
        longitudes = np.array(
            [_xyz_to_lonlat_rad(*pt.tolist())[0] for pt in unique_intersections]
        )

        # Handle special wrap-around cases by checking the face bounds
        if face_lon_bound_left >= face_lon_bound_right or (
            face_lon_bound_left == 0 and face_lon_bound_right == 2 * np.pi
        ):
            if not (
                (pt_lon_max >= np.pi and pt_lon_min >= np.pi)
                or (0 <= pt_lon_max <= np.pi and 0 <= pt_lon_min <= np.pi)
            ):
                # If the anti-meridian is crossed, instead of just being touched,add the wrap-around points
                if pt_lon_max != 2 * np.pi and pt_lon_min != 0:
                    # They're at different sides of the 0-lon, adding wrap-around points
                    longitudes = np.append(longitudes, [0.0, 2 * np.pi])
                elif pt_lon_max >= np.pi and pt_lon_min == 0:
                    # That means the face is actually from pt_lon_max to 2*pi.
                    # Replace the 0 in longnitude with 2*pi
                    longitudes[longitudes == 0] = 2 * np.pi

        # Ensure longitudes are sorted
        longitudes = np.unique(longitudes)
        longitudes.sort()

        # Split the sorted longitudes into start and end points of intervals
        starts = longitudes[::2]  # Start points
        ends = longitudes[1::2]  # End points

        # Create the intervals DataFrame
        Intervals_df = pd.DataFrame({"start": starts, "end": ends})
        # For consistency, sort the intervals by start
        interval_df_sorted = Intervals_df.sort_values(by="start").reset_index(drop=True)
        return interval_df_sorted

    except ValueError as e:
        default_print_options = np.get_printoptions()
        if (
            str(e)
            == "No intersections are found for the face, please make sure the build_latlon_box generates the correct results"
        ):
            # Set print options for full precision
            np.set_printoptions(precision=16, suppress=False)

            print(
                "ValueError: No intersections are found for the face, please make sure the build_latlon_box generates the correct results"
            )
            print(f"Face edges information: {face_edges_cart}")
            print(f"Constant z_0: {latitude_cart}")
            print(f"Face latlon bound information: {face_latlon_bound}")

            # Reset print options to default
            np.set_printoptions(**default_print_options)

            raise
        else:
            # Set print options for full precision
            np.set_printoptions(precision=17, suppress=False)

            print(f"Face edges information: {face_edges_cart}")
            print(f"Constant z_0: {latitude_cart}")
            print(f"Face latlon bound information: {face_latlon_bound}")

            # Reset print options to default
            np.set_printoptions(**default_print_options)

            raise  # Re-raise the exception if it's not the expected ValueError


def _process_overlapped_intervals(intervals_df):
    """Process the overlapped intervals using the sweep line algorithm,
    considering multiple intervals per face.

    Parameters
    ----------
    intervals_df : pd.DataFrame
        The DataFrame that contains the intervals and the corresponding face index.
        The columns of the DataFrame are: ['start', 'end', 'face_index']

    Returns
    -------
    dict
        A dictionary where keys are face indices and values are their respective contributions to the total length.
    float
        The total length of all intervals considering their overlaps.

    Example
    -------
    >>> intervals_data = [
    ...     {"start": 0.0, "end": 100.0, "face_index": 0},
    ...     {"start": 50.0, "end": 150.0, "face_index": 1},
    ...     {"start": 140.0, "end": 150.0, "face_index": 2},
    ... ]
    >>> intervals_df = pd.DataFrame(intervals_data)
    >>> overlap_contributions, total_length = _process_overlapped_intervals(
    ...     intervals_df
    ... )
    >>> print(overlap_contributions)
    >>> print(total_length)
    """
    events = []
    for idx, row in intervals_df.iterrows():
        events.append((row["start"], "start", row["face_index"]))
        events.append((row["end"], "end", row["face_index"]))

    events.sort(key=lambda x: (x[0], x[1]))

    active_faces = set()
    last_position = None
    total_length = 0.0
    overlap_contributions = {}

    for position, event_type, face_idx in events:
        if face_idx == 51:
            pass
        if last_position is not None and active_faces:
            segment_length = position - last_position
            segment_weight = segment_length / len(active_faces) if active_faces else 0
            for active_face in active_faces:
                overlap_contributions[active_face] = (
                    overlap_contributions.get(active_face, 0) + segment_weight
                )
            total_length += segment_length

        if event_type == "start":
            # use try catch to handle the case where the face_idx is not be able to be added
            try:
                active_faces.add(face_idx)
            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"Face index: {face_idx}")
                print(f"Position: {position}")
                print(f"Event type: {event_type}")
                print(f"Active faces: {active_faces}")
                print(f"Last position: {last_position}")
                print(f"Total length: {total_length}")
                print(f"Overlap contributions: {overlap_contributions}")
                print(f"Intervals data: {intervals_df}")
                raise

        elif event_type == "end":
            if face_idx in active_faces:
                active_faces.remove(face_idx)
            else:
                raise ValueError(
                    f"Error: Trying to remove face_idx {face_idx} which is not in active_faces"
                )

        last_position = position

    return overlap_contributions, total_length
