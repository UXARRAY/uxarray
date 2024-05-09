import numpy as np
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.intersections import gca_constLat_intersection
from uxarray.grid.coordinates import _xyz_to_lonlat_rad
import pandas as pd

DUMMY_EDGE_VALUE = [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE]


def _get_zonal_faces_weight_at_constLat(
    faces_edges_cart,
    latitude_cart,
    face_latlon_bound,
    is_directed=False,
    is_latlonface=False,
    is_face_GCA_list=None,
):
    """Utilize the sweep line algorithm to calculate the weight of each face at
    a constant latitude.

     Parameters
    ----------
    face_edges_cart : np.ndarray
        A list of face polygon represented by edges in Cartesian coordinates. Shape: (n_faces, n_edges, 2, 3)

    latitude_cart : float
        The latitude in Cartesian coordinates (The normalized z coordinate)

    face_latlon_bound : np.ndarray
        The list of latitude and longitude bounds of faces. Shape: (n_faces,2, 2),
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
    """
    intervals_list = []

    # Iterate through all faces and their edges
    for face_index, face_edges in enumerate(faces_edges_cart):
        if is_face_GCA_list is not None:
            is_GCA_list = is_face_GCA_list[face_index]
        else:
            is_GCA_list = None
        face_interval_df = _get_zonal_face_interval(
            face_edges,
            latitude_cart,
            face_latlon_bound[face_index],
            is_directed=is_directed,
            is_latlonface=is_latlonface,
            is_GCA_list=is_GCA_list,
        )
        for _, row in face_interval_df.iterrows():
            intervals_list.append(
                {"start": row["start"], "end": row["end"], "face_index": face_index}
            )

    intervals_df = pd.DataFrame(intervals_list)
    overlap_contributions, total_length = _process_overlapped_intervals(intervals_df)

    # Calculate weights for each face
    weights = {
        face_index: overlap_contributions.get(face_index, 0.0) / total_length
        for face_index in range(len(faces_edges_cart))
    }

    # Convert weights to DataFrame
    weights_df = pd.DataFrame(list(weights.items()), columns=["face_index", "weight"])
    return weights_df


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
        - overlap_flag (bool): A boolean indicating if any overlap with the latitude was detected.
        - overlap_edge (np.ndarray or None): The edge that overlaps with the latitude, if any; otherwise, None.
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
                intersections = gca_constLat_intersection(
                    edge, latitude_cart, is_directed=is_directed
                )
                if intersections.size == 0:
                    continue
                elif intersections.shape[0] == 2:
                    intersections_pts_list_cart.extend(intersections)
                else:
                    intersections_pts_list_cart.append(intersections[0])

    # Handle unique intersections and check for convex or concave cases
    unique_intersections = np.unique(intersections_pts_list_cart, axis=0)
    if len(unique_intersections) == 2:
        # TODO: vectorize?
        unique_intersection_lonlat = np.array(
            [_xyz_to_lonlat_rad(pt[0], pt[1], pt[2]) for pt in unique_intersections]
        )

        sorted_lonlat = np.sort(unique_intersection_lonlat, axis=0)
        pt_lon_min, pt_lon_max = sorted_lonlat[:, 0]
        return unique_intersections, pt_lon_min, pt_lon_max
    elif len(unique_intersections) != 0:
        raise ValueError(
            "UXarray doesn't support concave face with intersections points as currently, please modify your grids accordingly"
        )
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
    (
        unique_intersections,
        pt_lon_min,
        pt_lon_max,
    ) = _get_faces_constLat_intersection_info(
        face_edges_cart, latitude_cart, is_GCA_list, is_latlonface, is_directed
    )

    # Convert intersection points to longitude-latitude
    longitudes = np.array(
        [_xyz_to_lonlat_rad(*pt.tolist())[0] for pt in unique_intersections]
    )

    # Handle special wrap-around cases by checking the face bounds
    if face_lon_bound_left >= face_lon_bound_right:
        if not (
            (pt_lon_max >= np.pi and pt_lon_min >= np.pi)
            or (0 <= pt_lon_max <= np.pi and 0 <= pt_lon_min <= np.pi)
        ):
            # They're at different sides of the 0-lon, adding wrap-around points
            longitudes = np.append(longitudes, [0.0, 2 * np.pi])

    # Ensure longitudes are sorted
    longitudes.sort()

    # Split the sorted longitudes into start and end points of intervals
    starts = longitudes[::2]  # Start points
    ends = longitudes[1::2]  # End points

    # Create the intervals DataFrame
    Intervals_df = pd.DataFrame({"start": starts, "end": ends})
    # For consistency, sort the intervals by start
    interval_df_sorted = Intervals_df.sort_values(by="start").reset_index(drop=True)
    return interval_df_sorted


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
    ...     {'start': 0.0, 'end': 100.0, 'face_index': 0},
    ...     {'start': 50.0, 'end': 150.0, 'face_index': 1},
    ...     {'start': 140.0, 'end': 150.0, 'face_index': 2}
    ... ]
    >>> intervals_df = pd.DataFrame(intervals_data)
    >>> overlap_contributions, total_length = _process_overlapped_intervals(intervals_df)
    >>> print(overlap_contributions)
    >>> print(total_length)
    """
    events = []
    for idx, row in intervals_df.iterrows():
        events.append((row["start"], "start", row["face_index"]))
        events.append((row["end"], "end", row["face_index"]))

    events.sort()  # Sort by position and then by start/end

    active_faces = set()
    last_position = None
    total_length = 0.0
    overlap_contributions = {}

    for position, event_type, face_idx in events:
        if last_position is not None and active_faces:
            segment_length = position - last_position
            segment_weight = segment_length / len(active_faces) if active_faces else 0
            for active_face in active_faces:
                overlap_contributions[active_face] = (
                    overlap_contributions.get(active_face, 0) + segment_weight
                )
            total_length += segment_length

        if event_type == "start":
            active_faces.add(face_idx)
        elif event_type == "end":
            active_faces.remove(face_idx)

        last_position = position

    return overlap_contributions, total_length
