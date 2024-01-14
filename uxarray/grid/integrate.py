"""uxarray grid module."""

import numpy as np
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.intersections import gca_constLat_intersection
from uxarray.grid.arcs import node_xyz_to_lonlat_rad
import pandas as pd


def _get_zonal_faces_weight_at_constLat(
    faces_edges_cart,
    latitude_cart,
    face_latlon_bound,
    is_directed=False,
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

    is_face_GCA_list : np.ndarray, optional (default=None)
        A list of boolean values that indicates if the edge in that face is a GCA. Shape: (n_faces,n_edges).
        True means edge face is a GCA.
        False mean this edge is a constant latitude.
        If None, all edges are considered as GCA.

    Returns
    -------
    weights_df : pandas.DataFrame
        A DataFrame with the calculated weights of each face. The DataFrame has two columns:
        - 'face_index': The index of the face (integer).
        - 'weight': The calculated weight of the face in radian (float).
        The DataFrame is indexed by the face indices, providing a mapping from each face to its corresponding weight.
    """
    if is_face_GCA_list is None:
        is_face_GCA_list = np.ones(faces_edges_cart.shape[:2], dtype=bool)

    intervals_list = []

    # Iterate through all faces and their edges
    for face_index, (face_edges, is_GCA_list) in enumerate(
        zip(faces_edges_cart, is_face_GCA_list)
    ):
        face_interval_df = _get_zonal_face_interval(
            face_edges,
            latitude_cart,
            face_latlon_bound[face_index],
            is_directed=is_directed,
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


def _get_zonal_face_interval(
    face_edges_cart,
    latitude_rad,
    face_latlon_bound,
    is_directed=False,
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
    latitude_rad : float
        The latitude in radians.
    face_latlon_bound : np.ndarray
        The latitude and longitude bounds of the face. Shape: (2, 2), [[lat_min, lat_max], [lon_min, lon_max]]
    is_directed : bool, optional
        If True, the GCA is considered to be directed (from v0 to v1). If False, the GCA is undirected,
        and the smaller circle (less than 180 degrees) side of the GCA is used. Default is False.
    is_GCA_list : np.ndarray, optional
        An array indicating if each edge is a GCA (True) or a constant latitude (False). Shape: (n_edges,).
        If None, all edges are considered as GCAs. Default is None.

    Returns
    -------
    Intervals_df : pd.DataFrame
        A DataFrame containing the intervals stored as pandas Intervals for the face.
        The columns of the DataFrame are: ['start', 'end']
    """
    pt_lon_min = 3 * np.pi
    pt_lon_max = -3 * np.pi
    latZ = np.sin(latitude_rad)

    if is_GCA_list is None:
        is_GCA_list = np.ones(len(face_edges_cart), dtype=bool)

    overlap_flag = False
    overlap_edge = np.array([])

    intersections_pts_list_cart = []
    face_lon_bound_left, face_lon_bound_right = face_latlon_bound[1]
    interval_rows = []  # List to store interval data for DataFrame

    for edge_idx, edge in enumerate(face_edges_cart):
        n1, n2 = edge
        is_GCA = is_GCA_list[edge_idx]

        # Check if the edge is on the equator within the error tolerance
        if np.isclose(n1[2], 0.0, rtol=0, atol=ERROR_TOLERANCE) and np.isclose(
            n2[2], 0.0, rtol=0, atol=ERROR_TOLERANCE
        ):
            # This is a constant latitude edge
            is_GCA = False

        # Check if the edge is overlapped with the constant latitude within the error tolerance
        if (
            np.isclose(n1[2], latZ, rtol=0, atol=ERROR_TOLERANCE)
            and np.isclose(n2[2], latZ, rtol=0, atol=ERROR_TOLERANCE)
            and is_GCA is False
        ):
            overlap_flag = True

        # Skip the dummy edge
        if np.any(n1 == [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE]) or np.any(
            n2 == [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE]
        ):
            continue

        if is_GCA:
            intersections = gca_constLat_intersection(
                [n1, n2], latitude_rad, is_directed=is_directed
            )
            if intersections.size == 0:
                # The constant latitude didn't cross this edge
                continue
            elif intersections.shape[0] == 2:
                # The constant latitude goes across this edge twice
                intersections_pts_list_cart.append(intersections[0])
                intersections_pts_list_cart.append(intersections[1])
            else:
                intersections_pts_list_cart.append(intersections[0])
        else:
            # This is a constant latitude edge, we just need to check if the edge is overlapped with the constant latitude
            if overlap_flag:
                intersections_pts_list_cart = [n1, n2]
                overlap_edge = np.array([n1, n2])
                break

    # If an edge of a face is overlapped by the constant lat, then it will have 4 non-unique intersection pts
    # (A point is counted twice for two edges)
    unique_intersection = np.unique(intersections_pts_list_cart, axis=0)
    if len(unique_intersection) == 2:
        # The normal convex case:
        # Convert the intersection points back to lonlat
        unique_intersection_lonlat = np.array(
            [node_xyz_to_lonlat_rad(pt.tolist()) for pt in unique_intersection]
        )
        [pt_lon_min, pt_lon_max] = np.sort(
            [unique_intersection_lonlat[0][0], unique_intersection_lonlat[1][0]]
        )
    elif len(unique_intersection) != 0:
        # The concave cases
        raise ValueError(
            "UXarray doesn't support concave face with intersections points as ["
            + str(len(unique_intersection))
            + "] currently, please modify your grids accordingly"
        )
    elif len(unique_intersection) == 0:
        # No intersections are found in this face
        raise ValueError(
            "No intersections are found for the face , please make sure the buil_latlon_box generates the correct results"
        )
    cur_face_mag_rad = 0.0

    # Calculate the weight of the face in radian
    if face_lon_bound_left < face_lon_bound_right:
        # Normal case, The interval is not across the 0-lon
        interval_rows.append({"start": pt_lon_min, "end": pt_lon_max})
        cur_face_mag_rad = pt_lon_max - pt_lon_min
    else:
        # Longitude wrap-around
        if pt_lon_max >= np.pi and pt_lon_min >= np.pi:
            # They're both on the "left side" of the 0-lon
            interval_rows.append({"start": pt_lon_min, "end": pt_lon_max})
            cur_face_mag_rad = pt_lon_max - pt_lon_min
        if 0 <= pt_lon_max <= np.pi and 0 <= pt_lon_min <= np.pi:
            # They're both on the "right side" of the 0-lon
            interval_rows.append({"start": pt_lon_min, "end": pt_lon_max})
            cur_face_mag_rad = pt_lon_max - pt_lon_min
        else:
            # They're at the different side of the 0-lon
            interval_rows.append({"start": pt_lon_max, "end": 2 * np.pi})
            interval_rows.append({"start": 0.0, "end": pt_lon_min})
            cur_face_mag_rad = 2 * np.pi - pt_lon_max + pt_lon_min
    if np.abs(cur_face_mag_rad) >= 2 * np.pi:
        print(
            "Problematic face: the face span is "
            + str(cur_face_mag_rad)
            + ". The span should be less than 2pi"
        )

    if overlap_flag:
        # If the overlap_flag is true, check if we have the overlap_edge, if not, raise an error
        if overlap_edge.shape[0] != 2:
            raise ValueError(
                "The overlap_flag is true, but the overlap_edge size is "
                + str(overlap_edge.size)
                + " instead of 2, please check the code"
            )

    # Creating DataFrame from interval_rows
    Intervals_df = pd.DataFrame(interval_rows, columns=["start", "end"])
    return Intervals_df


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
