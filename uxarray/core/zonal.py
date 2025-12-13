import numpy as np
from numba import njit

from uxarray.grid.area import calculate_face_area
from uxarray.grid.geometry import _unique_points
from uxarray.grid.integrate import _zonal_face_weights, _zonal_face_weights_robust
from uxarray.grid.intersections import (
    gca_const_lat_intersection,
    get_number_of_intersections,
)
from uxarray.grid.utils import (
    _get_cartesian_face_edge_nodes_array,
    _small_angle_of_2_vectors,
)


def _compute_non_conservative_zonal_mean(uxda, latitudes, use_robust_weights=False):
    """Computes the non-conservative zonal mean across one or more latitudes."""
    import dask.array as da

    uxgrid = uxda.uxgrid
    n_nodes_per_face = uxgrid.n_nodes_per_face.values

    face_axis = uxda.get_axis_num("n_face")

    shape = list(uxda.shape)
    shape[face_axis] = len(latitudes)

    if np.issubdtype(uxda.dtype, np.integer) or np.issubdtype(uxda.dtype, np.bool_):
        # Promote integers/bools so we can represent NaNs
        result_dtype = np.float64
    else:
        # Preserve existing float/complex dtype
        result_dtype = uxda.dtype

    if isinstance(uxda.data, da.Array):
        # Pre-fill with NaNs so empty slices stay missing without extra work
        result = da.full(shape, np.nan, dtype=result_dtype)
    else:
        # Create a NumPy array for storing results
        result = np.full(shape, np.nan, dtype=result_dtype)

    faces_edge_nodes_xyz = _get_cartesian_face_edge_nodes_array(
        uxgrid.face_node_connectivity.values,
        uxgrid.n_face,
        uxgrid.n_max_face_nodes,
        uxgrid.node_x.values,
        uxgrid.node_y.values,
        uxgrid.node_z.values,
    )

    bounds = uxgrid.bounds.values

    for i, lat in enumerate(latitudes):
        face_indices = uxda.uxgrid.get_faces_at_constant_latitude(lat)

        idx = [slice(None)] * result.ndim
        idx[face_axis] = i

        if face_indices.size == 0:
            # No intersecting faces for this latitude
            continue

        z = np.sin(np.deg2rad(lat))

        fe = faces_edge_nodes_xyz[face_indices]
        nn = n_nodes_per_face[face_indices]
        b = bounds[face_indices]

        if use_robust_weights:
            w = _zonal_face_weights_robust(fe, z, b)["weight"].to_numpy()
        else:
            w = _zonal_face_weights(fe, b, nn, z)

        total = w.sum()

        if total == 0.0 or not np.isfinite(total):
            # If weights collapse to zero, keep the pre-filled NaNs
            continue

        data_slice = uxda.isel(n_face=face_indices, ignore_grid=True).data
        w_shape = [1] * data_slice.ndim
        w_shape[face_axis] = w.size
        w_reshaped = w.reshape(w_shape)
        weighted = (data_slice * w_reshaped).sum(axis=face_axis) / total

        result[tuple(idx)] = weighted

    return result


@njit(cache=True)
def _sort_points_by_angle(points):
    """Sort points by longitude angle for proper polygon formation.

    Parameters
    ----------
    points : ndarray
        Array of 3D points on unit sphere, shape (n_points, 3)

    Returns
    -------
    ndarray
        Points sorted by longitude angle
    """
    n_points = points.shape[0]
    if n_points <= 1:
        return points.copy()

    # Calculate angles (longitude)
    angles = np.empty(n_points, dtype=np.float64)
    x_axis = np.array([1.0, 0.0, 0.0])
    for i in range(n_points):
        # Project point to xy plane for longitude calculation
        point_xy = np.array([points[i, 0], points[i, 1], 0.0])
        point_xy_norm = np.linalg.norm(point_xy)

        if point_xy_norm < 1e-15:
            angles[i] = 0.0  # Point at pole
        else:
            point_xy_unit = point_xy / point_xy_norm
            angle = _small_angle_of_2_vectors(x_axis, point_xy_unit)
            # Determine sign based on y coordinate
            if points[i, 1] < 0:
                angle = -angle
            angles[i] = angle

    # Simple insertion sort (numba-friendly for small arrays)
    sorted_points = points.copy()
    for i in range(1, n_points):
        key_angle = angles[i]
        key_point = sorted_points[i].copy()
        j = i - 1

        while j >= 0 and angles[j] > key_angle:
            angles[j + 1] = angles[j]
            sorted_points[j + 1] = sorted_points[j]
            j -= 1

        angles[j + 1] = key_angle
        sorted_points[j + 1] = key_point

    return sorted_points


@njit(cache=True)
def _compute_band_overlap_area(
    face_edges_xyz, z_min, z_max, quadrature_rule="gaussian", order=4
):
    """Compute overlap area between a face and latitude band using area.py functions.

    This function finds the intersection polygon between a face and latitude band,
    then uses calculate_face_area from area.py with latitude_adjusted_area=True
    for accurate area computation when edges lie on constant latitude lines.

    Parameters
    ----------
    face_edges_xyz : ndarray
        Cartesian coordinates of face edge nodes, shape (n_edges, 2, 3)
    z_min, z_max : float
        Z-coordinate bounds of the latitude band (z = sin(latitude))
    quadrature_rule : str, optional
        Quadrature rule to use ("gaussian" or "triangular", default: "gaussian")
    order : int, optional
        Quadrature order (default: 4, same as area.py)

    Returns
    -------
    float
        Overlap area between face and latitude band
    """
    # Pre-allocate for maximum possible intersection points
    # Worst case: 2 intersections per edge * 2 boundaries + all vertices
    max_points = face_edges_xyz.shape[0] * 4 + face_edges_xyz.shape[0]
    polygon_points = np.empty((max_points, 3), dtype=np.float64)
    point_count = 0

    # Find intersections with z_min and z_max boundaries
    z_boundaries = np.array([z_min, z_max])
    for z_boundary in z_boundaries:
        for e in range(face_edges_xyz.shape[0]):
            edge = face_edges_xyz[e]
            inter = gca_const_lat_intersection(edge, z_boundary)
            nint = get_number_of_intersections(inter)

            for i in range(nint):
                if point_count < max_points:
                    polygon_points[point_count] = inter[i]
                    point_count += 1

    # Add face vertices that lie within the band
    for e in range(face_edges_xyz.shape[0]):
        vertex = face_edges_xyz[e, 0]  # First point of edge
        if z_min <= vertex[2] <= z_max:
            if point_count < max_points:
                polygon_points[point_count] = vertex
                point_count += 1

    if point_count < 3:
        return 0.0

    # Remove duplicate points
    unique_points = _unique_points(polygon_points[:point_count])

    if unique_points.shape[0] < 3:
        return 0.0

    # Sort points to form a proper polygon
    sorted_points = _sort_points_by_angle(unique_points)

    # Use area.py calculate_face_area with latitude adjustment
    x = sorted_points[:, 0]
    y = sorted_points[:, 1]
    z = sorted_points[:, 2]

    area, _ = calculate_face_area(
        x,
        y,
        z,
        quadrature_rule=quadrature_rule,
        order=order,
        latitude_adjusted_area=True,  # Key improvement: use latitude adjustment
    )

    return area


def _compute_conservative_zonal_mean_bands(uxda, bands):
    """
    Compute conservative zonal mean over latitude bands.

    Uses get_faces_between_latitudes to optimize computation by avoiding
    overlap area calculations for fully contained faces.

    Parameters
    ----------
    uxda : UxDataArray
        The data array to compute zonal means for
    bands : array-like
        Latitude band edges in degrees

    Returns
    -------
    result : array
        Zonal means for each band
    """
    import dask.array as da

    uxgrid = uxda.uxgrid
    face_axis = uxda.get_axis_num("n_face")

    # Pre-compute face properties
    faces_edge_nodes_xyz = _get_cartesian_face_edge_nodes_array(
        uxgrid.face_node_connectivity.values,
        uxgrid.n_face,
        uxgrid.n_max_face_nodes,
        uxgrid.node_x.values,
        uxgrid.node_y.values,
        uxgrid.node_z.values,
    )
    n_nodes_per_face = uxgrid.n_nodes_per_face.values
    face_bounds_lat = uxgrid.face_bounds_lat.values
    face_areas = uxgrid.face_areas.values

    bands = np.asarray(bands, dtype=float)
    if bands.ndim != 1 or bands.size < 2:
        raise ValueError("bands must be 1D with at least two edges")

    nb = bands.size - 1

    # Initialize result array
    shape = list(uxda.shape)
    shape[face_axis] = nb
    if isinstance(uxda.data, da.Array):
        result = da.zeros(shape, dtype=uxda.dtype)
    else:
        result = np.zeros(shape, dtype=uxda.dtype)

    for bi in range(nb):
        lat0 = float(np.clip(bands[bi], -90.0, 90.0))
        lat1 = float(np.clip(bands[bi + 1], -90.0, 90.0))

        # Ensure lat0 <= lat1
        if lat0 > lat1:
            lat0, lat1 = lat1, lat0

        z0 = np.sin(np.deg2rad(lat0))
        z1 = np.sin(np.deg2rad(lat1))
        zmin, zmax = (z0, z1) if z0 <= z1 else (z1, z0)

        # Step 1: Get fully contained faces
        fully_contained_faces = uxgrid.get_faces_between_latitudes((lat0, lat1))

        # Step 2: Get all overlapping faces (including partial)
        mask = ~((face_bounds_lat[:, 1] < lat0) | (face_bounds_lat[:, 0] > lat1))
        all_overlapping_faces = np.nonzero(mask)[0]

        if all_overlapping_faces.size == 0:
            # No faces in this band
            idx = [slice(None)] * result.ndim
            idx[face_axis] = bi
            result[tuple(idx)] = np.nan
            continue

        # Step 3: Partition faces into fully contained vs partially overlapping
        is_fully_contained = np.isin(all_overlapping_faces, fully_contained_faces)
        partially_overlapping_faces = all_overlapping_faces[~is_fully_contained]

        # Step 4: Compute weights
        all_weights = np.zeros(all_overlapping_faces.size, dtype=float)

        # For fully contained faces, use their full area
        if fully_contained_faces.size > 0:
            fully_contained_indices = np.where(is_fully_contained)[0]
            all_weights[fully_contained_indices] = face_areas[fully_contained_faces]

        # For partially overlapping faces, compute fractional area
        if partially_overlapping_faces.size > 0:
            partial_indices = np.where(~is_fully_contained)[0]
            for i, face_idx in enumerate(partially_overlapping_faces):
                nedge = n_nodes_per_face[face_idx]
                face_edges = faces_edge_nodes_xyz[face_idx, :nedge]
                overlap_area = _compute_band_overlap_area(face_edges, zmin, zmax)
                all_weights[partial_indices[i]] = overlap_area

        # Step 5: Compute weighted average
        data_slice = uxda.isel(n_face=all_overlapping_faces, ignore_grid=True).data
        total_weight = all_weights.sum()

        if total_weight == 0.0:
            weighted = np.nan * data_slice[..., 0]
        else:
            w_shape = [1] * data_slice.ndim
            w_shape[face_axis] = all_weights.size
            w_reshaped = all_weights.reshape(w_shape)
            weighted = (data_slice * w_reshaped).sum(axis=face_axis) / total_weight

        idx = [slice(None)] * result.ndim
        idx[face_axis] = bi
        result[tuple(idx)] = weighted

    return result
