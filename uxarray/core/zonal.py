import numpy as np

from uxarray.grid.area import calculate_face_area
from uxarray.grid.integrate import _zonal_face_weights, _zonal_face_weights_robust
from uxarray.grid.intersections import (
    gca_const_lat_intersection,
    get_number_of_intersections,
)
from uxarray.grid.utils import _get_cartesian_face_edge_nodes_array


def _compute_non_conservative_zonal_mean(uxda, latitudes, use_robust_weights=False):
    """Computes the non-conservative zonal mean across one or more latitudes."""
    import dask.array as da

    uxgrid = uxda.uxgrid
    n_nodes_per_face = uxgrid.n_nodes_per_face.values

    face_axis = uxda.get_axis_num("n_face")

    shape = list(uxda.shape)
    shape[face_axis] = len(latitudes)

    if isinstance(uxda.data, da.Array):
        # Create a Dask array for storing results
        result = da.zeros(shape, dtype=uxda.dtype)
    else:
        # Create a NumPy array for storing results
        result = np.zeros(shape, dtype=uxda.dtype)

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
        z = np.sin(np.deg2rad(lat))

        fe = faces_edge_nodes_xyz[face_indices]
        nn = n_nodes_per_face[face_indices]
        b = bounds[face_indices]

        if use_robust_weights:
            w = _zonal_face_weights_robust(fe, z, b)["weight"].to_numpy()
        else:
            w = _zonal_face_weights(fe, b, nn, z)

        total = w.sum()

        data_slice = uxda.isel(n_face=face_indices, ignore_grid=True).data
        w_shape = [1] * data_slice.ndim
        w_shape[face_axis] = w.size
        w_reshaped = w.reshape(w_shape)
        weighted = (data_slice * w_reshaped).sum(axis=face_axis) / total

        idx = [slice(None)] * result.ndim
        idx[face_axis] = i
        result[tuple(idx)] = weighted

    return result


def _compute_band_overlap_area(
    face_edges_xyz, z_min, z_max, quadrature_rule="gaussian", order=4
):
    """Compute overlap area between a face and latitude band using area.py functions.

    This function finds the intersection polygon between a face and latitude band,
    then uses calculate_face_area from area.py with latitude_adjusted_area=True
    for accurate area computation when edges lie on constant latitude lines.

    Parameters
    ----------
    face_edges_xyz : array-like
        Cartesian coordinates of face edge nodes
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
    # Collect all intersection and interior points to form the overlap polygon
    polygon_points = []

    # Find intersections with z_min and z_max boundaries
    for z_boundary in [z_min, z_max]:
        for e in range(face_edges_xyz.shape[0]):
            edge = face_edges_xyz[e]
            inter = gca_const_lat_intersection(edge, z_boundary)
            nint = get_number_of_intersections(inter)
            for i in range(nint):
                polygon_points.append(inter[i])

    # Add face vertices that lie within the band
    for e in range(face_edges_xyz.shape[0]):
        vertex = face_edges_xyz[e, 0]  # First point of edge
        if z_min <= vertex[2] <= z_max:
            polygon_points.append(vertex)

    if len(polygon_points) < 3:
        return 0.0

    # Remove duplicate points
    unique_points = []
    tolerance = 1e-10
    for pt in polygon_points:
        is_duplicate = False
        for existing_pt in unique_points:
            if np.linalg.norm(pt - existing_pt) < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(pt)

    if len(unique_points) < 3:
        return 0.0

    # Sort points to form a proper polygon
    unique_points = np.array(unique_points)

    # Compute centroid and sort by angle
    center = np.mean(unique_points, axis=0)
    center = center / np.linalg.norm(center)

    angles = []
    for pt in unique_points:
        # Use longitude for sorting (works well for latitude bands)
        angle = np.arctan2(pt[1], pt[0])
        angles.append(angle)

    sorted_indices = np.argsort(angles)
    sorted_points = unique_points[sorted_indices]

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
                # Use area.py functions with latitude adjustment for accurate area calculation
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
