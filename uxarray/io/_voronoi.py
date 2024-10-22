import xarray as xr
from uxarray.constants import INT_FILL_VALUE
from uxarray.conventions import ugrid

import numpy as np
from scipy.spatial import SphericalVoronoi


def _spherical_voronoi_from_points(points, boundary_points=None, **kwargs):
    """TODO:"""
    out_ds = xr.Dataset()

    # Validate boundary_points if provided
    if boundary_points is not None:
        boundary_points = np.asarray(boundary_points)
        if boundary_points.ndim != 1:
            raise ValueError(
                "boundary_points must be a 1D array-like of point indices."
            )
        if np.any(boundary_points < 0) or np.any(boundary_points >= len(points)):
            raise ValueError("boundary_points contain invalid indices.")

    # Create the Spherical Voronoi object
    sv = SphericalVoronoi(points, **kwargs)

    # Sort the vertices for each region
    sv.sort_vertices_of_regions()

    # Retrieve the vertices (Cartesian coordinates)
    vertices = sv.vertices  # Shape: (M, 3)

    # Convert vertices to spherical coordinates (lon, lat in degrees)
    node_lon_rad = np.arctan2(vertices[:, 1], vertices[:, 0])
    node_lat_rad = np.arcsin(
        vertices[:, 2] / np.linalg.norm(vertices, axis=1)
    )  # Normalize if necessary

    node_lon = np.rad2deg(node_lon_rad)
    node_lat = np.rad2deg(node_lat_rad)

    # Retrieve the regions (list of lists of vertex indices)
    voronoi_regions = (
        sv.regions
    )  # List of regions; each region is a list of vertex indices

    # Determine which regions to keep
    if boundary_points is not None:
        # Exclude regions corresponding to boundary points
        # Each region in SphericalVoronoi corresponds to an input point
        # Thus, regions to exclude are those with indices in boundary_points
        total_regions = len(voronoi_regions)
        if total_regions != len(points):
            raise ValueError(
                f"Number of Voronoi regions ({total_regions}) does not match number of input points ({len(points)})."
            )
        # Create a mask to keep regions not in boundary_points
        keep_region_mask = np.ones(total_regions, dtype=bool)
        keep_region_mask[boundary_points] = False

        # Filter regions
        filtered_regions = [
            region for i, region in enumerate(voronoi_regions) if keep_region_mask[i]
        ]

        if len(filtered_regions) == 0:
            raise ValueError(
                "All regions have been excluded. No regions remain after excluding boundary points."
            )
    else:
        # No boundary points provided; keep all regions
        filtered_regions = voronoi_regions

    # Identify all vertex indices used by the filtered regions
    used_vertex_indices = set(
        vertex for region in filtered_regions for vertex in region
    )

    # Create a mapping from old vertex indices to new indices
    # Only keep vertices that are used by at least one region
    used_vertex_indices_sorted = sorted(used_vertex_indices)
    old_to_new_vertex_map = {
        old_idx: new_idx for new_idx, old_idx in enumerate(used_vertex_indices_sorted)
    }

    # Update face_node_connectivity with new vertex indices
    max_num_nodes = (
        max(len(face) for face in filtered_regions) if filtered_regions else 0
    )

    face_node_connectivity = np.full(
        (len(filtered_regions), max_num_nodes), INT_FILL_VALUE, dtype=int
    )

    for i, face in enumerate(filtered_regions):
        for j, vertex in enumerate(face):
            face_node_connectivity[i, j] = old_to_new_vertex_map[vertex]

    # Assign the filtered and reindexed triangles to the Dataset
    out_ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_connectivity,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    # Extract and assign node coordinates for used vertices
    filtered_node_lon = node_lon[used_vertex_indices_sorted]
    filtered_node_lat = node_lat[used_vertex_indices_sorted]

    out_ds["node_lon"] = xr.DataArray(
        data=filtered_node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )

    out_ds["node_lat"] = xr.DataArray(
        data=filtered_node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    return out_ds
