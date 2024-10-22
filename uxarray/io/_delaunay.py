import numpy as np
import xarray as xr
from scipy.spatial import ConvexHull
from uxarray.conventions import ugrid


def _spherical_delaunay_from_points(points, boundary_points=None):
    """TODO:"""
    out_ds = xr.Dataset()

    # Validate boundary_points if provided
    if boundary_points is not None:
        if boundary_points.ndim != 1:
            raise ValueError(
                "boundary_points must be a 1D array-like of point indices."
            )
        if np.any(boundary_points < 0) or np.any(boundary_points >= len(points)):
            raise ValueError("boundary_points contain invalid indices.")

    # Compute the 3D Convex hull of Cartesian points on the surface of a sphere
    hull = ConvexHull(points)

    triangles = hull.simplices  # Shape: (n_triangles, 3)

    # If boundary_points are provided, filter out triangles containing any boundary points
    if boundary_points is not None:
        # Create a boolean mask where True indicates triangles that do NOT contain any boundary points
        mask = ~np.isin(triangles, boundary_points).any(axis=1)

        # Apply the mask to filter out unwanted triangles
        filtered_triangles = triangles[mask]
    else:
        # No boundary points provided; keep all triangles
        filtered_triangles = triangles

    # If boundary_points are provided, exclude them from the node list
    if boundary_points is not None:
        # Create a mask for nodes that are not boundary points
        node_mask = ~np.isin(np.arange(len(points)), boundary_points)

        # Get the indices of nodes to keep
        kept_node_indices = np.nonzero(node_mask)[0]

        # Create a mapping from old indices to new indices
        # Initialize with -1 for all nodes
        index_mapping = -np.ones(len(points), dtype=int)
        # Assign new indices only to kept nodes
        index_mapping[kept_node_indices] = np.arange(len(kept_node_indices))

        # Check if any triangles remain after filtering
        if len(filtered_triangles) == 0:
            raise ValueError("No triangles remain after excluding boundary points.")

        # Update triangle indices to the new node numbering
        new_triangles = index_mapping[filtered_triangles]

        # Ensure no invalid indices remain (shouldn't happen due to prior filtering)
        if np.any(new_triangles == -1):
            raise ValueError(
                "Filtered triangles contain boundary points after reindexing."
            )

        # Assign the filtered and reindexed triangles to the Dataset
        out_ds["face_node_connectivity"] = xr.DataArray(
            data=new_triangles,
            dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
            attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
        )

        # Extract and assign node coordinates excluding boundary points
        filtered_points = points[kept_node_indices]

    else:
        # No boundary points; use all triangles and nodes
        out_ds["face_node_connectivity"] = xr.DataArray(
            data=filtered_triangles,
            dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
            attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
        )
        filtered_points = points

    # Extract node coordinates
    node_x = filtered_points[:, 0]
    node_y = filtered_points[:, 1]
    node_z = filtered_points[:, 2]

    # Convert Cartesian coordinates to spherical coordinates (longitude and latitude in degrees)
    node_lon_rad = np.arctan2(node_y, node_x)
    node_lat_rad = np.arcsin(
        node_z / np.linalg.norm(filtered_points, axis=1)
    )  # Ensure points are on the unit sphere

    node_lon = np.degrees(node_lon_rad)
    node_lat = np.degrees(node_lat_rad)

    # Assign node coordinates to the Dataset
    out_ds["node_lon"] = xr.DataArray(
        data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )

    out_ds["node_lat"] = xr.DataArray(
        data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    return out_ds
