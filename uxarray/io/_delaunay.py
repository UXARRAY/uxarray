import numpy as np
import xarray as xr

from uxarray.conventions import ugrid
from uxarray.grid.geometry import stereographic_projection


def _spherical_delaunay_from_points(points, boundary_points=None):
    """Generates a spherical Delaunay triangulation from given points,
    excluding triangles where all three nodes are boundary points."""
    from scipy.spatial import ConvexHull

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

    # Compute the 3D Convex hull of Cartesian points on the surface of a sphere
    hull = ConvexHull(points)

    # Obtain delaunay triangles
    triangles = hull.simplices

    if boundary_points is not None:
        # Create a boolean mask where True indicates triangles that do not have all nodes as boundary points
        mask = ~np.isin(triangles, boundary_points).all(axis=1)

        # Apply the mask to filter out unwanted triangles
        triangles = triangles[mask]

    # Assign the filtered triangles to the Dataset
    out_ds["face_node_connectivity"] = xr.DataArray(
        data=triangles,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    # Extract node coordinates
    node_x = points[:, 0]
    node_y = points[:, 1]
    node_z = points[:, 2]

    # Convert Cartesian coordinates to spherical coordinates (longitude and latitude in degrees)
    node_lon_rad = np.arctan2(node_y, node_x)
    node_lat_rad = np.arcsin(node_z / np.linalg.norm(points, axis=1))

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


def _regional_delaunay_from_points(points, boundary_points=None):
    """Generates a regional Delaunay triangulation from given points,
    excluding triangles where all three nodes are boundary points."""
    from scipy.spatial import Delaunay

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

    node_x = points[:, 0]
    node_y = points[:, 1]
    node_z = points[:, 2]

    # Convert Cartesian coordinates to spherical coordinates (longitude and latitude in degrees)
    node_lon_rad = np.arctan2(node_y, node_x)
    node_lat_rad = np.arcsin(node_z / np.linalg.norm(points, axis=1))

    node_lon = np.degrees(node_lon_rad)
    node_lat = np.degrees(node_lat_rad)

    x_plane, y_plane = stereographic_projection(node_lon, node_lat, 0, 0)

    points = np.column_stack((x_plane, y_plane))

    delaunay = Delaunay(points)

    # Obtain delaunay triangles
    triangles = delaunay.simplices

    if boundary_points is not None:
        # Create a boolean mask where True indicates triangles that do not have all nodes as boundary points
        mask = ~np.isin(triangles, boundary_points).all(axis=1)

        # Apply the mask to filter out unwanted triangles
        triangles = triangles[mask]

    # Assign the filtered triangles to the Dataset
    out_ds["face_node_connectivity"] = xr.DataArray(
        data=triangles,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    # Assign node coordinates to the Dataset
    out_ds["node_lon"] = xr.DataArray(
        data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )

    out_ds["node_lat"] = xr.DataArray(
        data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    return out_ds
