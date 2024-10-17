import numpy as np
import xarray as xr
from scipy.spatial import ConvexHull
from uxarray.conventions import ugrid


def _spherical_delaunay_from_points(points):
    """TODO: Docstring"""
    out_ds = xr.Dataset()

    # 3D Convex hull of Cartesian points on the surface of a sphere
    hull = ConvexHull(points)

    triangles = hull.simplices

    out_ds["face_node_connectivity"] = xr.DataArray(
        data=triangles,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    node_x = points[:, 0]
    node_y = points[:, 1]
    node_z = points[:, 2]

    node_lon_rad = np.arctan2(node_y, node_x)
    node_lat_rad = np.arcsin(node_z)

    node_lon = np.degrees(node_lon_rad)
    node_lat = np.degrees(node_lat_rad)

    out_ds["node_lon"] = xr.DataArray(
        data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )

    out_ds["node_lat"] = xr.DataArray(
        data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    return out_ds
