import numpy as np
import xarray as xr

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.conventions import ugrid


def _spherical_voronoi_from_points(points, **kwargs):
    """Generates a spherical Delaunay Voronoi diagram from given points."""
    from scipy.spatial import SphericalVoronoi

    if not np.allclose(np.linalg.norm(points, axis=1), 1.0):
        raise ValueError("All input points must lie on the unit sphere.")

    out_ds = xr.Dataset()

    # Create the Spherical Voronoi object
    sv = SphericalVoronoi(points, **kwargs)

    # Sort the vertices for each region
    sv.sort_vertices_of_regions()

    # Retrieve the vertices (Cartesian coordinates)
    vertices = sv.vertices

    # Convert vertices to spherical coordinates (lon, lat in degrees)
    node_lon = np.rad2deg(np.arctan2(vertices[:, 1], vertices[:, 0]))
    node_lat = np.rad2deg(np.arcsin(vertices[:, 2]))  # Assuming unit sphere

    # Retrieve the regions (list of lists of vertex indices)
    voronoi_regions = sv.regions

    # Determine the maximum number of nodes per face
    max_num_nodes = max(len(face) for face in voronoi_regions) if voronoi_regions else 0

    # Initialize the face_node_connectivity array with fill values
    face_node_connectivity = np.full(
        (len(voronoi_regions), max_num_nodes), INT_FILL_VALUE, dtype=INT_DTYPE
    )

    for i, region in enumerate(voronoi_regions):
        face_length = len(region)
        face_node_connectivity[i, :face_length] = region

    # Assign face_node_connectivity to the dataset
    out_ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_connectivity,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    out_ds["node_lon"] = xr.DataArray(
        data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )

    out_ds["node_lat"] = xr.DataArray(
        data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    return out_ds
