import xarray as xr
from uxarray.constants import INT_FILL_VALUE
from uxarray.conventions import ugrid

import numpy as np
from scipy.spatial import SphericalVoronoi


def _spherical_voronoi_from_points(points):
    """TODO: Docstring"""
    out_ds = xr.Dataset()

    # Create the Spherical Voronoi object
    sv = SphericalVoronoi(points)

    # Sort the vertices for each region
    sv.sort_vertices_of_regions()

    # Retrieve the vertices (Cartesian coordinates)
    vertices = sv.vertices

    # TODO: Preserve cartesian vertices

    # Convert vertices to spherical coordinates (lon, lat)
    node_lon = np.arctan2(vertices[:, 1], vertices[:, 0])
    node_lat = np.arcsin(vertices[:, 2])

    node_lon = np.rad2deg(node_lon)
    node_lat = np.rad2deg(node_lat)

    out_ds["node_lon"] = xr.DataArray(
        data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )

    out_ds["node_lat"] = xr.DataArray(
        data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    # TODO: original coordinates are preserved as face centers

    # Regions give the indices of vertices forming the polygon for each face
    voronoi_regions = sv.regions

    # Determine the maximum number of nodes per face
    max_num_nodes = max(len(face) for face in voronoi_regions)

    face_node_connectivity = np.full(
        (len(voronoi_regions), max_num_nodes), INT_FILL_VALUE
    )

    # Populate the dense face-node connectivity array
    for i, face in enumerate(voronoi_regions):
        face_node_connectivity[i, : len(face)] = face

    out_ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_connectivity,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    return out_ds
