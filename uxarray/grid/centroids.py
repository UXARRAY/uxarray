import xarray as xr
import numpy as np
from uxarray.grid.coordinates import node_xyz_to_lonlat_rad, node_lonlat_rad_to_xyz


def centroid_from_mean_verts(self, repopulate=True):
    """Finds the centroids of faces based on the mean value of the vertices."""

    latlon = False
    if "Mesh2_node_z" not in self._ds:
        latlon = True
        lon = self.Mesh2_node_x.values
        lat = self.Mesh2_node_y.values
        [node_x, node_y, node_z] = node_lonlat_rad_to_xyz([lon, lat])
    else:
        node_x = self.Mesh2_node_cart_x.values
        node_y = self.Mesh2_node_cart_y.values
        node_z = self.Mesh2_node_cart_z.values

    face_nodes = self.Mesh2_face_nodes.values
    nNodes_per_face = self.nNodes_per_face.values

    mesh2_face_x = []
    mesh2_face_y = []
    mesh2_face_z = []

    for cur_face_nodes, n_nodes in zip(face_nodes, nNodes_per_face):
        mesh2_face_x.append(np.mean(node_x[cur_face_nodes[:n_nodes]]))
        mesh2_face_y.append(np.mean(node_y[cur_face_nodes[:n_nodes]]))
        mesh2_face_z.append(np.mean(node_z[cur_face_nodes[:n_nodes]]))

    if latlon:
        [centroid_lon, centroid_lat
        ] = node_xyz_to_lonlat_rad([mesh2_face_x, mesh2_face_y, mesh2_face_z])

        if "Mesh2_face_x" not in self._ds or repopulate:
            self._ds["Mesh2_face_x"] = xr.DataArray(
                centroid_lon,
                dims=["nMesh2_face"],
                attrs={"standard_name": "degrees_east"})
        if "Mesh2_face_y" not in self._ds or repopulate:
            self._ds["Mesh2_face_y"] = xr.DataArray(
                centroid_lat,
                dims=["nMesh2_face"],
                attrs={"standard_name": "degrees_north"})
    else:
        if "Mesh2_face_x" not in self._ds or repopulate:
            self._ds["Mesh2_face_x"] = xr.DataArray(
                mesh2_face_x,
                dims=["nMesh2_face"],
                attrs={"standard_name": "degrees_east"})
        if "Mesh2_face_y" not in self._ds or repopulate:
            self._ds["Mesh2_face_y"] = xr.DataArray(
                mesh2_face_y,
                dims=["nMesh2_face"],
                attrs={"standard_name": "degrees_north"})
        if "Mesh2_face_z" not in self._ds or repopulate:
            self._ds["Mesh2_face_z"] = xr.DataArray(
                mesh2_face_z,
                dims=["nMesh2_face"],
                attrs={"standard_name": "elevation"})
