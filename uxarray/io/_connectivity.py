import xarray as xr
import numpy as np

from uxarray.constants import INT_FILL_VALUE, INT_DTYPE


def _read_face_vertices(face_vertices, latlon):
    """Create a grid with faces constructed from vertices specified by the
    given argument.

    Parameters
    ----------
    dataset : ndarray, list, tuple, required
        Input vertex coordinates that form our face(s)
    """
    grid_ds = xr.Dataset()
    grid_ds["Mesh2"] = xr.DataArray(
        attrs={
            "cf_role": "mesh_topology",
            "long_name": "Topology data of unstructured mesh",
            "topology_dimension": -1,
            "node_coordinates": "Mesh2_node_x Mesh2_node_y Mesh2_node_z",
            "node_dimension": "nMesh2_node",
            "face_node_connectivity": "Mesh2_face_nodes",
            "face_dimension": "nMesh2_face"
        })

    grid_ds.Mesh2.attrs['topology_dimension'] = face_vertices.ndim

    if latlon:
        x_units = "degrees_east"
        y_units = "degrees_north"
    else:
        x_units = 'm'
        y_units = 'm'
        z_units = 'm'

    x_coord = face_vertices[:, :, 0].flatten()
    y_coord = face_vertices[:, :, 1].flatten()

    if face_vertices[0][0].size > 2:
        z_coord = face_vertices[:, :, 2].flatten()
    else:
        z_coord = x_coord * 0.0

    # Identify unique vertices and their indices
    unique_verts, indices = np.unique(face_vertices.reshape(
        -1, face_vertices.shape[-1]),
                                      axis=0,
                                      return_inverse=True)

    # Nodes index that contain a fill value
    fill_value_mask = np.logical_or(unique_verts[:, 0] == INT_FILL_VALUE,
                                    unique_verts[:, 1] == INT_FILL_VALUE)
    if face_vertices[0][0].size > 2:
        fill_value_mask = np.logical_or(unique_verts[:, 0] == INT_FILL_VALUE,
                                        unique_verts[:, 1] == INT_FILL_VALUE,
                                        unique_verts[:, 2] == INT_FILL_VALUE)

    # Get the indices of all the False values in fill_value_mask
    false_indices = np.where(fill_value_mask == True)[0]

    # Check if any False values were found
    indices = indices.astype(INT_DTYPE)
    if false_indices.size > 0:

        # Remove the rows corresponding to False values in unique_verts
        unique_verts = np.delete(unique_verts, false_indices, axis=0)

        # Update indices accordingly
        for i, idx in enumerate(false_indices):
            indices[indices == idx] = INT_FILL_VALUE
            indices[(indices > idx) & (indices != INT_FILL_VALUE)] -= 1

    if latlon:
        grid_ds["Mesh2_node_x"] = xr.DataArray(data=unique_verts[:, 0],
                                               dims=["nMesh2_node"],
                                               attrs={"units": x_units})
        grid_ds["Mesh2_node_y"] = xr.DataArray(data=unique_verts[:, 1],
                                               dims=["nMesh2_node"],
                                               attrs={"units": y_units})
    else:
        grid_ds["Mesh2_node_cart_x"] = xr.DataArray(data=unique_verts[:, 0],
                                                    dims=["nMesh2_node"],
                                                    attrs={"units": x_units})

        grid_ds["Mesh2_node_cart_y"] = xr.DataArray(data=unique_verts[:, 1],
                                                    dims=["nMesh2_node"],
                                                    attrs={"units": y_units})

        if face_vertices.shape[-1] > 2:
            grid_ds["Mesh2_node_cart_z"] = xr.DataArray(
                data=unique_verts[:, 2],
                dims=["nMesh2_node"],
                attrs={"units": z_units})
        else:
            grid_ds["Mesh2_node_cart_z"] = xr.DataArray(
                data=unique_verts[:, 1] * 0.0,
                dims=["nMesh2_node"],
                attrs={"units": z_units})

    # Create connectivity array using indices of unique vertices
    connectivity = indices.reshape(face_vertices.shape[:-1])
    grid_ds["Mesh2_face_nodes"] = xr.DataArray(
        data=xr.DataArray(connectivity).astype(INT_DTYPE),
        dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
        attrs={
            "cf_role": "face_node_connectivity",
            "_FillValue": INT_FILL_VALUE,
            "start_index": 0
        })

    return grid_ds
