import numpy as np
import xarray as xr

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


def _read_face_vertices(face_vertices, latlon):
    """Create a grid with faces constructed from vertices specified by the
    given argument.

    Parameters
    ----------
    dataset : ndarray, list, tuple, required
        Input vertex coordinates that form our face(s)
    """
    grid_ds = xr.Dataset()

    if latlon:
        x_units = "degrees_east"
        y_units = "degrees_north"
    else:
        x_units = "m"
        y_units = "m"
        z_units = "m"

    # x_coord = face_vertices[:, :, 0].flatten()
    # y_coord = face_vertices[:, :, 1].flatten()
    #
    # if face_vertices[0][0].size > 2:
    #     z_coord = face_vertices[:, :, 2].flatten()
    # else:
    #     z_coord = x_coord * 0.0

    # Identify unique vertices and their indices
    unique_verts, indices = np.unique(
        face_vertices.reshape(-1, face_vertices.shape[-1]), axis=0, return_inverse=True
    )

    # Nodes index that contain a fill value
    fill_value_mask = np.logical_or(
        unique_verts[:, 0] == INT_FILL_VALUE, unique_verts[:, 1] == INT_FILL_VALUE
    )
    if face_vertices[0][0].size > 2:
        fill_value_mask = np.logical_or(
            unique_verts[:, 0] == INT_FILL_VALUE,
            unique_verts[:, 1] == INT_FILL_VALUE,
            unique_verts[:, 2] == INT_FILL_VALUE,
        )

    # Get the indices of all the False values in fill_value_mask
    false_indices = np.where(fill_value_mask)[0]

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
        grid_ds["node_lon"] = xr.DataArray(
            data=unique_verts[:, 0], dims=["n_node"], attrs={"units": x_units}
        )
        grid_ds["node_lat"] = xr.DataArray(
            data=unique_verts[:, 1], dims=["n_node"], attrs={"units": y_units}
        )
    else:
        grid_ds["node_x"] = xr.DataArray(
            data=unique_verts[:, 0], dims=["n_node"], attrs={"units": x_units}
        )

        grid_ds["node_y"] = xr.DataArray(
            data=unique_verts[:, 1], dims=["n_node"], attrs={"units": y_units}
        )

        if face_vertices.shape[-1] > 2:
            grid_ds["node_z"] = xr.DataArray(
                data=unique_verts[:, 2], dims=["n_node"], attrs={"units": z_units}
            )
        else:
            grid_ds["node_z"] = xr.DataArray(
                data=unique_verts[:, 1] * 0.0, dims=["n_node"], attrs={"units": z_units}
            )

    # Create connectivity array using indices of unique vertices
    connectivity = indices.reshape(face_vertices.shape[:-1])
    grid_ds["face_node_connectivity"] = xr.DataArray(
        data=xr.DataArray(connectivity).astype(INT_DTYPE),
        dims=["n_face", "n_max_face_nodes"],
        attrs={
            "cf_role": "face_node_connectivity",
            "_FillValue": INT_FILL_VALUE,
            "start_index": 0,
        },
    )

    return grid_ds
