import numpy as np

from uxarray.grid.integrate import _zonal_face_weights, _zonal_face_weights_robust
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
