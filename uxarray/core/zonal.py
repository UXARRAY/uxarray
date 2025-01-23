import numpy as np
import dask.array as da


from uxarray.grid.integrate import _zonal_face_weights, _zonal_face_weights_robust
from uxarray.grid.utils import _get_cartesian_face_edge_nodes


def _compute_non_conservative_zonal_mean(uxda, latitudes, use_robust_weights=False):
    """Computes the non-conservative zonal mean across one or more latitudes."""
    uxgrid = uxda.uxgrid
    n_nodes_per_face = uxgrid.n_nodes_per_face.values
    shape = uxda.shape[:-1] + (len(latitudes),)
    if isinstance(uxda.data, da.Array):
        # Create a Dask array for storing results
        result = da.zeros(shape, dtype=uxda.dtype)
    else:
        # Create a NumPy array for storing results
        result = np.zeros(shape, dtype=uxda.dtype)

    faces_edge_nodes_xyz = _get_cartesian_face_edge_nodes(
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

        faces_edge_nodes_xyz_candidate = faces_edge_nodes_xyz[face_indices, :, :, :]

        n_nodes_per_face_candidate = n_nodes_per_face[face_indices]

        bounds_candidate = bounds[face_indices]

        if use_robust_weights:
            weights = _zonal_face_weights_robust(
                faces_edge_nodes_xyz_candidate, z, bounds_candidate
            )["weight"].to_numpy()
        else:
            weights = _zonal_face_weights(
                faces_edge_nodes_xyz_candidate,
                bounds_candidate,
                n_nodes_per_face_candidate,
                z,
            )

        total_weight = weights.sum()
        result[..., i] = ((uxda.data[..., face_indices] * weights) / total_weight).sum(
            axis=-1
        )

    return result
