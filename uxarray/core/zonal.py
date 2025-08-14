import numpy as np

from uxarray.grid.arcs import compute_arc_length
from uxarray.grid.area import get_gauss_quadrature_dg
from uxarray.grid.integrate import _zonal_face_weights, _zonal_face_weights_robust
from uxarray.grid.intersections import (
    gca_const_lat_intersection,
    get_number_of_intersections,
)
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


def _band_overlap_area_for_face(face_edges_xyz, z_min, z_max, order=3):
    """Approximate overlap area between a face and latitude band [z_min,z_max].

    Integrates A = ∫_{phi1}^{phi2} L(phi) cos(phi) dphi, where L(phi) is the
    intersection length of the face with the constant-latitude circle at phi.
    Here z = sin(phi), so phi = arcsin(z) and dz = cos(phi) dphi.

    We change variable to z and integrate ∫ L(arcsin(z)) dz using Gauss–Legendre.
    """
    # Use Gaussian quadrature nodes/weights on [0,1], then scale to [z_min,z_max]
    dG, dW = get_gauss_quadrature_dg(order)
    xi = dG[0]  # nodes on [0,1]
    wi = dW  # weights for [0,1]
    half = 0.5 * (z_max - z_min)
    mid = 0.5 * (z_max - z_min) + z_min

    acc = 0.0
    for t, w in zip(xi, wi):
        z = mid + half * (2.0 * t - 1.0)  # map [0,1] -> [-1,1] -> [z_min,z_max]
        pts = []
        for e in range(face_edges_xyz.shape[0]):
            edge = face_edges_xyz[e]
            inter = gca_const_lat_intersection(edge, z)
            nint = get_number_of_intersections(inter)
            if nint == 1:
                pts.append(inter[0])
            elif nint == 2:
                pts.append(inter[0])
                pts.append(inter[1])

        if len(pts) == 0:
            L = 0.0
        else:
            used = [False] * len(pts)
            L = 0.0
            for i in range(len(pts)):
                if used[i]:
                    continue
                best_j = -1
                best_d2 = 1e300
                for j in range(len(pts)):
                    if i == j or used[j]:
                        continue
                    d2 = float(((pts[i] - pts[j]) ** 2).sum())
                    if d2 < best_d2:
                        best_d2 = d2
                        best_j = j
                if best_j >= 0:
                    L += compute_arc_length(pts[i], pts[best_j])
                    used[i] = True
                    used[best_j] = True
        acc += (2.0 * w) * L

    area = half * acc
    return area


def _compute_conservative_zonal_mean_bands(uxda, bands):
    import dask.array as da

    uxgrid = uxda.uxgrid
    face_axis = uxda.get_axis_num("n_face")

    faces_edge_nodes_xyz = _get_cartesian_face_edge_nodes_array(
        uxgrid.face_node_connectivity.values,
        uxgrid.n_face,
        uxgrid.n_max_face_nodes,
        uxgrid.node_x.values,
        uxgrid.node_y.values,
        uxgrid.node_z.values,
    )
    n_nodes_per_face = uxgrid.n_nodes_per_face.values

    bands = np.asarray(bands, dtype=float)
    if bands.ndim != 1 or bands.size < 2:
        raise ValueError("bands must be 1D with at least two edges")

    nb = bands.size - 1

    shape = list(uxda.shape)
    shape[face_axis] = nb
    if isinstance(uxda.data, da.Array):
        result = da.zeros(shape, dtype=uxda.dtype)
    else:
        result = np.zeros(shape, dtype=uxda.dtype)

    fb = uxgrid.face_bounds_lat.values

    for bi in range(nb):
        lat0 = float(np.clip(bands[bi], -90.0, 90.0))
        lat1 = float(np.clip(bands[bi + 1], -90.0, 90.0))
        z0 = np.sin(np.deg2rad(lat0))
        z1 = np.sin(np.deg2rad(lat1))
        zmin, zmax = (z0, z1) if z0 <= z1 else (z1, z0)

        mask = ~((fb[:, 1] < min(lat0, lat1)) | (fb[:, 0] > max(lat0, lat1)))
        face_idx = np.nonzero(mask)[0]
        if face_idx.size == 0:
            continue

        overlap = np.zeros(face_idx.size, dtype=float)
        for k, fi in enumerate(face_idx):
            nedge = n_nodes_per_face[fi]
            fe = faces_edge_nodes_xyz[fi, :nedge]
            overlap[k] = _band_overlap_area_for_face(fe, zmin, zmax, order=3)

        data_slice = uxda.isel(n_face=face_idx, ignore_grid=True).data
        w = overlap
        total = w.sum()
        if total == 0.0:
            weighted = np.nan * data_slice[..., 0]
        else:
            w_shape = [1] * data_slice.ndim
            w_shape[face_axis] = w.size
            w_reshaped = w.reshape(w_shape)
            weighted = (data_slice * w_reshaped).sum(axis=face_axis) / total

        idx = [slice(None)] * result.ndim
        idx[face_axis] = bi
        result[tuple(idx)] = weighted

    return result
