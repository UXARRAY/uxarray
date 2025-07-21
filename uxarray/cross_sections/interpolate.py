import numpy as np
import xarray as xr
from numba import njit, prange

from uxarray.constants import INT_DTYPE


@njit(parallel=True)
def _fill_numba(flat_orig, face_idx, n_face, n_samples):
    M = flat_orig.shape[0]
    out = np.full((M, n_samples), np.nan, flat_orig.dtype)
    for i in prange(n_samples):
        f = face_idx[i]
        if 0 <= f < n_face:
            out[:, i] = flat_orig[:, f]
    return out


def _compute_face_idx(uxda, lons, lats):
    pts = np.column_stack((lons, lats))
    faces = uxda.uxgrid.get_faces_containing_point(pts, return_counts=False)
    return np.array([row[0] if row else -1 for row in faces], dtype=INT_DTYPE)


def _interpolate(uxda, lons, lats, new_dim):
    n_samples = len(lons)
    # dims setup
    orig_dims = list(uxda.dims)
    face_axis = orig_dims.index("n_face")
    new_dims = [new_dim if d == "n_face" else d for d in orig_dims]
    dim_axis = new_dims.index(new_dim)

    # face‐index lookup
    face_idx = _compute_face_idx(uxda, lons, lats)

    # output shape
    # sizes = dict(uxda.sizes)
    # shape = [n_samples if d == new_dim else sizes[d] for d in new_dims]

    # coords: keep all except 'n_face'
    coords = {d: uxda.coords[d] for d in uxda.coords if d != "n_face"}
    coords[new_dim] = (new_dim, lons if new_dim == "lon" else lats)
    const_name = "lat" if new_dim == "lon" else "lon"
    coords[const_name] = lats[0] if new_dim == "lon" else lons[0]

    # move face axis to last and flatten
    arr_last = np.moveaxis(uxda.data, face_axis, -1)
    flat_orig = arr_last.reshape(-1, arr_last.shape[-1])  # (M, n_face)

    flat_filled = _fill_numba(
        flat_orig, face_idx, arr_last.shape[-1], n_samples
    )  # (M, sample)
    filled_last = flat_filled.reshape(*arr_last.shape[:-1], n_samples)
    data = np.moveaxis(filled_last, -1, dim_axis)

    return xr.DataArray(
        data, dims=new_dims, coords=coords, name=uxda.name, attrs=uxda.attrs
    )


def interpolate_along_constant_latitude(uxda, lat, lons):
    """Interpolates data that intersects a line of constant latitude onto samples residing on that line of constant latitude."""
    lats = np.full_like(lons, lat)
    return _interpolate(uxda, lons, lats, new_dim="lon")


def interpolate_along_constant_longitude(uxda, lon, lats):
    """Interpolates data that intersects a line of constant longitude onto samples residing on that line of constant longitude."""
    lons = np.full_like(lats, lon)
    return _interpolate(uxda, lons, lats, new_dim="lat")
