import xarray as xr
import numpy as np

from uxarray.constants import INT_DTYPE
from uxarray.conventions import ugrid


def _read_geos_cs(in_ds: xr.Dataset):
    """Reads and encodes a GEOS Cube-Sphere grid into the UGRID conventions.

    https://gmao.gsfc.nasa.gov/gmaoftp/ops/GEOSIT_sample/doc/CS_Description_c180_v1.pdf
    """
    out_ds = xr.Dataset()

    node_lon = in_ds["corner_lons"].values.ravel()
    node_lat = in_ds["corner_lats"].values.ravel()

    out_ds["node_lon"] = xr.DataArray(
        data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )

    out_ds["node_lat"] = xr.DataArray(
        data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    if "lons" in in_ds:
        face_lon = in_ds["lons"].values.ravel()
        face_lat = in_ds["lats"].values.ravel()

        out_ds["face_lon"] = xr.DataArray(
            data=face_lon, dims=ugrid.FACE_DIM, attrs=ugrid.FACE_LON_ATTRS
        )

        out_ds["face_lat"] = xr.DataArray(
            data=face_lat, dims=ugrid.FACE_DIM, attrs=ugrid.FACE_LAT_ATTRS
        )

    nf, nx, ny = in_ds["corner_lons"].shape

    # generate indices for all corner nodes
    idx = np.arange(nx * ny * nf, dtype=INT_DTYPE).reshape(nf, nx, ny)

    # calculate indices of corner nodes for each face
    tl = idx[:, :-1, :-1].reshape(-1)
    tr = idx[:, :-1, 1:].reshape(-1)
    bl = idx[:, 1:, :-1].reshape(-1)
    br = idx[:, 1:, 1:].reshape(-1)

    # Concatenate corner node indices for all faces
    face_node_connectivity = np.column_stack((br, bl, tl, tr))

    out_ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_connectivity,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    # GEOS-CS does not return a source_dims_dict
    return out_ds, None
