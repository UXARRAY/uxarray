import xarray as xr
import numpy as np

from uxarray.constants import INT_DTYPE
from uxarray.conventions import ugrid


def _read_geos_cs(in_ds: xr.Dataset):
    out_ds = xr.Dataset()

    node_lon = in_ds["corner_lons"].values.ravel()
    node_lat = in_ds["corner_lats"].values.ravel()

    out_ds["node_lon"] = xr.DataArray(
        data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )

    out_ds["node_lat"] = xr.DataArray(
        data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    nf, nx, ny = in_ds["corner_lons"].shape

    # Generate indices for all corner nodes
    idx = np.arange(nx * ny * nf, dtype=INT_DTYPE).reshape(nf, nx, ny)

    # Calculate indices of corner nodes for each face
    tl = idx[:, :-1, :-1]
    tr = idx[:, :-1, 1:]
    bl = idx[:, 1:, :-1]
    br = idx[:, 1:, 1:]

    # Reshape indices for concatenation
    tl = tl.reshape(-1)
    tr = tr.reshape(-1)
    bl = bl.reshape(-1)
    br = br.reshape(-1)

    # Concatenate corner node indices for all faces
    face_node_connectivity = np.column_stack((br, bl, tl, tr))

    out_ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_connectivity,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )
