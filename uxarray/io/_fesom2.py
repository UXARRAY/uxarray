import pandas as pd
import numpy as np
import xarray as xr

from uxarray.conventions import ugrid


def _read_fesom2_asci(grid_path):
    source_dims_dict = {}
    ugrid_ds = xr.Dataset()

    nodes = pd.read_csv(
        grid_path + "/nod2d.out",
        delim_whitespace=True,
        skiprows=1,
        names=["node_number", "x", "y", "flag"],
    )

    x2 = nodes.x.values
    x2 = np.where(x2 > 180, x2 - 360, x2)
    y2 = nodes.y.values

    ugrid_ds["node_lon"] = xr.DataArray(
        data=x2, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )
    ugrid_ds["node_lat"] = xr.DataArray(
        data=y2, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    file_content = pd.read_csv(
        grid_path + "/elem2d.out",
        delim_whitespace=True,
        skiprows=1,
        names=["first_elem", "second_elem", "third_elem"],
    )

    elem = file_content.values - 1

    ugrid_ds["face_node_connectivity"] = xr.DataArray(
        data=elem,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    return ugrid_ds, source_dims_dict


def _read_fesom2_netcdf():
    pass
