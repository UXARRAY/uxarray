import xarray as xr

import uxarray.conventions.ugrid as ugrid


def _read_topology(node_lon, node_lat, face_node_connectivity, fill_value,
                   start_index, **kwargs):

    ds = xr.DataArray()

    ds['node_lon'] = xr.DataArray(data=node_lon,
                                  dims=ugrid.NODE_DIMS,
                                  attrs=ugrid.NODE_LON_ATTRS)

    ds['node_lat'] = xr.DataArray(data=node_lat,
                                  dims=ugrid.NODE_DIMS,
                                  attrs=ugrid.NODE_LAT_ATTRS)

    if "edge_lon" in kwargs:
        # store edge coordinates, if present
        ds['edge_lon'] = xr.DataArray(data=kwargs['edge_lon'],
                                      dims=ugrid.EDGE_DIMS,
                                      attrs=ugrid.EDGE_LON_ATTRS)

        ds['edge_lat'] = xr.DataArray(data=kwargs['edge_lat'],
                                      dims=ugrid.EDGE_DIMS,
                                      attrs=ugrid.EDGE_LAT_ATTRS)

    if "face_lon" in kwargs:
        # store face coordinates, if present
        ds['face_lon'] = xr.DataArray(data=kwargs['face_lon'],
                                      dims=ugrid.FACE_DIMS,
                                      attrs=ugrid.FACE_LON_ATTRS)

        ds['face_lat'] = xr.DataArray(data=kwargs['face_lat'],
                                      dims=ugrid.FACE_DIMS,
                                      attrs=ugrid.FACE_LAT_ATTRS)

    # TODO: Connectivity

    return ds
