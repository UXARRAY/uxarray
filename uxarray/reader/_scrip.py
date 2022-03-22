# Create list of names that can be used for node coorin_ds
_X_NODES = ("lat", "latitude", "Lat", "Latitude", "X", "x", "grid_corner_lat",
            "grid_corner_latitude")

_Y_NODES = ("lon", "longitude", "Lon", "Longitude", "Y", "y", "grid_corner_lon",
            "grid_corner_longitude")


def _viewkeys(d):
    """Return either the keys or viewkeys method for a dictionary.
    Parameters
    ----------
    Args:
        d :obj:`dict`: A dictionary.
    Returns:
        view method: Either the keys or viewkeys method.
    """

    func = getattr(d, "viewkeys", None)
    if func is None:
        func = d.keys
    return func()


def _get_nodes(in_ds):
    """Outer wrapper function for datasets that are not cf compliant.

    in_ds : :class:`xarray.Dataset`
        Scrip dataset of interest being used

    return : :class:`xarray.Dataset`
        Dataset updated with cf compliant naming of lat/lon variables
    """

    node_x = None
    node_y = None

    for name in _X_NODES:
        if name in _viewkeys(in_ds):
            in_ds['Mesh2_node_x'] = in_ds[name]
            node_x = in_ds['Mesh2_node_x']

            break

    for name in _Y_NODES:
        if name in _viewkeys(in_ds):
            in_ds['Mesh2_node_y'] = in_ds[name]
            node_y = in_ds['Mesh2_node_y']

            break

    return node_x, node_y


def _is_scrip(in_ds, out_ds):
    """If in file is an unstructued SCRIP file, function will reassign SCRIP
    variables to UGRID conventions.

    :param in_ds: original SCRIP file
    :param out_ds:
        file to be returned by _populate_scrip_data, used as an empty placeholder file
        to store reassigned SCRIP variables in UGRID conventions

    :return: variables for out_ds that follow UGRID conventions
    """

    if in_ds['grid_area'].all(
    ):  # Presence indicates if SCRIP file is unstructured grid
        out_ds['Mesh2_node_x'] = in_ds['grid_corner_lon']
        out_ds['Mesh2_node_y'] = in_ds['grid_corner_lat']

        # Create array using matching grid corners to create face nodes
        face_arr = []
        for i in range(len(in_ds_ne8['grid_corner_lat'] - 1)):
            x = in_ds_ne8['grid_corner_lon'][i].values
            y = in_ds_ne8['grid_corner_lat'][i].values
            face = np.hstack([x[:, np.newaxis], y[:, np.newaxis]])
            face_arr.append(face)

        face_node = np.asarray(face_arr)

        out_ds['Mesh2_face_nodes'] = xr.DataArray(
            face_node, dims=['grid_size', 'grid_corners', 'lat/lon'])
    else:
        raise Exception("Structured scrip files are not yet supported")
        # print("Structured scrip files are not yet supported")


def _populate_scrip_data(in_ds):
    """Function to reassign lat/lon variables to mesh2_node variables. Could be
    possible to expand this to include:

    - Error message if variable name is not in current dictionary
    - Further capability to have a user input for variables not in dictionary

    Parameters
    ----------
    in_ds : :class:`xarray.Dataset`
        Scrip dataset of interest being used

    Returns
    --------
    out_ds : :class:`xarray.Variable`
        Reassigns data variables to be mesh2_node_x

    node_y : :class:`xarray.Variable`
        Reassigns data variables to be mesh2_node_y
    """
    out_ds = xr.Dataset()
    try:
        _is_scrip(in_ds, out_ds)

    except KeyError:
        if in_ds['Mesh2']:
            try:
                out_ds = in_ds
                return out_ds
            except:
                raise Exception(
                    "Variables not in form 'Mesh2_node_x' or 'Mesh2_node_y' please specify cf=False"
                )

    out_ds["Mesh2"] = xr.DataArray(
        attrs={
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 2D unstructured mesh",
            "topology_dimension": 2,
            "node_coordinates": "Mesh2_node_x Mesh2_node_y Mesh2_node_z",
            "node_dimension": "nMesh2_node",
            "face_node_connectivity": "Mesh2_face_nodes",
            "face_dimension": "nMesh2_face"
        })

    return out_ds


from uxarray.reader._scrip import _populate_scrip_data
import xarray as xr
import sys
import pytest
import numpy as np

import os
from pathlib import Path

if "--cov" in str(sys.argv):
    from uxarray.reader._scrip import _populate_scrip_data
else:
    import uxarray

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

ne30 = '/Users/misi1684/uxarray/test/meshfiles/outCSne30.ug'
ne8 = '/Users/misi1684/uxarray/test/meshfiles/outCSne8.nc'

in_ds_ne30 = xr.open_dataset(ne30, decode_times=False,
                             engine='netcdf4')  # mesh2_node_x/y
in_ds_ne8 = xr.open_dataset(ne8, decode_times=False,
                            engine='netcdf4')  # grid_corner_lat/lon
new = _populate_scrip_data(in_ds_ne30)
# print(in_ds_ne30['Mesh2_face_nodes'], "original")
print(new, "New")

# face_arr = []
# for i in range(len(in_ds_ne8['grid_corner_lat'] - 1)):
#     x = in_ds_ne8['grid_corner_lon'][i].values
#     y = in_ds_ne8['grid_corner_lat'][i].values
#     face = np.column_stack([x[:, np.newaxis], y[:, np.newaxis]])
#     x_vars = in_ds_ne8['grid_corner_lon']
#     # print(x_vars)
#     face_arr.append(face)
#
# as_arr = np.asarray(face_arr)

# ESMF files https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/share/meshes/
# ESMF/SCRIP info: https://earthsystemmodeling.org/docs/release/ESMF_6_2_0/ESMF_refdoc/node3.html
