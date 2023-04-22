import numpy as np
import xarray as xr

from uxarray.helpers import node_xyz_to_lonlat_rad, node_lonlat_rad_to_xyz


def _populate_cartesian_xyz_coord(grid):
    """A helper function that populates the xyz attribute in UXarray.ds. This
    function is called when we need to use the cartesian coordinates for each
    node to do the calculation but the input data only has the "Mesh2_node_x"
    and "Mesh2_node_y" in degree.

    Parameters
    ----------
    grid : ux.Grid
        Source Grid

    Note
    ----
    In the UXarray, we abide the UGRID convention and make sure the following attributes will always have its
    corresponding units as stated below:

    Mesh2_node_x
     unit:  "degree_east" for longitude
    Mesh2_node_y
     unit:  "degrees_north" for latitude
    Mesh2_node_z
     unit:  "m"
    Mesh2_node_cart_x
     unit:  "m"
    Mesh2_node_cart_y
     unit:  "m"
    Mesh2_node_cart_z
     unit:  "m"
    """

    # Check if the cartesian coordinates are already populated
    if "Mesh2_node_cart_x" in grid.ds.keys():
        return

    # check for units and create Mesh2_node_cart_x/y/z set to grid.ds
    nodes_lon_rad = np.deg2rad(grid.Mesh2_node_x.values)
    nodes_lat_rad = np.deg2rad(grid.Mesh2_node_y.values)
    nodes_rad = np.stack((nodes_lon_rad, nodes_lat_rad), axis=1)
    nodes_cart = np.asarray(list(map(node_lonlat_rad_to_xyz, list(nodes_rad))))

    grid.ds["Mesh2_node_cart_x"] = xr.DataArray(
        data=nodes_cart[:, 0],
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "cartesian x",
            "units": "m",
        })
    grid.ds["Mesh2_node_cart_y"] = xr.DataArray(
        data=nodes_cart[:, 1],
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "cartesian y",
            "units": "m",
        })
    grid.ds["Mesh2_node_cart_z"] = xr.DataArray(
        data=nodes_cart[:, 2],
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "cartesian z",
            "units": "m",
        })


def _populate_lonlat_coord(grid):
    """Helper function that populates the longitude and latitude and store it
    into the Mesh2_node_x and Mesh2_node_y. This is called when the input data
    has "Mesh2_node_x", "Mesh2_node_y", "Mesh2_node_z" in meters. Since we want
    "Mesh2_node_x" and "Mesh2_node_y" always have the "degree" units. For more
    details, please read the following.

    Parameters
    ----------
    grid : ux.Grid
        Source Grid

    Raises
    ------
        RuntimeError
            Mesh2_node_x/y/z are not represented in the cartesian format with the unit 'm'/'meters' when calling this function"

    Note
    ----
    In the UXarray, we abide the UGRID convention and make sure the following attributes will always have its
    corresponding units as stated below:

    Mesh2_node_x
     unit:  "degree_east" for longitude
    Mesh2_node_y
     unit:  "degrees_north" for latitude
    Mesh2_node_z
     unit:  "m"
    Mesh2_node_cart_x
     unit:  "m"
    Mesh2_node_cart_y
     unit:  "m"
    Mesh2_node_cart_z
     unit:  "m"
    """

    # Check if the "Mesh2_node_x" is already in longitude
    if "degree" in grid.ds.Mesh2_node_x.units:
        return

    # Check if the input Mesh2_node_xyz" are represented in the cartesian format with the unit "m"
    if ("m" not in grid.ds.Mesh2_node_x.units) or (
            "m" not in grid.ds.Mesh2_node_y.units) \
            or ("m" not in grid.ds.Mesh2_node_z.units):
        raise RuntimeError(
            "Expected: Mesh2_node_x/y/z should be represented in the cartesian format with the "
            "unit 'm' when calling this function")

    # Put the cartesian coordinates inside the proper data structure
    grid.ds["Mesh2_node_cart_x"] = xr.DataArray(
        data=grid.ds["Mesh2_node_x"].values)
    grid.ds["Mesh2_node_cart_y"] = xr.DataArray(
        data=grid.ds["Mesh2_node_y"].values)
    grid.ds["Mesh2_node_cart_z"] = xr.DataArray(
        data=grid.ds["Mesh2_node_z"].values)

    # convert the input cartesian values into the longitude latitude degree
    nodes_cart = np.stack(
        (grid.ds["Mesh2_node_x"].values, grid.ds["Mesh2_node_y"].values,
         grid.ds["Mesh2_node_z"].values),
        axis=1).tolist()
    nodes_rad = list(map(node_xyz_to_lonlat_rad, nodes_cart))
    nodes_degree = np.rad2deg(nodes_rad)
    grid.ds["Mesh2_node_x"] = xr.DataArray(
        data=nodes_degree[:, 0],
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "longitude",
            "long_name": "longitude of mesh nodes",
            "units": "degrees_east",
        })
    grid.ds["Mesh2_node_y"] = xr.DataArray(
        data=nodes_degree[:, 1],
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "lattitude",
            "long_name": "latitude of mesh nodes",
            "units": "degrees_north",
        })
