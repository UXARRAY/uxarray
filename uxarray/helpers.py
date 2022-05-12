import math
import numpy as np
import xarray as xr
from pathlib import PurePath

from .get_quadratureDG import get_gauss_quadratureDG


# helper function to find file type
def determine_file_type(filepath):
    """Checks file path and contents to determine file type. Supports detection
    of UGrid, SCRIP, Exodus and shape file.

    Parameters: string, required
       Filepath of the file for which the filetype is to be determined.

    Returns: string
       File type: ug, exo, scrip or shp

    Raises:
       RuntimeError: Invalid file type
    """
    msg = ""
    mesh_filetype = "unknown"
    # exodus with coord
    try:
        # extract the file name and extension
        path = PurePath(filepath)
        file_extension = path.suffix

        # try to open file with xarray and test for exodus
        ext_ds = xr.open_dataset(filepath, mask_and_scale=False)["coord"]
        mesh_filetype = "exo"
    except KeyError as e:
        # exodus with coordx
        try:
            ext_ds = xr.open_dataset(filepath, mask_and_scale=False)["coordx"]
            mesh_filetype = "exo"
        except KeyError as e:
            # scrip with grid_center_lon
            try:
                ext_ds = xr.open_dataset(
                    filepath, mask_and_scale=False)["grid_center_lon"]
                mesh_filetype = "scrip"
            except KeyError as e:

                # check mesh topology and dimension
                try:
                    standard_name = lambda v: v is not None
                    # getkeys_filter_by_attribute(filepath, attr_name, attr_val)
                    # return type KeysView
                    base_dv_nc = list(
                        xr.open_dataset(
                            filepath, mask_and_scale=False).filter_by_attrs(
                                node_coordinates=standard_name).keys())[0]
                    base_dv_fc = list(
                        xr.open_dataset(
                            filepath, mask_and_scale=False).filter_by_attrs(
                                face_node_connectivity=standard_name).keys())[0]
                    base_dv_td = list(
                        xr.open_dataset(
                            filepath, mask_and_scale=False).filter_by_attrs(
                                topology_dimension=standard_name).keys())[0]
                    base_dv_mt = list(
                        xr.open_dataset(filepath,
                                        mask_and_scale=False).filter_by_attrs(
                                            cf_role="mesh_topology").keys())[0]
                    if base_dv_mt != "" and base_dv_td != "" and base_dv_fc != "" and base_dv_nc != "":
                        mesh_filetype = "ugrid"
                    else:
                        print(
                            "cf_role is other than mesh_topology, the input NetCDF file is not UGRID format"
                        )
                        exit()
                except KeyError as e:
                    msg = str(e) + ': {}'.format(filepath)
    except (TypeError, AttributeError) as e:
        msg = str(e) + ': {}'.format(filepath)
    except (RuntimeError, OSError) as e:
        # check if this is a shp file
        # we won't use xarray to load that file
        if file_extension == ".shp":
            mesh_filetype = "shp"
        else:
            msg = str(e) + ': {}'.format(filepath)
    except ValueError as e:
        # check if this is a shp file
        # we won't use xarray to load that file
        if file_extension == ".shp":
            mesh_filetype = "shp"
        else:
            msg = str(e) + ': {}'.format(filepath)
    finally:
        if msg != "":
            msg = "Unable to determine file type, mesh file not supported" + ': {}'.format(
                filepath)
            raise ValueError(msg)

    return mesh_filetype


def spherical_to_cartesian_unit(lat, lon, r=6371):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = r * math.cos(lat) * math.cos(lon)  # x coordinate
    y = r * math.cos(lat) * math.sin(lon)  # y coordinate
    z = r * math.sin(lat)  # z coordinate

    coord = np.array([x, y, z])
    # make it coord on a sphere with unit radius
    unit_coord = coord / np.linalg.norm(coord)

    return unit_coord


def spherical_to_cartesian_unit(node, r=6371):
    """Converts spherical (lat/lon) coordinates to cartesian (x,y,z).

    Final output is cartesian coordinates on a sphere of unit radius

    Parameters:
    -----------

    node: a list consisting of lat and lon

    Returns: numpy array
        Cartesian coordinates of length 3
    """
    lat = node[0]
    lon = node[1]
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = r * math.cos(lat) * math.cos(lon)  # x coordinate
    y = r * math.cos(lat) * math.sin(lon)  # y coordinate
    z = r * math.sin(lat)  # z coordinate

    coord = np.array([x, y, z])
    # make it coord on a sphere with unit radius
    unit_coord = coord / np.linalg.norm(coord)

    return unit_coord


# Calculate the area of all faces.
def calculate_face_area(x, y, z, type="spherical"):
    """Calculate area of a face on sphere.

    Parameters
    ----------

    x : list, required
        x-coordinate of all the nodes forming the face

    y : list, required
        y-coordinate of all the nodes forming the face

    z : list, required
        z-coordinate of all the nodes forming the face

    type : str, optional
        coordinate type, default is spherical, can be cartesian also.
    """
    area = 0  # set area to 0
    order = 6
    dG, dW = get_gauss_quadratureDG(order)

    num_nodes = len(x)

    # num triangles is two less than the total number of nodes
    num_triangles = num_nodes - 2
    # Using tempestremap GridElements: https://github.com/ClimateGlobalChange/tempestremap/blob/master/src/GridElements.cpp
    # loop thru all triangles
    for j in range(0, num_triangles):
        node1 = [x[0], y[0], z[0]]
        node2 = [x[j + 1], y[j + 1], z[j + 1]]
        node3 = [x[j + 2], y[j + 2], z[j + 2]]
        if (type == "spherical"):
            node1 = spherical_to_cartesian_unit(node1)
            node2 = spherical_to_cartesian_unit(node2)
            node3 = spherical_to_cartesian_unit(node3)
        for p in range(len(dW)):
            for q in range(len(dW)):
                dA = dG[p]
                dB = dG[q]
                jacobian = calculate_spherical_triangle_jacobian(
                    node1, node2, node3, dA, dB)
                area += dW[p] * dW[q] * jacobian
    return area


def calculate_spherical_triangle_jacobian(node1, node2, node3, dA, dB):
    """Helper function for calculating face area."""
    dF = [(1.0 - dB) * ((1.0 - dA) * node1[0] + dA * node2[0]) + dB * node3[0],
          (1.0 - dB) * ((1.0 - dA) * node1[1] + dA * node2[1]) + dB * node3[1],
          (1.0 - dB) * ((1.0 - dA) * node1[2] + dA * node2[2]) + dB * node3[2]]

    dDaF = [(1.0 - dB) * (node2[0] - node1[0]),
            (1.0 - dB) * (node2[1] - node1[1]),
            (1.0 - dB) * (node2[2] - node1[2])]

    dDbF = [
        -(1.0 - dA) * node1[0] - dA * node2[0] + node3[0],
        -(1.0 - dA) * node1[1] - dA * node2[1] + node3[1],
        -(1.0 - dA) * node1[2] - dA * node2[2] + node3[2]
    ]

    dInvR = 1.0 / math.sqrt(dF[0] * dF[0] + dF[1] * dF[1] + dF[2] * dF[2])

    dDaG = [
        dDaF[0] * (dF[1] * dF[1] + dF[2] * dF[2]) - dF[0] *
        (dDaF[1] * dF[1] + dDaF[2] * dF[2]),
        dDaF[1] * (dF[0] * dF[0] + dF[2] * dF[2]) - dF[1] *
        (dDaF[0] * dF[0] + dDaF[2] * dF[2]),
        dDaF[2] * (dF[0] * dF[0] + dF[1] * dF[1]) - dF[2] *
        (dDaF[0] * dF[0] + dDaF[1] * dF[1])
    ]

    dDbG = [
        dDbF[0] * (dF[1] * dF[1] + dF[2] * dF[2]) - dF[0] *
        (dDbF[1] * dF[1] + dDbF[2] * dF[2]),
        dDbF[1] * (dF[0] * dF[0] + dF[2] * dF[2]) - dF[1] *
        (dDbF[0] * dF[0] + dDbF[2] * dF[2]),
        dDbF[2] * (dF[0] * dF[0] + dF[1] * dF[1]) - dF[2] *
        (dDbF[0] * dF[0] + dDbF[1] * dF[1])
    ]

    dDenomTerm = dInvR * dInvR * dInvR

    dDaG[0] *= dDenomTerm
    dDaG[1] *= dDenomTerm
    dDaG[2] *= dDenomTerm

    dDbG[0] *= dDenomTerm
    dDbG[1] *= dDenomTerm
    dDbG[2] *= dDenomTerm

    #  Cross product gives local Jacobian
    nodeCross = cross(dDaG, dDbG)
    # print("nc", nodeCross)
    dJacobian = math.sqrt(nodeCross[0] * nodeCross[0] +
                          nodeCross[1] * nodeCross[1] +
                          nodeCross[2] * nodeCross[2])

    return dJacobian


def cross(a, b):
    """Cross product of 3D vector a and b."""
    c = [0] * 3
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c


def dot(a, b):
    """Dot product of 3D vector a and b."""
    c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return c
