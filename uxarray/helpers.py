import math

import numpy as np
import xarray as xr
from pathlib import PurePath
import numpy as np
import copy

from .get_quadratureDG import get_gauss_quadratureDG, get_tri_quadratureDG
from numba import njit, config
from .utilities import normalize_in_place

config.DISABLE_JIT = True


def parse_grid_type(filepath, **kw):
    """Checks input and contents to determine grid type. Supports detection of
    UGrid, SCRIP, Exodus and shape file.

    Parameters
    ----------
    filepath : str
       Filepath of the file for which the filetype is to be determined.

    Returns
    -------
    mesh_filetype : str
        File type of the file, ug, exo, scrip or shp

    Raises
    ------
    RuntimeError
            If invalid file type
    ValueError
        If file is not in UGRID format
    """
    # extract the file name and extension
    path = PurePath(filepath)
    file_extension = path.suffix
    # short-circuit for shapefiles
    if file_extension == ".shp":
        mesh_filetype, dataset = "shp", None
        return mesh_filetype, dataset

    dataset = xr.open_dataset(filepath, mask_and_scale=False, **kw)
    # exodus with coord or coordx
    if "coord" in dataset:
        mesh_filetype = "exo"
    elif "coordx" in dataset:
        mesh_filetype = "exo"
    # scrip with grid_center_lon
    elif "grid_center_lon" in dataset:
        mesh_filetype = "scrip"
    # ugrid topology
    elif _is_ugrid(dataset):
        mesh_filetype = "ugrid"
    else:
        raise RuntimeError(f"Could not recognize {filepath} format.")
    return mesh_filetype, dataset

    # check mesh topology and dimension
    try:
        standard_name = lambda v: v is not None
        # getkeys_filter_by_attribute(filepath, attr_name, attr_val)
        # return type KeysView
        ext_ds = xr.open_dataset(filepath, mask_and_scale=False)
        node_coords_dv = ext_ds.filter_by_attrs(
            node_coordinates=standard_name).keys()
        face_conn_dv = ext_ds.filter_by_attrs(
            face_node_connectivity=standard_name).keys()
        topo_dim_dv = ext_ds.filter_by_attrs(
            topology_dimension=standard_name).keys()
        mesh_topo_dv = ext_ds.filter_by_attrs(cf_role="mesh_topology").keys()
        if list(mesh_topo_dv)[0] != "" and list(topo_dim_dv)[0] != "" and list(
                face_conn_dv)[0] != "" and list(node_coords_dv)[0] != "":
            mesh_filetype = "ugrid"
        else:
            raise ValueError(
                "cf_role is other than mesh_topology, the input NetCDF file is not UGRID format"
            )
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


@njit
def _spherical_to_cartesian_unit_(node, r=6371):
    """Converts spherical (lat/lon) coordinates to cartesian (x,y,z).

    Final output is cartesian coordinates on a sphere of unit radius

    Parameters:
    -----------
    node: a list consisting of lat and lon

    Returns: numpy array
        Cartesian coordinates of length 3
    """
    lon = node[0]
    lat = node[1]
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = r * np.cos(lat) * np.cos(lon)  # x coordinate
    y = r * np.cos(lat) * np.sin(lon)  # y coordinate
    z = r * np.sin(lat)  # z coordinate

    coord = np.array([x, y, z])
    # make it coord on a sphere with unit radius
    unit_coord = coord / np.linalg.norm(coord)

    return unit_coord


# Calculate the area of all faces.
@njit
def calculate_face_area(x,
                        y,
                        z,
                        quadrature_rule="gaussian",
                        order=4,
                        coords_type="spherical"):
    """Calculate area of a face on sphere.

    Parameters
    ----------
    x : list, required
        x-coordinate of all the nodes forming the face

    y : list, required
        y-coordinate of all the nodes forming the face

    z : list, required
        z-coordinate of all the nodes forming the face

    quadrature_rule : str, optional
        triangular and Gaussian quadrature supported, expected values: "triangular" or "gaussian"

    order: int, optional
        Order of the quadrature rule. Default is 4.

        Supported values:
            - Gaussian Quadrature: 1 to 10
            - Triangular: 1, 4, 8, 10 and 12

    coords_type : str, optional
        coordinate type, default is spherical, can be cartesian also.
    """
    area = 0.0  # set area to 0
    order = order

    if quadrature_rule == "gaussian":
        dG, dW = get_gauss_quadratureDG(order)
    elif quadrature_rule == "triangular":
        dG, dW = get_tri_quadratureDG(order)
    else:
        raise ValueError(
            "Invalid quadrature rule, specify gaussian or triangular")

    num_nodes = len(x)

    # num triangles is two less than the total number of nodes
    num_triangles = num_nodes - 2

    # Using tempestremap GridElements: https://github.com/ClimateGlobalChange/tempestremap/blob/master/src/GridElements.cpp
    # loop through all sub-triangles of face
    for j in range(0, num_triangles):
        node1 = np.array([x[0], y[0], z[0]], dtype=np.float64)
        node2 = np.array([x[j + 1], y[j + 1], z[j + 1]], dtype=np.float64)
        node3 = np.array([x[j + 2], y[j + 2], z[j + 2]], dtype=np.float64)
        if (coords_type == "spherical"):
            node1 = _spherical_to_cartesian_unit_(node1)
            node2 = _spherical_to_cartesian_unit_(node2)
            node3 = _spherical_to_cartesian_unit_(node3)
        for p in range(len(dW)):
            if quadrature_rule == "gaussian":
                for q in range(len(dW)):
                    dA = dG[0][p]
                    dB = dG[0][q]
                    jacobian = calculate_spherical_triangle_jacobian(
                        node1, node2, node3, dA, dB)
                    area += dW[p] * dW[q] * jacobian
            elif quadrature_rule == "triangular":
                dA = dG[p][0]
                dB = dG[p][1]
                jacobian = calculate_spherical_triangle_jacobian_barycentric(
                    node1, node2, node3, dA, dB)
                area += dW[p] * jacobian

    return area


@njit
def get_all_face_area_from_coords(x,
                                  y,
                                  z,
                                  face_nodes,
                                  dim,
                                  quadrature_rule="triangular",
                                  order=4,
                                  coords_type="spherical"):
    """Given coords, connectivity and other area calculation params, this
    routine loop over all faces and return an numpy array with areas of each
    face.

    Parameters
    ----------
    x : ndarray, required
        x-coordinate of all the nodes

    y : ndarray, required
        y-coordinate of all the nodes

    z : ndarray, required
        z-coordinate of all the nodes

    face_nodes : 2D ndarray, required
         node ids of each face

    dim : int, required
         dimension

    quadrature_rule : str, optional
        "triangular" or "gaussian". Defaults to triangular

    order : int, optional
        count or order for Gaussian or spherical resp. Defaults to 4 for spherical.

    coords_type : str, optional
        coordinate type, default is spherical, can be cartesian also.

    Returns
    -------
    area of all faces : ndarray
    """
    num_faces = face_nodes.shape[0]
    area = np.zeros(num_faces)  # set area of each face to 0

    for i in range(num_faces):

        face_z = np.zeros(len(face_nodes[i]))

        face_x = x[face_nodes[i]]
        face_y = y[face_nodes[i]]
        # check if z dimension
        if dim > 2:
            face_z = z[face_nodes[i]]

        # After getting all the nodes of a face assembled call the  cal. face area routine
        face_area = calculate_face_area(face_x, face_y, face_z, quadrature_rule,
                                        order, coords_type)

        area[i] = face_area

    return area


@njit
def calculate_spherical_triangle_jacobian(node1, node2, node3, dA, dB):
    """Calculate Jacobian of a spherical triangle. This is a helper function
    for calculating face area.

    Parameters
    ----------
    node1 : list, required
        First node of the triangle

    node2 : list, required
        Second node of the triangle

    node3 : list, required
        Third node of the triangle

    dA : float, required
        quadrature point

    dB : float, required
        quadrature point

    Returns
    -------
    jacobian : float
    """
    dF = np.array([
        (1.0 - dB) * ((1.0 - dA) * node1[0] + dA * node2[0]) + dB * node3[0],
        (1.0 - dB) * ((1.0 - dA) * node1[1] + dA * node2[1]) + dB * node3[1],
        (1.0 - dB) * ((1.0 - dA) * node1[2] + dA * node2[2]) + dB * node3[2]
    ])

    dDaF = np.array([(1.0 - dB) * (node2[0] - node1[0]),
                     (1.0 - dB) * (node2[1] - node1[1]),
                     (1.0 - dB) * (node2[2] - node1[2])])

    dDbF = np.array([
        -(1.0 - dA) * node1[0] - dA * node2[0] + node3[0],
        -(1.0 - dA) * node1[1] - dA * node2[1] + node3[1],
        -(1.0 - dA) * node1[2] - dA * node2[2] + node3[2]
    ])

    dInvR = 1.0 / np.sqrt(dF[0] * dF[0] + dF[1] * dF[1] + dF[2] * dF[2])

    dDaG = np.array([
        dDaF[0] * (dF[1] * dF[1] + dF[2] * dF[2]) - dF[0] *
        (dDaF[1] * dF[1] + dDaF[2] * dF[2]),
        dDaF[1] * (dF[0] * dF[0] + dF[2] * dF[2]) - dF[1] *
        (dDaF[0] * dF[0] + dDaF[2] * dF[2]),
        dDaF[2] * (dF[0] * dF[0] + dF[1] * dF[1]) - dF[2] *
        (dDaF[0] * dF[0] + dDaF[1] * dF[1])
    ])

    dDbG = np.array([
        dDbF[0] * (dF[1] * dF[1] + dF[2] * dF[2]) - dF[0] *
        (dDbF[1] * dF[1] + dDbF[2] * dF[2]),
        dDbF[1] * (dF[0] * dF[0] + dF[2] * dF[2]) - dF[1] *
        (dDbF[0] * dF[0] + dDbF[2] * dF[2]),
        dDbF[2] * (dF[0] * dF[0] + dF[1] * dF[1]) - dF[2] *
        (dDbF[0] * dF[0] + dDbF[1] * dF[1])
    ])

    dDenomTerm = dInvR * dInvR * dInvR

    dDaG *= dDenomTerm
    dDbG *= dDenomTerm

    #  Cross product gives local Jacobian
    nodeCross = np.cross(dDaG, dDbG)
    dJacobian = np.sqrt(nodeCross[0] * nodeCross[0] +
                        nodeCross[1] * nodeCross[1] +
                        nodeCross[2] * nodeCross[2])

    return dJacobian


@njit
def calculate_spherical_triangle_jacobian_barycentric(node1, node2, node3, dA,
                                                      dB):
    """Calculate Jacobian of a spherical triangle. This is a helper function
    for calculating face area.

    Parameters
    ----------
    node1 : list, required
        First node of the triangle

    node2 : list, required
        Second node of the triangle

    node3 : list, required
        Third node of the triangle

    dA : float, required
        first component of barycentric coordinates of quadrature point

    dB : float, required
        second component of barycentric coordinates of quadrature point

    Returns
    -------
    jacobian : float
    """

    dF = np.array([
        dA * node1[0] + dB * node2[0] + (1.0 - dA - dB) * node3[0],
        dA * node1[1] + dB * node2[1] + (1.0 - dA - dB) * node3[1],
        dA * node1[2] + dB * node2[2] + (1.0 - dA - dB) * node3[2]
    ])

    dDaF = np.array(
        [node1[0] - node3[0], node1[1] - node3[1], node1[2] - node3[2]])

    dDbF = np.array(
        [node2[0] - node3[0], node2[1] - node3[1], node2[2] - node3[2]])

    dInvR = 1.0 / np.sqrt(dF[0] * dF[0] + dF[1] * dF[1] + dF[2] * dF[2])

    dDaG = np.array([
        dDaF[0] * (dF[1] * dF[1] + dF[2] * dF[2]) - dF[0] *
        (dDaF[1] * dF[1] + dDaF[2] * dF[2]),
        dDaF[1] * (dF[0] * dF[0] + dF[2] * dF[2]) - dF[1] *
        (dDaF[0] * dF[0] + dDaF[2] * dF[2]),
        dDaF[2] * (dF[0] * dF[0] + dF[1] * dF[1]) - dF[2] *
        (dDaF[0] * dF[0] + dDaF[1] * dF[1])
    ])

    dDbG = np.array([
        dDbF[0] * (dF[1] * dF[1] + dF[2] * dF[2]) - dF[0] *
        (dDbF[1] * dF[1] + dDbF[2] * dF[2]),
        dDbF[1] * (dF[0] * dF[0] + dF[2] * dF[2]) - dF[1] *
        (dDbF[0] * dF[0] + dDbF[2] * dF[2]),
        dDbF[2] * (dF[0] * dF[0] + dF[1] * dF[1]) - dF[2] *
        (dDbF[0] * dF[0] + dDbF[1] * dF[1])
    ])

    dDenomTerm = dInvR * dInvR * dInvR

    dDaG *= dDenomTerm
    dDbG *= dDenomTerm

    #  Cross product gives local Jacobian
    nodeCross = np.cross(dDaG, dDbG)
    dJacobian = np.sqrt(nodeCross[0] * nodeCross[0] +
                        nodeCross[1] * nodeCross[1] +
                        nodeCross[2] * nodeCross[2])

    return 0.5 * dJacobian


def _is_ugrid(ds):
    """Check mesh topology and dimension."""
    standard_name = lambda v: v is not None
    # getkeys_filter_by_attribute(filepath, attr_name, attr_val)
    # return type KeysView
    node_coords_dv = ds.filter_by_attrs(node_coordinates=standard_name)
    face_conn_dv = ds.filter_by_attrs(face_node_connectivity=standard_name)
    topo_dim_dv = ds.filter_by_attrs(topology_dimension=standard_name)
    mesh_topo_dv = ds.filter_by_attrs(cf_role="mesh_topology")
    if len(mesh_topo_dv) != 0 and len(topo_dim_dv) != 0 and len(
            face_conn_dv) != 0 and len(node_coords_dv) != 0:
        return True
    else:
        return False


# helper function to insert a new point into the latlon box
def insert_pt_in_latlonbox(old_box, new_pt, is_lon_periodic=True):
    """Compare the new point's latitude and longitude with the target the
    latlonbox.

    Parameters: old_box: float array, the original lat lon box [[lat_0, lat_1],[lon_0, lon_1]],required
                new_pt: float array, the new lat lon point [lat, lon], required
                is_lon_periodic: Flag indicating the latlonbox is a regional (default to be True).

    Returns: float array, a lat lon box [[lat_0, lat_1],[lon_0, lon_1]]

    Raises:
       Exception: Logic Errors
    """
    old_lon_width = 2.0 * np.pi
    lat_pt = new_pt[0]
    lon_pt = new_pt[1]
    latlon_box = old_box  # The returned point

    if lon_pt < 0.0:
        raise Exception('lon_pt out of range ( {} < 0)"'.format(lon_pt))

    if lon_pt > old_lon_width:
        raise Exception('lon_pt out of range ( {} > {})"'.format(
            lon_pt, old_lon_width))

    # Expand latitudes
    if lat_pt > latlon_box[0][1]:
        latlon_box[0][1] = lat_pt

    if lat_pt < latlon_box[0][0]:
        latlon_box[0][0] = lat_pt

    # Expand longitude, if non-periodic
    if not is_lon_periodic:
        if lon_pt > latlon_box[1][1]:
            latlon_box[1][1] = lon_pt
        if lon_pt < latlon_box[1][0]:
            latlon_box[1][0] = lon_pt
        return

    # New longitude lies within existing range
    if latlon_box[1][0] <= latlon_box[1][1]:
        if latlon_box[1][0] <= lon_pt <= latlon_box[1][1]:
            return
    else:
        if lon_pt >= latlon_box[1][0] or lon_pt <= latlon_box[1][0]:
            return

    # New longitude lies outside existing range
    box_a = latlon_box
    box_a[1][0] = lon_pt

    box_b = latlon_box
    box_b[1][1] = lon_pt

    # The updated box is the box of minimum width
    d_width_now = get_latlonbox_width(latlon_box)
    d_width_a = get_latlonbox_width(box_a)
    d_width_b = get_latlonbox_width(box_b)

    if (d_width_a - d_width_now) < -1.0e-14 or (d_width_b -
                                                d_width_now) < -1.0e-14:
        raise Exception('logic error')

    if d_width_a < d_width_b:
        return box_a
    else:
        return box_b


# helper function to calculate the latlonbox width
def get_latlonbox_width(latlonbox, is_lon_periodic):
    """Calculate the width of this LatLonBox
    Parameters: latlonbox: float array, lat lon box [[lat_0, lat_1],[lon_0, lon_1]],required
                is_lon_periodic: boolean, Flag indicating the latlonbox is a regional (default to be True).

    Returns: float array, a lat lon box [[lat_0, lat_1],[lon_0, lon_1]]

    Raises:
       Exception: Logic Errors
    """

    if not is_lon_periodic:
        return latlonbox[1][1] - latlonbox[1][0]

    if latlonbox[1][0] == latlonbox[1][1]:
        return 0.0
    elif latlonbox[1][0] <= latlonbox[1][1]:
        return latlonbox[1][1] - latlonbox[1][0]
    else:
        latlonbox[1][1] - latlonbox[1][0] + (2 * np.pi)


# Convert the node coordinate from 2D longitude/latitude to normalized 3D xyz
def convert_node_lonlat_rad_to_xyz(node_coord):
    """
    Parameters: float list, required
       the input 2D coordinates[longitude, latitude]
    Returns: float list, the 3D coordinates in [X, Y, Z]
    """
    lon = node_coord[0]
    lat = node_coord[1]
    return [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)]


# helper function to calculate latitude and longitude from a node's normalized 3D Cartesian
# coordinates, in radians.
def convert_node_xyz_to_lonlat_rad(node_coord):
    """Calculate the latitude and longitude in radiance for a node represented in the [x, y, z] 3D Cartesian coordinates.
    Parameters: node_coord: float array, [x, y, z],required

    Returns: float array, [latitude_rad, longitude_rad]

    Raises:
       Exception: Logic Errors
    """
    reference_tolerance = 1.0e-12
    [dx, dy, dz] = normalize_in_place(node_coord)

    d_mag_2 = dx * dx + dy * dy + dz * dz

    d_mag = np.absolute(d_mag_2)
    dx /= d_mag
    dy /= d_mag
    dz /= d_mag

    d_lon_rad = 0.0
    d_lat_rad = 0.0

    if np.absolute(dz) < (1.0 - reference_tolerance):
        d_lon_rad = math.atan2(dy, dx)
        d_lat_rad = np.arcsin(dz)

        if d_lon_rad < 0.0:
            d_lon_rad += 2.0 * np.pi
    elif dz > 0.0:
        d_lon_rad = 0.0
        d_lat_rad = 0.5 * np.pi
    else:
        d_lon_rad = 0.0
        d_lat_rad = -0.5 * np.pi

    return [d_lon_rad, d_lat_rad]


# helper function to insert a new point into the latlon box
def insert_pt_in_latlonbox(old_box, new_pt, is_lon_periodic=True):
    """Compare the new point's latitude and longitude with the target the
    latlonbox.

    Parameters: old_box: float array, the original lat lon box [[lat_0, lat_1],[lon_0, lon_1]],required
                new_pt: float array, the new lat lon point [lon, lat], required
                is_lon_periodic: Flag indicating the latlonbox is a regional (default to be True).

    Returns: float array, a lat lon box [[lat_0, lat_1],[lon_0, lon_1]]

    Raises:
       Exception: Logic Errors
    """
    # If the box is null (no point inserted yet)

    if old_box[0][0] == old_box[0][1] == 404.0:
        latlon_box = old_box
        latlon_box[0] = [new_pt[0], new_pt[0]]

    if old_box[1][0] == old_box[1][1] == 404.0:
        latlon_box = old_box
        latlon_box[1] = [new_pt[1], new_pt[1]]

    if old_box[0][0] == old_box[0][1] == old_box[1][0] == old_box[1][1] ==  404.0:
        return latlon_box

    # Deal with the pole point
    if new_pt[1] == 404.0 and (
            (np.absolute(new_pt[0] - 0.5 * np.pi) < 1.0e-12) or (np.absolute(new_pt[0] - (-0.5 * np.pi)) < 1.0e-12)):
        latlon_box = old_box
        if np.absolute(new_pt[0] - 0.5 * np.pi) < 1.0e-12:
            latlon_box[0][1] = 0.5 * np.pi
        elif np.absolute(new_pt[0] - (-0.5 * np.pi)) < 1.0e-12:
            latlon_box[0][0] = -0.5 * np.pi
        return latlon_box


    old_lon_width = 2.0 * np.pi
    lat_pt = new_pt[0]
    lon_pt = new_pt[1]
    latlon_box = old_box  # The returned box

    if lon_pt < 0.0:
        raise Exception('lon_pt out of range ( {} < 0)"'.format(lon_pt))

    if lon_pt > old_lon_width:
        raise Exception('lon_pt out of range ( {} > {})"'.format(
            lon_pt, old_lon_width))

    # Expand latitudes
    if lat_pt > latlon_box[0][1]:
        latlon_box[0][1] = lat_pt

    if lat_pt < latlon_box[0][0]:
        latlon_box[0][0] = lat_pt

    # Expand longitude, if non-periodic
    if not is_lon_periodic:
        if lon_pt > latlon_box[1][1]:
            latlon_box[1][1] = lon_pt
        if lon_pt < latlon_box[1][0]:
            latlon_box[1][0] = lon_pt
        return latlon_box

    # New longitude lies within existing range
    if latlon_box[1][0] <= latlon_box[1][1]:
        if lon_pt >= latlon_box[1][0] and lon_pt <= latlon_box[1][1]:
            return latlon_box
    else:
        if lon_pt >= latlon_box[1][0] or lon_pt <= latlon_box[1][1]:
            return latlon_box

    # New longitude lies outside of existing range
    box_a = copy.deepcopy(latlon_box)
    box_a[1][0] = lon_pt

    box_b = copy.deepcopy(latlon_box)
    box_b[1][1] = lon_pt

    # The updated box is the box of minimum width
    d_width_now = get_latlonbox_width(latlon_box)
    d_width_a = get_latlonbox_width(box_a)
    d_width_b = get_latlonbox_width(box_b)

    if (d_width_a - d_width_now) < -1.0e-14 or (d_width_b -
                                                d_width_now) < -1.0e-14:
        raise Exception('logic error')

    if d_width_a < d_width_b:
        return box_a
    else:
        return box_b


# helper function to calculate the latlonbox width
def get_latlonbox_width(latlonbox, is_lon_periodic=True):
    """Calculate the width of this LatLonBox
    Parameters: latlonbox: float array, lat lon box [[lat_0, lat_1],[lon_0, lon_1]],required
                is_lon_periodic: boolean, Flag indicating the latlonbox is a regional (default to be True).

    Returns: float array, a lat lon box [[lat_0, lat_1],[lon_0, lon_1]]

    Raises:
       Exception: Logic Errors
    """

    if not is_lon_periodic:
        return latlonbox[1][1] - latlonbox[1][0]

    if latlonbox[1][0] == latlonbox[1][1]:
        return 0.0
    elif latlonbox[1][0] <= latlonbox[1][1]:
        return latlonbox[1][1] - latlonbox[1][0]
    else:
        return latlonbox[1][1] - latlonbox[1][0] + (2 * np.pi)


import plotly.graph_objs as go


def vector_plot(tvects, is_vect=True, orig=[0, 0, 0]):
    """Plot vectors using plotly."""

    if is_vect:
        if not hasattr(orig[0], "__iter__"):
            coords = [[orig, np.sum([orig, v], axis=0)] for v in tvects]
        else:
            coords = [[o, np.sum([o, v], axis=0)] for o, v in zip(orig, tvects)]
    else:
        coords = tvects

    data = []
    for i, c in enumerate(coords):
        X1, Y1, Z1 = zip(c[0])
        X2, Y2, Z2 = zip(c[1])
        vector = go.Scatter3d(x=[X1[0], X2[0]],
                              y=[Y1[0], Y2[0]],
                              z=[Z1[0], Z2[0]],
                              marker=dict(size=[0, 5],
                                          color=['blue'],
                                          line=dict(width=5,
                                                    color='DarkSlateGrey')),
                              name='Vector' + str(i + 1))
        data.append(vector)

    layout = go.Layout(margin=dict(l=4, r=4, b=4, t=4))
    fig = go.Figure(data=data, layout=layout)
    fig.show()


# helper function to calculate the angle of 3D vectors u,v in radian
def angle_of_2_vectors(u, v):
    # 𝜃=2 𝑎𝑡𝑎𝑛2(|| ||𝑣||𝑢−||𝑢||𝑣 ||, || ||𝑣||𝑢+||𝑢||𝑣 ||)
    # this formula comes from W. Kahan's advice in his paper "How Futile are Mindless Assessments of Roundoff in
    # Floating-Point Computation?" (https://www.cs.berkeley.edu/~wkahan/Mindless.pdf), section 12 "Mangled Angles."
    v_norm_times_u = [np.linalg.norm(v) * u[i] for i in range(0, len(u))]
    u_norm_times_v = [np.linalg.norm(u) * v[i] for i in range(0, len(v))]
    vec_minus = [
        v_norm_times_u[i] - u_norm_times_v[i]
        for i in range(0, len(u_norm_times_v))
    ]
    vec_sum = [
        v_norm_times_u[i] + u_norm_times_v[i]
        for i in range(0, len(u_norm_times_v))
    ]
    angle_u_v_rad = 2 * math.atan2(np.linalg.norm(vec_minus),
                                   np.linalg.norm(vec_sum))
    return angle_u_v_rad


# Quantitative method to find the maximum latitude between in a great circle arc
def max_latitude_rad(v1, v2):
    """Quantitative method to find the maximum latitude between in a great circle arc
    Parameters:
        v1: float array [lon, lat] in degree east
        v2: float array [lon, lat] in degree east
    Returns: float, maximum latitude in radian
    """

    # Find the parametrized equation for the great circle passing through v1 and v2
    err_tolerance = 1.0e-15
    b_lonlat = np.deg2rad(v1)
    c_lonlat = np.deg2rad(v2)

    v1_cart = convert_node_lonlat_rad_to_xyz(np.deg2rad(v1))
    v2_cart = convert_node_lonlat_rad_to_xyz(np.deg2rad(v2))
    v_temp = np.cross(v1_cart, v2_cart)
    v0 = np.cross(v_temp, v1_cart)
    v0 = normalize_in_place(v0)

    max_section = [v1_cart,
                   v2_cart]  # record the subsection that has the maximum latitude

    # Only stop the iteration when two endpoints are extremely closed
    while np.absolute(b_lonlat[1] - c_lonlat[1]) >= err_tolerance or np.absolute(
            b_lonlat[0] - c_lonlat[0]) >= err_tolerance:
        max_lat = -np.pi  # reset the max_latitude for each while loop
        v_b = max_section[0]
        v_c = max_section[1]

        # Divide the angle of v1/v2 into 10 subsections, the leftover will be put in the last one
        # Update v0 based on max_section[0], since the angle is always from max_section[0] to v0
        angle_v1_v2_rad = angle_of_2_vectors(v_b, v_c)
        v0 = np.cross(v_temp, v_b)
        v0 = normalize_in_place(v0)
        avg_angle_rad = angle_v1_v2_rad / 10

        for i in range(0, 10):
            angle_rad_prev = avg_angle_rad * i
            if i >= 9:
                angle_rad_next = angle_v1_v2_rad
            else:
                angle_rad_next = angle_rad_prev + avg_angle_rad

            # Get the two vectors of this section
            w1_new = [np.cos(angle_rad_prev) * v_b[i] + np.sin(
                angle_rad_prev) * v0[i] for i in range(0, len(v_b))]
            w2_new = [np.cos(angle_rad_next) * v_b[i] + np.sin(
                angle_rad_next) * v0[i] for i in range(0, len(v_b))]

            # convert the 3D [x, y, z] vector into 2D lat/lon vector
            w1_lonlat = convert_node_xyz_to_lonlat_rad(w1_new)
            w2_lonlat = convert_node_xyz_to_lonlat_rad(w2_new)

            # Manually set the left and right boundaries to avoid error accumulation
            if i == 0:
                w1_lonlat[1] = b_lonlat[1]
            elif i >= 9:
                w2_lonlat[1] = c_lonlat[1]

            max_lat = max(max_lat, w1_lonlat[1], w2_lonlat[1])

            if np.absolute(w2_lonlat[1] -
                           w1_lonlat[1]) <= err_tolerance or w1_lonlat[
                1] == max_lat == w2_lonlat[1]:
                max_section = [w1_new, w2_new]
                break

            # if the largest absolute value of lat at each sub-interval point b_i.
            # Repeat algorithm with the sub-interval points (b,c)=(b_{i-1},b_{i+1})
            if np.absolute(max_lat - w1_lonlat[1]) <= err_tolerance:
                if i != 0:
                    angle_rad_prev -= avg_angle_rad
                    w1_new = [np.cos(angle_rad_prev) * v_b[i] + np.sin(
                        angle_rad_prev) * v0[i] for i in range(0, len(v_b))]
                    w2_new = [np.cos(angle_rad_next) * v_b[i] + np.sin(
                        angle_rad_next) * v0[i] for i in range(0, len(v_b))]
                    max_section = [w1_new, w2_new]
                else:
                    max_section = [v_b, w2_new]

            elif np.absolute(max_lat - w2_lonlat[1]) <= err_tolerance:
                if i != 9:
                    angle_rad_next += avg_angle_rad
                    w1_new = [np.cos(angle_rad_prev) * v_b[i] + np.sin(
                        angle_rad_prev) * v0[i] for i in range(0, len(v_b))]
                    w2_new = [np.cos(angle_rad_next) * v_b[i] + np.sin(
                        angle_rad_next) * v0[i] for i in range(0, len(v_b))]
                    max_section = [w1_new, w2_new]
                else:
                    max_section = [w1_new, v_c]

        b_lonlat = convert_node_xyz_to_lonlat_rad(copy.deepcopy(max_section[0]))
        c_lonlat = convert_node_xyz_to_lonlat_rad(copy.deepcopy(max_section[1]))

    return np.average([b_lonlat[1], c_lonlat[1]])


# Quantitative method to find the minimum latitude between in a great circle arc recursively
def min_latitude_rad(v1, v2):
    """Quantitative method to find the minimum latitude between in a great circle arc recursively
    Parameters:
        v1: float array [lon, lat] in degree east
        v2: float array [lon, lat] in degree east

    Returns: float, minimum latitude in radian
    """

    # Find the parametrized equation for the great circle passing through v1 and v2
    err_tolerance = 1.0e-15
    b_lonlat = np.deg2rad(v1)
    c_lonlat = np.deg2rad(v2)

    v1_cart = convert_node_lonlat_rad_to_xyz(np.deg2rad(v1))
    v2_cart = convert_node_lonlat_rad_to_xyz(np.deg2rad(v2))
    v_temp = np.cross(v1_cart, v2_cart)
    v0 = np.cross(v_temp, v1_cart)
    v0 = normalize_in_place(v0)

    min_section = [v1_cart,
                   v2_cart]  # record the subsection that has the maximum latitude

    # Only stop the iteration when two endpoints are extremely closed
    while np.absolute(b_lonlat[1] - c_lonlat[1]) >= err_tolerance or np.absolute(
            b_lonlat[0] - c_lonlat[0]) >= err_tolerance:
        min_lat = np.pi  # reset the max_latitude for each while loop
        v_b = min_section[0]
        v_c = min_section[1]

        # Divide the angle of v1/v2 into 10 subsections, the leftover will be put in the last one
        # Update v0 based on min_section[0], since the angle is always from min_section[0] to v0
        angle_v1_v2_rad = angle_of_2_vectors(v_b, v_c)
        v0 = np.cross(v_temp, v_b)
        v0 = normalize_in_place(v0)
        avg_angle_rad = angle_v1_v2_rad / 10

        for i in range(0, 10):
            angle_rad_prev = avg_angle_rad * i
            if i >= 9:
                angle_rad_next = angle_v1_v2_rad
            else:
                angle_rad_next = angle_rad_prev + avg_angle_rad

            # Get the two vectors of this section
            w1_new = [np.cos(angle_rad_prev) * v_b[i] + np.sin(
                angle_rad_prev) * v0[i] for i in range(0, len(v_b))]
            w2_new = [np.cos(angle_rad_next) * v_b[i] + np.sin(
                angle_rad_next) * v0[i] for i in range(0, len(v_b))]

            # convert the 3D [x, y, z] vector into 2D lat/lon vector
            w1_lonlat = convert_node_xyz_to_lonlat_rad(w1_new)
            w2_lonlat = convert_node_xyz_to_lonlat_rad(w2_new)

            # Manually set the left and right boundaries to avoid error accumulation
            if i == 0:
                w1_lonlat[1] = b_lonlat[1]
            elif i >= 9:
                w2_lonlat[1] = c_lonlat[1]

            min_lat = min(min_lat, w1_lonlat[1], w2_lonlat[1])

            if np.absolute(w2_lonlat[1] -
                           w1_lonlat[1]) <= err_tolerance or w1_lonlat[
                1] == min_lat == w2_lonlat[1]:
                min_section = [w1_new, w2_new]
                break

            # if the largest absolute value of lat at each sub-interval point b_i.
            # Repeat algorithm with the sub-interval points (b,c)=(b_{i-1},b_{i+1})
            if np.absolute(min_lat - w1_lonlat[1]) <= err_tolerance:
                if i != 0:
                    angle_rad_prev -= avg_angle_rad
                    w1_new = [np.cos(angle_rad_prev) * v_b[i] + np.sin(
                        angle_rad_prev) * v0[i] for i in range(0, len(v_b))]
                    w2_new = [np.cos(angle_rad_next) * v_b[i] + np.sin(
                        angle_rad_next) * v0[i] for i in range(0, len(v_b))]
                    min_section = [w1_new, w2_new]
                else:
                    min_section = [v_b, w2_new]

            elif np.absolute(min_lat - w2_lonlat[1]) <= err_tolerance:
                if i != 9:
                    angle_rad_next += avg_angle_rad
                    w1_new = [np.cos(angle_rad_prev) * v_b[i] + np.sin(
                        angle_rad_prev) * v0[i] for i in range(0, len(v_b))]
                    w2_new = [np.cos(angle_rad_next) * v_b[i] + np.sin(
                        angle_rad_next) * v0[i] for i in range(0, len(v_b))]
                    min_section = [w1_new, w2_new]
                else:
                    min_section = [w1_new, v_c]

        b_lonlat = convert_node_xyz_to_lonlat_rad(copy.deepcopy(min_section[0]))
        c_lonlat = convert_node_xyz_to_lonlat_rad(copy.deepcopy(min_section[1]))

    return np.average([b_lonlat[1], c_lonlat[1]])


# Quantitative method to find the minimum and maximum Longitude between in a great circle
def minmax_Longitude_rad(v1, v2):
    """Quantitative method to find the minimum Longitude between in a great circle arc.
      And it assumes that an edge's longitude span cannot be larger than 180 degree.
    Parameters:
        v1: float array [lon, lat] in degree east
        v2: float array [lon, lat] in degree east

    Returns: float array, [lon_min, lon_max] in radian
    """
    # First reorder the two ends points based on the rule: the span of its longitude must less than 180 degree
    [start_lon, end_lon] = np.sort([v1[0], v2[0]])
    if end_lon - start_lon <= 180:
        return [np.deg2rad(start_lon), np.deg2rad(end_lon)]
    else:
        # swap the start and end longitude
        temp_lon = start_lon
        start_lon = end_lon
        end_lon = temp_lon
    return [np.deg2rad(start_lon), np.deg2rad(end_lon)]


# helper function to calculate the point position of the intersection
def get_intersection_point(w0, w1, v0, v1):
    """Helper function to calculate the intersection point of two great circle
    arcs in 3D coordinates.

    Parameters
    ----------
    w0: float array [x, y, z], the end point of great circle arc w
    w1: float array [x, y, z], the other end point of great circle arc w
    v0: float array [x, y, z], the end point of great circle arc v
    v1: float array [x, y, z], the other end point of great circle arc v


    Returns: float array, the result vector [x, y, z]
     [x, y, z]: the 3D coordinates of the intersection point
     [0, 0, 0]: Indication that two great circle arcs are parallel to each other
     [-1, -1, -1]: Indication that two great circle arcs doesn't have intersection
    """
    w0 = normalize_in_place(w0)
    w1 = normalize_in_place(w1)
    v0 = normalize_in_place(v0)
    v1 = normalize_in_place(v1)
    x1 = np.cross(np.cross(w0, w1), np.cross(v0, v1)).tolist()
    x2 = [-x1[0], -x1[1], -x1[2]]

    # Find out whether X1 or X2 is within the interval [wo, w1]

    if within(w0[0], x1[0], w1[0]) and within(w0[1], x1[1], w1[1]) and within(
            w0[2], x1[2], w1[2]):
        return x1
    elif within(w0[0], x2[0], w1[0]) and within(w0[1], x2[1], w1[1]) and within(
            w0[2], x2[2], w1[2]):
        return x2
    elif x1[0] == 0 and x1[1] == 0 and x1[2] == 0:
        return [0, 0, 0]  # two vectors are parallel to each other
    else:
        return [-1, -1, -1]  # Intersection out of the interval or


# Helper function for the test_generate_Latlon_bounds_longitude_minmax
def expand_longitude_rad(min_lon_rad_edge, max_lon_rad_edge, minmax_lon_rad_face):
    """Helper function top expand the longitude boundary of a face

    Parameters
    ----------
    min_lon_rad_edge, max_lon_rad_edge: float
    minmax_lon_rad_face: float array [min_lon_rad_face, max_lon_rad_face]

    Returns:
    minmax_lon_rad_face: float array [new_min_lon_rad_face, new_max_lon_rad_face]
    """
    # Longitude range expansion: Compare between [min_lon_rad_edge, max_lon_rad_edge] and minmax_lon_rad_face
    if minmax_lon_rad_face[0] <= minmax_lon_rad_face[1]:
        if min_lon_rad_edge <= max_lon_rad_edge:
            if min_lon_rad_edge < minmax_lon_rad_face[0] and max_lon_rad_edge < minmax_lon_rad_face[1]:
                # First try to add from the left:
                left_width = minmax_lon_rad_face[1] - min_lon_rad_edge
                if left_width <= np.pi:
                    minmax_lon_rad_face = [min_lon_rad_edge, minmax_lon_rad_face[1]]
                else:
                    # add from the right:
                    minmax_lon_rad_face = [minmax_lon_rad_face[0], min_lon_rad_edge]

            elif min_lon_rad_edge > minmax_lon_rad_face[0] and max_lon_rad_edge > minmax_lon_rad_face[1]:
                # First try to add from the right
                right_width = max_lon_rad_edge - minmax_lon_rad_face[0]
                if right_width <= np.pi:
                    minmax_lon_rad_face = [minmax_lon_rad_face[0], max_lon_rad_edge]
                else:
                    # then add from the left
                    minmax_lon_rad_face = [max_lon_rad_edge, minmax_lon_rad_face[1]]

            else:
                minmax_lon_rad_face = [min(min_lon_rad_edge, minmax_lon_rad_face[0]),
                                       max(max_lon_rad_edge, minmax_lon_rad_face[1])]

        else:
            # The min_lon_rad_edge is on the left side of minmax_lon_rad_face range
            if minmax_lon_rad_face[1] <= np.pi:
                minmax_lon_rad_face = [min_lon_rad_edge, max(max_lon_rad_edge, minmax_lon_rad_face[1])]
            else:
                # if it's on the right side of the minmax_lon_rad_face range
                minmax_lon_rad_face = [min(min_lon_rad_edge, minmax_lon_rad_face[0]), max_lon_rad_edge]

    else:
        if min_lon_rad_edge <= max_lon_rad_edge:
            if __on_left(minmax_lon_rad_face, [min_lon_rad_edge, max_lon_rad_edge], safe_call=True):
                # First try adding from the left:
                left_width = (2 * np.pi - min_lon_rad_edge) + minmax_lon_rad_face[1]
                if left_width <= np.pi:
                    minmax_lon_rad_face = [min_lon_rad_edge, minmax_lon_rad_face[1]]
                else:
                    # Then add from the right
                    minmax_lon_rad_face = [minmax_lon_rad_face[0], min_lon_rad_edge]

            elif __on_right(minmax_lon_rad_face, [min_lon_rad_edge, max_lon_rad_edge], safe_call=True):
                # First try adding from the right
                right_width = (2 * np.pi - minmax_lon_rad_face[0]) + max_lon_rad_edge
                if right_width <= np.pi:
                    minmax_lon_rad_face = [minmax_lon_rad_face[0], max_lon_rad_edge]
                else:
                    # Then try adding from the left
                    minmax_lon_rad_face = [max_lon_rad_edge, minmax_lon_rad_face[1]]

            else:
                if within(minmax_lon_rad_face[1], min_lon_rad_edge, minmax_lon_rad_face[0]):
                    minmax_lon_rad_face[0] = min_lon_rad_edge
                else:
                    minmax_lon_rad_face[0] = minmax_lon_rad_face[0]

                if 2 * np.pi > max_lon_rad_edge >= minmax_lon_rad_face[0] or max_lon_rad_edge < minmax_lon_rad_face[1]:
                    minmax_lon_rad_face[1] = minmax_lon_rad_face[1]
                else:
                    minmax_lon_rad_face[1] = max(minmax_lon_rad_face[1], max_lon_rad_edge)

        else:
            minmax_lon_rad_face[0] = min(min_lon_rad_edge, minmax_lon_rad_face[0])
            minmax_lon_rad_face[1] = max(max_lon_rad_edge, minmax_lon_rad_face[1])

    return minmax_lon_rad_face


# helper function to determine whether the insert_edge is on the left side of the ref_edge
def __on_left(ref_edge, insert_edge, safe_call=False):
    """Helper function used for the longitude test case only. Only designed to consider a specific scenario
    as described below

    Parameters
    ----------
    ref_edge: The edge that goes across the 0 longitude line: [min_longitude, max_longitude] and min_long > max_long

    insert_edge: the inserted edge, [min_longitude, max_longitude]

    safe_call (default to be False): When call this function, user must make sure it's under the safe and ideal condition

    Returns: boolean

    True: the insert_edge is on the left side of the ref_edge ( the insert_edge's min_longitude
            is larger than 180 longitude, and its max_longitude between 180 longitude and the max_longitude of the ref_edge
    False: It's not on the left side of the ref_edge. Cannot guarantee it's on the right side
    """
    if ref_edge[0] <= ref_edge[1]:
        raise Exception('This function can only be applied to the edge that goes across the 0 longitude line')
    if not safe_call:
        raise Exception('Calling this function here is not safe')
    left_flag = False
    if insert_edge[1] >= ref_edge[1] and insert_edge[1] >= ref_edge[0]:
        if within(ref_edge[1], insert_edge[0], ref_edge[0]):
            left_flag = True
    elif insert_edge[1] <= ref_edge[1] and insert_edge[1] <= ref_edge[0]:
        if within(ref_edge[1], insert_edge[0], ref_edge[0]):
            left_flag = True
    return left_flag


# helper function to determine whether the insert_edge is on the right side of the ref_edge
def __on_right(ref_edge, insert_edge, safe_call=False):
    """Helper function used for the longitude test case only. Only designed to consider a specific scenario
    as described below

    Parameters
    ----------
    ref_edge: The edge that goes across the 0 longitude line: [min_longitude, max_longitude] and min_long > max_long

    insert_edge: the inserted edge, [min_longitude, max_longitude]

    safe_call (default to be False): When call this function, user must make sure it's under the safe and ideal condition

    Returns: boolean

    True: the insert_edge is on the right side of the ref_edge ( the insert_edge's min_longitude
            is between the ref_edge's min_longitude and 0 longitude, and the insert_edge's max_longitude is between
            ref_edge's max_longitude and 180 longitude
    False: It's not on the right side of the ref_edge. Cannot guarantee it's on the left side
    """
    if ref_edge[0] <= ref_edge[1]:
        raise Exception('This function can only be applied to the edge that goes across the 0 longitude line')
    if not safe_call:
        raise Exception('Calling this function here is not safe')
    right_flag = False
    if insert_edge[0] >= ref_edge[0] and insert_edge[0] >= ref_edge[1]:
        if within(ref_edge[1], insert_edge[1], ref_edge[0]):
            right_flag = True
    elif insert_edge[0] <= ref_edge[0] and insert_edge[0] <= ref_edge[1]:
        if within(ref_edge[1], insert_edge[1], ref_edge[0]):
            right_flag = True

    return right_flag


# helper function for get_intersection_point to determine whether one point is between the other two points
def within(p, q, r):
    """Helper function for get_intersection_point to determine whether the
    number q is between p and r.

    Parameters
    ----------
    p, q, r: float

    Returns: boolean
    """
    return p <= q <= r or r <= q <= p


class Edge:
    """The Uxarray Edge object class for undirected edge.

    In current implementation, each node is the node index
    """

    def __init__(self, input_edge):
        """Initializing the Edge object from input edge [node 0, node 1]

        Parameters
        ----------

        input_edge : xarray.Dataset, ndarray, list, tuple, required
            - The indexes of two nodes [node0_index, node1_index], the order doesn't matter

        ----------------
        """
        # for every input_edge, sort the node index in ascending order.
        edge_sorted = np.sort(input_edge)
        self.node0 = edge_sorted[0]
        self.node1 = edge_sorted[1]

    def __lt__(self, other):
        if self.node0 != other.node0:
            return self.node0 < other.node0
        else:
            return self.node1 < other.node1

    def __eq__(self, other):
        # Undirected edge
        return (self.node0 == other.node0 and self.node1 == other.node1) or \
               (self.node1 == other.node0 and self.node0 == other.node1)

    def __hash__(self):
        # Collisions are possible for hash
        return hash(self.node0 + self.node1)

    # Return nodes in list
    def get_nodes(self):
        return [self.node0, self.node1]
