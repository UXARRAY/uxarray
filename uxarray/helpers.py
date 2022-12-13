import numpy as np
import xarray as xr
from pathlib import PurePath
from .get_quadratureDG import get_gauss_quadratureDG, get_tri_quadratureDG
from numba import njit, config
import math

config.DISABLE_JIT = False
int_dtype = np.uint32


def parse_grid_type(dataset):
    """Checks input and contents to determine grid type. Supports detection of
    UGrid, SCRIP, Exodus and shape file.

    Parameters
    ----------
    dataset : Xarray dataset
       Xarray dataset of the grid

    Returns
    -------
    mesh_type : str
        File type of the file, ug, exo, scrip or shp

    Raises
    ------
    RuntimeError
            If invalid file type
    ValueError
        If file is not in UGRID format
    """
    # exodus with coord or coordx
    if "coord" in dataset:
        mesh_type = "exo"
    elif "coordx" in dataset:
        mesh_type = "exo"
    # scrip with grid_center_lon
    elif "grid_center_lon" in dataset:
        mesh_type = "scrip"
    # ugrid topology
    elif _is_ugrid(dataset):
        mesh_type = "ugrid"
    else:
        raise RuntimeError(f"Could not recognize dataset format.")
    return mesh_type

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
            mesh_type = "ugrid"
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
            mesh_type = "shp"
        else:
            msg = str(e) + ': {}'.format(filepath)
    except ValueError as e:
        # check if this is a shp file
        # we won't use xarray to load that file
        if file_extension == ".shp":
            mesh_type = "shp"
        else:
            msg = str(e) + ': {}'.format(filepath)
    finally:
        if msg != "":
            msg = "Unable to determine file type, mesh file not supported" + ': {}'.format(
                filepath)
            raise ValueError(msg)

    return mesh_type


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
            node1 = np.array(
                _convert_node_lonlat_rad_to_xyz(
                    [np.deg2rad(x[0]), np.deg2rad(y[0])]))
            node2 = np.array(
                _convert_node_lonlat_rad_to_xyz(
                    [np.deg2rad(x[j + 1]),
                     np.deg2rad(y[j + 1])]))
            node3 = np.array(
                _convert_node_lonlat_rad_to_xyz(
                    [np.deg2rad(x[j + 2]),
                     np.deg2rad(y[j + 2])]))

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

    face_nodes = face_nodes[:].astype(int_dtype)

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


def grid_center_lat_lon(ds):
    """Using scrip file variables ``grid_corner_lat`` and ``grid_corner_lon``,
    calculates the ``grid_center_lat`` and ``grid_center_lon``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset that contains ``grid_corner_lat`` and ``grid_corner_lon``
        data variables

    Returns
    -------
    center_lon : :class:`numpy.ndarray`
        The calculated center longitudes of the grid box based on the corner
        points
    center_lat : :class:`numpy.ndarray`
        The calculated center latitudes of the grid box based on the corner
        points
    """

    # Calculate and create grid center lat/lon
    scrip_corner_lon = ds['grid_corner_lon']
    scrip_corner_lat = ds['grid_corner_lat']

    # convert to radians
    rad_corner_lon = np.deg2rad(scrip_corner_lon)
    rad_corner_lat = np.deg2rad(scrip_corner_lat)

    # get nodes per face
    nodes_per_face = rad_corner_lat.shape[1]

    # geographic center of each cell
    x = np.sum(np.cos(rad_corner_lat) * np.cos(rad_corner_lon),
               axis=1) / nodes_per_face
    y = np.sum(np.cos(rad_corner_lat) * np.sin(rad_corner_lon),
               axis=1) / nodes_per_face
    z = np.sum(np.sin(rad_corner_lat), axis=1) / nodes_per_face

    center_lon = np.rad2deg(np.arctan2(y, x))
    center_lat = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))

    # Make negative lons positive
    center_lon[center_lon < 0] += 360

    return center_lat, center_lon


@njit
def _convert_node_lonlat_rad_to_xyz(node_coord):
    """Helper function to Convert the node coordinate from 2D
    longitude/latitude to normalized 3D xyz.

    Parameters
    ----------
    node: float list
        2D coordinates[longitude, latitude] in radiance

    Returns
    ----------
    float list
        the result array of the unit 3D coordinates [x, y, z] vector where :math:`x^2 + y^2 + z^2 = 1`

    Raises
    ----------
    RuntimeError
        The input array doesn't have the size of 3.
    """
    if len(node_coord) != 2:
        raise RuntimeError(
            "Input array should have a length of 2: [longitude, latitude]")
    lon = node_coord[0]
    lat = node_coord[1]
    return [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)]


@njit
def _convert_node_xyz_to_lonlat_rad(node_coord):
    """Calculate the latitude and longitude in radiance for a node represented
    in the [x, y, z] 3D Cartesian coordinates.

    Parameters
    ----------
    node_coord: float list
        3D Cartesian Coordinates [x, y, z] of the node

    Returns
    ----------
    float list
        the result array of longitude and latitude in radian [longitude_rad, latitude_rad]

    Raises
    ----------
    RuntimeError
        The input array doesn't have the size of 3.
    """
    if len(node_coord) != 3:
        raise RuntimeError("Input array should have a length of 3: [x, y, z]")
    reference_tolerance = 1.0e-12
    [dx, dy, dz] = _normalize_in_place(node_coord)
    dx /= np.absolute(dx * dx + dy * dy + dz * dz)
    dy /= np.absolute(dx * dx + dy * dy + dz * dz)
    dz /= np.absolute(dx * dx + dy * dy + dz * dz)

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


@njit
def _normalize_in_place(node):
    """Helper function to project an arbitrary node in 3D coordinates [x, y, z]
    on the unit sphere. It uses the `np.linalg.norm` internally to calculate
    the magnitude.

    Parameters
    ----------
    node: float list
        3D Cartesian Coordinates [x, y, z]

    Returns
    ----------
    float list
        the result unit vector [x, y, z] where :math:`x^2 + y^2 + z^2 = 1`

    Raises
    ----------
    RuntimeError
        The input array doesn't have the size of 3.
    """
    if len(node) != 3:
        raise RuntimeError("Input array should have a length of 3: [x, y, z]")

    return list(np.array(node) / np.linalg.norm(np.array(node), ord=2))
