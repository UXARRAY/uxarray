import numpy as np
import xarray as xr
from pathlib import PurePath
from .get_quadratureDG import get_gauss_quadratureDG, get_tri_quadratureDG
from numba import njit, config
import math

from uxarray._zonal_avg_utilities import _newton_raphson_solver_for_intersection_pts

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

    Parameters
    ----------
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


# Convert the node coordinate from 2D longitude/latitude to normalized 3D xyz
def convert_node_lonlat_rad_to_xyz(node_coord):
    """
    Parameters: float list, required
       the input 2D coordinates[longitude, latitude] in radiance
    Returns: float list, the 3D coordinates in [x, y, z]
    """
    lon = node_coord[0]
    lat = node_coord[1]
    return [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)]


# helper function to calculate latitude and longitude from a node's normalized 3D Cartesian
# coordinates, in radians.
def convert_node_xyz_to_lonlat_rad(node_coord):
    """Calculate the latitude and longitude in radiance for a node represented
    in the [x, y, z] 3D Cartesian coordinates.

    Parameters: node_coord: float array, [x, y, z],required
    Returns: float array, [longitude_rad, latitude_rad]
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


# helper function to project node on the unit sphere
def normalize_in_place(node):
    """Helper function to project an arbitrary node in 3D coordinates [x, y, z]
    on the unit sphere.
    Parameters
    ----------
    node: float array [x, y, z]
    Returns: float array, the result vector [x, y, z]
    """
    magnitude = np.sqrt(node[0] * node[0] + node[1] * node[1] +
                        node[2] * node[2])
    if magnitude == 0:
        return [0,0,0]

    return [node[0] / magnitude, node[1] / magnitude, node[2] / magnitude]


# helper function to calculate the angle of 3D vectors u,v in radian
def _angle_of_2_vectors(u, v):
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


# helper function for get_intersection_point to determine whether one point is between the other two points
def _within(p, q, r):
    """Helper function for get_intersection_point to determine whether the
    number q is between p and r.
    Parameters
    ----------
    p, q, r: float
    Returns: boolean
    """
    return p <= q <= r or r <= q <= p


# Helper function to get the radius of a constant latitude arc
def _get_radius_of_latitude_rad(latitude):
    longitude = 0.0
    [x, y, z] = convert_node_lonlat_rad_to_xyz([longitude, latitude])
    radius = np.sqrt(x * x + y * y)
    return radius


def _get_approx_intersection_point_gcr_constlat(gcr_cart, const_lat_rad):
    """Helper function to get the approximate cartesian coordinates intersections of a great circle arc and line of constant latitude
    Details explained in the paper chapt.2.2
    """
    [n1, n2] = gcr_cart
    gcr_rad = [convert_node_xyz_to_lonlat_rad(n1),convert_node_xyz_to_lonlat_rad(n2)]
    res = [[-1, -1, -1], [-1, -1, -1]]

    #  Determine if latitude is maximized between endpointscted latitude on the interval a ∈ [0, 1].
    min_lat = min(gcr_rad[0][1],gcr_rad[1][1])
    max_lat = get_gcr_max_lat_rad(gcr_cart)
    if not _within(min_lat, const_lat_rad, max_lat):
        return res
    # If z1 = z2 = 0 then the great circle arc corresponds to the equator.
    if n1[2] == n2[2] == 0 and const_lat_rad != 0:
        return res

    # To maximize conditioning, one should choose x1 to satisfy |z1| ≥ |z2|. If this inequality does not hold,
    # x1 and x2 should be swapped first
    if n1[2] < n2[2]:
        temp = n1
        n1 = n2
        n2 = temp

    z_0 = np.sin(const_lat_rad)
    n_x = n1[1] * n2[2] - n2[1] * n1[2]
    n_y = -n1[0] * n2[2] + n2[0] * n1[2]
    a = n_x * n_x + n_y * n_y
    b = 2 * z_0 * np.dot(n1, n2) - 2 * z_0 * n2[2] * (1 / n1[2])
    c = (z_0 ** 2) * (n1[2] ** (-2)) - 1
    if b * b - 4 * a * c < 0:
        return res

    if a == b == c == 0:
        return [[0,0,0], [-1,-1,-1]]
    [t1, t2] = np.roots([a, b, c])
    x1 = [z_0 * n1[0] * (1 / n1[2]) + t1 * n_y, z_0 * n1[1] * (1 / n1[2]) - t1 * n_x, z_0]
    x2 = [z_0 * n1[0] * (1 / n1[2]) + t2 * n_y, z_0 * n1[1] * (1 / n1[2]) - t2 * n_x, z_0]

    # Once the point of intersection x is found, one should test if either or both of these points lies on the
    # interval between x1 and x2
    if _pt_within_gcr(x1, [n1, n2]):
        res[0] = x1
    if _pt_within_gcr(x2, [n1, n2]):
        res[1] = x2

    return res


def get_intersection_pt(gcr_cart, const_lat_rad):
    const_lat_z = np.sin(const_lat_rad)
    initial_guess = _get_approx_intersection_point_gcr_constlat(gcr_cart, const_lat_rad)
    intersection_pt = [[-1, -1, -1], [-1, -1, -1]]

    if initial_guess[0] != [-1, -1, -1]:
        newton_input = [initial_guess[0][0], initial_guess[0][1], const_lat_z]
        res = _newton_raphson_solver_for_intersection_pts(newton_input, gcr_cart[0], gcr_cart[1])
        intersection_pt[0] = normalize_in_place([res[0], res[1], const_lat_z])
    if initial_guess[1] != [-1, -1, -1]:
        newton_input = [initial_guess[1][0], initial_guess[1][1], const_lat_z]
        res = _newton_raphson_solver_for_intersection_pts(newton_input, gcr_cart[0], gcr_cart[1])
        intersection_pt[1] = normalize_in_place([res[0], res[1], const_lat_z])

    return intersection_pt

def _pt_within_gcr(pt_cart, gcr_cart):
    # Helper function to determine if a point lies within the interval of the gcr

    # First determine if the pt lies on the plane defined by the gcr
    if np.absolute(np.dot(np.cross(gcr_cart[0], gcr_cart[1]), pt_cart) - 0) > 1.0e-12:
        return False

    # If we have determined the point lies on the gcr plane, we only need to check if the pt's longitude lie within
    # the gcr
    pt_lonlat_rad = convert_node_xyz_to_lonlat_rad(pt_cart)
    gcr_lonlat_rad = [convert_node_xyz_to_lonlat_rad(pt) for pt in gcr_cart]

    # Special case: when the gcr and the point are all on the same longitude line:
    if gcr_lonlat_rad[0][0] == gcr_lonlat_rad[1][0] == pt_lonlat_rad[0]:
        # Now use the latitude to determine if the pt falls between the interval
        return _within(gcr_lonlat_rad[0][1], pt_lonlat_rad[1], gcr_lonlat_rad[1][1])


    # First we need to deal with the longitude wrap-around case
    # x0--> 0 lon --> x1
    if np.absolute(gcr_lonlat_rad[1][0] -  gcr_lonlat_rad[0][0]) >= np.deg2rad(180):
        if _within(np.deg2rad(180), gcr_lonlat_rad[0][0], np.deg2rad(360)) and _within(0, gcr_lonlat_rad[1][0], np.deg2rad(180)):
            return _within(gcr_lonlat_rad[0][0], pt_lonlat_rad[0], np.deg2rad(360)) or _within(0, pt_lonlat_rad[0],gcr_lonlat_rad[1][0])
        elif _within(np.deg2rad(180), gcr_lonlat_rad[1][0], np.deg2rad(360)) and _within(0, gcr_lonlat_rad[0][0], np.deg2rad(180)):
            # x1 <-- 0 lon <-- x0
            return _within(gcr_lonlat_rad[1][0], pt_lonlat_rad[0], np.deg2rad(360)) or _within(
            0, pt_lonlat_rad[0], gcr_lonlat_rad[0][0])
    else:
        return _within(gcr_lonlat_rad[0][0], pt_lonlat_rad[0], gcr_lonlat_rad[1][0])



def _sort_intersection_pts_with_lon(pts_lonlat_rad_list, longitude_bound_rad):
    # This function will sort the intersection points while considering the longitude wrap-around problem.
    res = []
    if longitude_bound_rad[0] <= longitude_bound_rad[1]:
        # Normal case,
        for pt in pts_lonlat_rad_list:
            res.append(pt[0] - longitude_bound_rad[0])

    else:
        # The face that go across the 0 longitude
        for pt in pts_lonlat_rad_list:
            if _within(np.pi, pt[0], 2 * np.pi):
                res.append(pt[0] - longitude_bound_rad[0])
            else:
                res.append(pt[0] + (2 * np.pi - longitude_bound_rad[0]))

    res.sort()
    return res


def _get_cart_vector_magnitude(start, end):
    x1 = start
    x2 = end
    x1_x2 = [x1[0] - x2[0], x1[1] - x2[1], x1[2] - x2[2]]
    x1_x2_mag = np.sqrt(x1_x2[0] ** 2 + x1_x2[1] ** 2 + x1_x2[2] ** 2)
    return x1_x2_mag

def get_gcr_max_lat_rad(gcr_cart):
    # Helper function in paper 2.1.2 Maximum latitude of a great circle arc
    n1 = gcr_cart[0]
    n2 = gcr_cart[1]
    dot_n1_n2 = np.dot(n1, n2)
    d_de_nom = (n1[2] + n2[2]) * (dot_n1_n2 - 1.0)
    d_a_max = (n1[2] * np.dot(n1, n2) - n2[2]) / d_de_nom
    if (d_a_max > 0.0) and (d_a_max < 1.0):
        node3 = [0.0, 0.0, 0.0]
        node3[0] = n1[0] * (1 - d_a_max) + n2[0] * d_a_max
        node3[1] = n1[1] * (1 - d_a_max) + n2[1] * d_a_max
        node3[2] = n1[2] * (1 - d_a_max) + n2[2] * d_a_max
        node3 = normalize_in_place(node3)

        d_lat_rad = node3[2]

        if d_lat_rad > 1.0:
            d_lat_rad = 0.5 * np.pi
        elif d_lat_rad < -1.0:
            d_lat_rad = -0.5 * np.pi
        else:
            d_lat_rad = np.arcsin(d_lat_rad)
        return d_lat_rad
    else:
        return max(convert_node_xyz_to_lonlat_rad(n1)[1], convert_node_xyz_to_lonlat_rad(n2)[1])


