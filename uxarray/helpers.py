import os
import numpy as np
import xarray as xr
from pathlib import PurePath
import heapq


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
                    ext_ds = xr.open_dataset(filepath, mask_and_scale=False)
                    node_coords_dv = ext_ds.filter_by_attrs(
                        node_coordinates=standard_name).keys()
                    face_conn_dv = ext_ds.filter_by_attrs(
                        face_node_connectivity=standard_name).keys()
                    topo_dim_dv = ext_ds.filter_by_attrs(
                        topology_dimension=standard_name).keys()
                    mesh_topo_dv = ext_ds.filter_by_attrs(
                        cf_role="mesh_topology").keys()
                    if list(mesh_topo_dv)[0] != "" and list(topo_dim_dv)[
                        0] != "" and list(face_conn_dv)[0] != "" and list(
                        node_coords_dv)[0] != "":
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
        if msg != "":  # we did not catch this above
            msg = "Unable to determine file type, mesh file not supported" + ': {}'.format(
                filepath)
            print(msg)
            os._exit(0)

    return mesh_filetype


# Convert the node coordinate from 2D longitude/latitude to normalized 3D xyz
def convert_node_latlon_rad_to_xyz(node_coord):
    lon = node_coord[0]
    lat = node_coord[1]
    return [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)]


# helper function to calculate latitude and longitude from a node's normalized 3D Cartesian
# coordinates, in radians.
def convert_node_xyz_to_latlon_rad(node_coord):
    """Calculate the latitude and longitude in radiance for a node represented in the [x, y, z] 3D Cartesian coordinates.
    Parameters: node_coord: float array, [x, y, z],required

    Returns: float array, [latitude_rad, longitude_rad]

    Raises:
       Exception: Logic Errors
    """
    reference_tolerance = 1.0e-12
    dx = node_coord[0]
    dy = node_coord[1]
    dz = node_coord[2]

    d_mag_2 = dx * dx + dy * dy + dz * dz
    # if np.absolute(d_mag_2 - 1.0) >= 0.01:
    #     raise Exception('Grid point has non-unit magnitude:({}, {}, {}) (magnitude {})'.format(dx, dy, dz,d_mag_2 )) #"(%1.15e, %1.15e, %1.15e) (magnitude %1.15e)",

    d_mag = np.absolute(d_mag_2)
    dx /= d_mag
    dy /= d_mag
    dz /= d_mag

    d_lon_rad = 0.0
    d_lat_rad = 0.0

    if np.absolute(dz) < (1.0 - reference_tolerance):
        d_lon_rad = np.arctan(dy / dx)
        d_lat_rad = np.arcsin(dz)

        if d_lon_rad < 0.0:
            d_lon_rad += 2.0 * np.pi
    elif dz > 0.0:
        d_lon_rad = 0.0
        d_lat_rad = 0.5 * np.pi
    else:
        d_lon_rad = 0.0
        d_lat_rad = -0.5 * np.pi

    return [d_lat_rad, d_lon_rad]


# helper function to insert a new point into the latlon box
def insert_pt_in_latlonbox(old_box, new_pt, is_lon_periodic=True):
    """Compare the new point's latitude and longitude with the target the latlonbox.

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
        raise Exception('lon_pt out of range ( {} > {})"'.format(lon_pt, old_lon_width))

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
        if lon_pt >= latlon_box[1][0] and lon_pt <= latlon_box[1][1]:
            return
    else:
        if lon_pt >= latlon_box[1][0] or lon_pt <= latlon_box[1][0]:
            return

    # New longitude lies outside of existing range
    box_a = latlon_box
    box_a[1][0] = lon_pt

    box_b = latlon_box
    box_b[1][1] = lon_pt

    # The updated box is the box of minimum width
    d_width_now = get_latlonbox_width(latlon_box)
    d_width_a = get_latlonbox_width(box_a)
    d_width_b = get_latlonbox_width(box_b)

    if (d_width_a - d_width_now) < -1.0e-14 or (d_width_b - d_width_now) < -1.0e-14:
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


def dot_product(vec0, vec1):
    return vec0[0] * vec1[0] + vec0[1] * vec1[1] + vec0[2] * vec1[2]


# helper function to calculate the cross product of two vectors with the same dimensions
def cross_product(vec0, vec1):
    """Helper function to calculate the cross product of two vectors. Only support the
     2D and 3D calculation. The two vectors must have the same dimensions

     Parameters
     ----------
     vec0: list [x, y, z(optional)]
     vec1: list [x, y, z(optional)]

    """

    # check if two vectors are in the same dimensions:
    if len(vec0) != len(vec1):
        raise ValueError("two vectors must have the same dimension")

    if len(vec0) == 2:
        return vec0[0] * vec1[1] - vec0[1] * vec1[0]
    elif len(vec0) == 3:
        vec_x = vec0[1] * vec1[2] - vec0[2] * vec1[1]
        vec_y = vec0[0] * vec1[2] - vec0[2] * vec1[0]
        vec_z = vec0[0] * vec1[1] - vec0[1] * vec1[0]
        return [vec_x, vec_y, vec_z]
    else:
        raise ValueError("The vector dimensions cannot be larger than 3")


# helper function to project node on the unit sphere
def normalize_in_place(node):
    if len(node) == 2:
        magnitude = np.sqrt(node[0] * node[0] + node[1] * node[1])
        return [node[0] / magnitude, node[1] / magnitude]

    elif len(node) == 3:
        magnitude = np.sqrt(node[0] * node[0] + node[1] * node[1] + node[2] * node[2])
        return [node[0] / magnitude, node[1] / magnitude, node[2] / magnitude]


# helper function to calculate the point position of the intersection
def get_intersection_point(w0, w1, v0, v1):
    w0 = normalize_in_place(w0)
    w1 = normalize_in_place(w1)
    v0 = normalize_in_place(v0)
    v1 = normalize_in_place(v1)
    x1 = cross_product(cross_product(w0, w1), cross_product(v0, v1))
    x2 = -1 * x1

    # Find out whether X1 or X2 is within the interval [wo, w1]

    if within(w0[0], x1[0], w1[0]) and within(w0[1], x1[1], w1[1]):
        if (len(x1) == 3 and within(w0[2], x1[2], w1[2])) or len(x1) == 2:
            return x1
    elif within(w0[0], x2[0], w1[0]) and within(w0[1], x2[1], w1[1]):
        if (len(x2) == 3 and within(w0[2], x2[2], w1[2])) or len(x2) == 2:
            return x2
    elif x1[0] * x1[1] * x1[0] == 0:
        return [0, 0, 0]  # two vectors are parallel to each other
    else:
        return [-1, -1, -1]  # Intersection out of the interval or


def within(p, q, r):
    return p <= q <= r or r <= q <= p


def is_parallel(w0, w1, v0, v1):
    res = get_intersection_point(w0, w1, v0, v1)
    if res[0] * res[1] * res[2] == 0:
        return True
    else:
        return False


def is_intersect(w0, w1, v0, v1):
    res = get_intersection_point(w0, w1, v0, v1)

    # two vectors are intersected within range and not parralel
    if (res != [0, 0, 0]) and (res != [-1, -1, -1]):
        return True
    else:
        return False


# Helper function to sort edges of a polygon using dictionaries
def sort_edge(edge_list):
    dict_edge = {}
    for edge in edge_list:
        edge = np.sort(edge)
        if edge[0] in dict_edge:
            new_list = dict_edge[edge[0]]
            new_list.append(edge[1])
            dict_edge[edge[0]] = new_list
        else:
            dict_edge[edge[0]] = [edge[1]]

    dict_edge = sorted(dict_edge.items())

    # Pack them back as a list
    res = []
    for k, v in dict_edge:
        v.sort()
        for node_1 in v:
            res.append([k, node_1])
    return res


class Edge:
    """The Uxarray Edge object class for undirected edge.

    """

    def __init__(self, input_edge):
        """ Initializing the Edge object from input edge [node 0, node 1]

        """
        # for every input_edge, sort the node index in ascending order.
        edge_sorted = np.sort(input_edge)
        self.node0_index = edge_sorted[0]
        self.node1_index = edge_sorted[1]

    # TODO: PriorityQueue is not working properly for the current implementation now
    def __lt__(self, other):
        # self < other, sorted by the lowest node index
        if self.node0_index != other.node0_index:
            return self.node0_index < other.node0_index
        else:
            return self.node1_index < other.node1_index

    def __eq__(self, other):
        # Undirected edge
        return (self.node0_index == other.node0_index and self.node1_index == other.node1_index) or \
               (self.node1_index == other.node0_index and self.node0_index == other.node1_index)

    def __hash__(self):
        # Collisions are possible for hash
        return hash(self.node0_index + self.node1_index)

    # Return nodes in list
    def get_nodes(self):
        return [self.node0_index, self.node1_index]


class Node:
    def __init__(self, input_node):
        self.x = input_node[0]
        self.y = input_node[1]
        self.z = input_node[2]

    def __lt__(self, other):
        # sorted in the order of x, y, z
        if self.x != other.x:
            return self.x < other.x
        elif self.y != other.y:
            return self.y < other.y
        else:
            return self.z < self.z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash(self.x + self.y + self.z)
