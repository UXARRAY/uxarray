import xarray as xr
from pathlib import PurePath
import numpy as np
import copy


def parse_grid_type(filepath, **kw):
    """Checks input and contents to determine grid type. Supports detection of
    UGrid, SCRIP, Exodus and shape file.
    Parameters: string, required
       Filepath of the file for which the filetype is to be determined.
    Returns: string and ugrid aware xarray.Dataset
       File type: ug, exo, scrip or shp
    Raises:
       RuntimeError: Invalid grid type
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


# Convert the node coordinate from 2D longitude/latitude to normalized 3D xyz
def convert_node_latlon_rad_to_xyz(node_coord):
    """
    Parameters: float list, required
       the input 2D coordinates[latitude, longitude]
    Returns: float list, the 3D coordinates in [X, Y, Z]
    """
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
    latlon_box = old_box  # The returned box

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
        return latlon_box

    # New longitude lies within existing range
    if latlon_box[1][0] <= latlon_box[1][1]:
        if lon_pt >= latlon_box[1][0] and lon_pt <= latlon_box[1][1]:
            return latlon_box
    else:
        if lon_pt >= latlon_box[1][0] or lon_pt <= latlon_box[1][0]:
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

    if (d_width_a - d_width_now) < -1.0e-14 or (d_width_b - d_width_now) < -1.0e-14:
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


# helper function to calculate the dot product of two vectors in 3D [x, y, z] coordinates
def dot_product(vec0, vec1):
    return vec0[0] * vec1[0] + vec0[1] * vec1[1] + vec0[2] * vec1[2]


# helper function to calculate the cross product of two vectors with the same dimensions
def cross_product(vec0, vec1):
    """Helper function to calculate the cross product of two vectors. Only support the
     3D calculation. The two vectors must have the same dimensions

     Parameters
     ----------
     vec0: list [x, y, z]
     vec1: list [x, y, z]

     Returns: float array, the result vector [x, y, z]

    """

    # check if two vectors are in the same dimensions:
    if len(vec0) != len(vec1):
        raise ValueError("two vectors must have the same dimension")

    if len(vec0) != 3:
        raise ValueError("The vector dimensions have to be 3")
    vec_x = vec0[1] * vec1[2] - vec0[2] * vec1[1]
    vec_y = vec0[0] * vec1[2] - vec0[2] * vec1[0]
    vec_z = vec0[0] * vec1[1] - vec0[1] * vec1[0]
    return [vec_x, vec_y, vec_z]


# helper function to project node on the unit sphere
def normalize_in_place(node):
    """Helper function to project an arbitrary node in 3D coordinates [x, y, z] on the unit sphere

     Parameters
     ----------
     node: float array [x, y, z]

     Returns: float array, the result vector [x, y, z]

    """
    magnitude = np.sqrt(node[0] * node[0] + node[1] * node[1] + node[2] * node[2])
    return [node[0] / magnitude, node[1] / magnitude, node[2] / magnitude]


# helper function to calculate the point position of the intersection
def get_intersection_point(w0, w1, v0, v1):
    """Helper function to calculate the intersection point of two great circle arcs in 3D coordinates

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
    x1 = cross_product(cross_product(w0, w1), cross_product(v0, v1))
    x2 = [-x1[0], -x1[1], -x1[2]]

    # Find out whether X1 or X2 is within the interval [wo, w1]

    if within(w0[0], x1[0], w1[0]) and within(w0[1], x1[1], w1[1]) and within(w0[2], x1[2], w1[2]):
        return x1
    elif within(w0[0], x2[0], w1[0]) and within(w0[1], x2[1], w1[1]) and within(w0[2], x2[2], w1[2]):
        return x2
    elif x1[0] * x1[1] * x1[0] == 0:
        return [0, 0, 0]  # two vectors are parallel to each other
    else:
        return [-1, -1, -1]  # Intersection out of the interval or


# helper function for get_intersection_point to determine whether one point is between the other two points
def within(p, q, r):
    """Helper function for get_intersection_point to determine whether the number q is between p and r

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
        """ Initializing the Edge object from input edge [node 0, node 1]

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
