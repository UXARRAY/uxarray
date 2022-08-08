import xarray as xr
from pathlib import PurePath
import numpy as np

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

# Convert the node coordinate from 2D longitude/latitude to normalized 3D xyz
def convert_node_latlon_rad_to_xyz(node_coord):
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
        return [node[0] / magnitude, node[1] / magnitude, node[2] / magnitude]

    return [d_lat_rad, d_lon_rad]