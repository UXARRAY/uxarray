import xarray as xr
from pathlib import PurePath


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
