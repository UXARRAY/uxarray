import os

import pandas as pd
import xarray as xr

from uxarray.constants import INT_FILL_VALUE
from uxarray.conventions import ugrid


def _read_fesom2_asci(grid_path):
    """Reads a FESOM2 ASCII grid file from the specified path and encodes it in
    the UGRID conventions.

    Parameters:
    -----------
    grid_path : str
        The path to the directory containing the FESOM2 ASCII grid

    Returns:
    --------
    ugrid_ds : xr.Dataset
        An xarray Dataset containing the FESOM2 grid information encoded in the UGRID conventions
    source_dims_dict : dict
        A dictionary mapping FESM2 dimensions to UGRID dimensions

    Raises:
    -------
    FileNotFoundError
        If required files are not found in the provided path. At least "nod2d.out" and "elem2d.out" are required to
        construct a UGRID-compliant grid.
    """
    source_dims_dict = {}
    ugrid_ds = xr.Dataset()

    node_lon, node_lat = _parse_nod2d(grid_path)

    ugrid_ds["node_lon"] = xr.DataArray(
        data=node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )
    ugrid_ds["node_lat"] = xr.DataArray(
        data=node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    face_node_connectivity = _parse_elem2d(grid_path)

    ugrid_ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_connectivity,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    edge_face_connectivity = _parse_edge_tri(grid_path)

    if edge_face_connectivity is not None:
        ugrid_ds["edge_face_connectivity"] = xr.DataArray(
            data=edge_face_connectivity,
            dims=ugrid.EDGE_FACE_CONNECTIVITY_DIMS,
            attrs=ugrid.EDGE_FACE_CONNECTIVITY_ATTRS,
        )

    edge_node_connectivity = _parse_edges(grid_path)

    if edge_node_connectivity is not None:
        ugrid_ds["edge_node_connectivity"] = xr.DataArray(
            data=edge_node_connectivity,
            dims=ugrid.EDGE_NODE_CONNECTIVITY_DIMS,
            attrs=ugrid.EDGE_NODE_CONNECTIVITY_ATTRS,
        )

    return ugrid_ds, source_dims_dict


def _parse_nod2d(grid_path):
    """Parses the 'nod2d.out' file to extract node longitude and latitude
    values.

    Parameters:
    -----------
    grid_path : str
        The path to the directory containing 'nod2d.out'.

    Returns:
    --------
    node_lon : np.ndarray
        Array of node longitude values.
    node_lat : np.ndarray
        Array of node latitude values.
    """

    file_path = os.path.join(grid_path, "nod2d.out")
    if not os.path.isfile(file_path):
        raise FileNotFoundError("TODO: ")

    nodes = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=1,
        names=["node_number", "x", "y", "flag"],
    )

    node_lon = nodes.x.values
    node_lat = nodes.y.values

    return node_lon, node_lat


def _parse_elem2d(grid_path):
    """Parses the 'elem2d.out' file to extract face-node connectivity
    information.

    Parameters:
    -----------
    grid_path : str
        The path to the directory containing 'elem2d.out'.

    Returns:
    --------
    face_node_connectivity : np.ndarray
        Array of face-node connectivity data, with indices adjusted to be 0-based.
    """
    file_path = os.path.join(grid_path, "elem2d.out")
    if not os.path.isfile(file_path):
        raise FileNotFoundError("TODO: ")

    file_content = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=1,
        names=["first_elem", "second_elem", "third_elem"],
    )

    face_node_connectivity = file_content.values - 1

    return face_node_connectivity


def _parse_edge_tri(grid_path):
    """Parses the 'edge_tri.out' file to extract edge-face connectivity
    information.

    Parameters:
    -----------
    grid_path : str
        The path to the directory containing 'edge_tri.out'.

    Returns:
    --------
    edge_face_connectivity : np.ndarray or None
        Array of edge-face connectivity data, with missing values filled using INT_FILL_VALUE.
        Returns None if the file is not found (optional file).
    """
    file_path = os.path.join(grid_path, "edge_tri.out")
    if not os.path.isfile(file_path):
        return None
    file_content = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=0,
        names=["first_elem", "second_elem"],
    )

    edge_face_connectivity = file_content.values - 1
    edge_face_connectivity[edge_face_connectivity == -1000] = INT_FILL_VALUE

    return edge_face_connectivity


def _parse_edges(grid_path):
    """Parses the 'edges.out' file to extract edge-node connectivity
    information.

    Parameters:
    -----------
    grid_path : str
        The path to the directory containing 'edges.out'.

    Returns:
    --------
    edge_node_connectivity : np.ndarray or None
        Array of edge-node connectivity data, with indices adjusted to be 0-based.
        Returns None if the file is not found (optional file).
    """
    file_path = os.path.join(grid_path, "edges.out")
    if not os.path.isfile(file_path):
        return None
    file_content = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=0,
        names=["first_elem", "second_elem"],
    )

    edge_node_connectivity = file_content.values - 1
    return edge_node_connectivity


def _read_fesom2_netcdf(in_ds):
    """Reads a FESOM2 NetCDF grid dataset and encodes it in tbe UGRID
    conventions.

    Parameters:
    -----------
    in_ds : xr.Dataset
        Dataset containing FESOM2 grid variables

    Returns:
    --------
    ugrid_ds : xr.Dataset
        An xarray Dataset containing the FESOM2 grid information encoded in the UGRID conventions
    source_dims_dict : dict
        A dictionary mapping FESM2 dimensions to UGRID dimensions

    Raises:
    -------
    FileNotFoundError
        If required files are not found in the provided path. At least "nod2d.out" and "elem2d.out" are required to
        construct a UGRID-compliant grid.
    """
    source_dims_dict = {"ncells": "n_face"}
    ugrid_ds = xr.Dataset()

    node_dim = in_ds["lon"].dims[0]

    ugrid_ds["node_lon"] = (
        in_ds["lon"]
        .rename("node_lon")
        .assign_attrs(ugrid.NODE_LON_ATTRS)
        .swap_dims({node_dim: ugrid.NODE_DIM})
    )
    ugrid_ds["node_lat"] = (
        in_ds["lat"]
        .rename("node_lat")
        .assign_attrs(ugrid.NODE_LAT_ATTRS)
        .swap_dims({node_dim: ugrid.NODE_DIM})
    )

    face_node_connectivity = in_ds["triag_nodes"] - 1

    ugrid_ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_connectivity,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    return ugrid_ds, source_dims_dict
