import numpy as np
import xarray as xr

from uxarray.io._esmf import _esmf_to_ugrid_dims
from uxarray.io._icon import _icon_to_ugrid_dims
from uxarray.io._mpas import _mpas_to_ugrid_dims
from uxarray.io._ugrid import _is_ugrid, _read_ugrid


def _parse_grid_type(dataset):
    """Checks input and contents to determine grid type. Supports detection of
    UGrid, SCRIP, Exodus, ESMF, and shape file.

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

    _structured, lon_name, lat_name = _is_structured(dataset)

    if "coord" in dataset:
        # exodus with coord or coordx
        mesh_type = "Exodus"
    elif "coordx" in dataset:
        mesh_type = "Exodus"
    elif "grid_center_lon" in dataset:
        # scrip with grid_center_lon
        mesh_type = "Scrip"
    elif _is_ugrid(dataset):
        # ugrid topology is present
        mesh_type = "UGRID"
    elif "verticesOnCell" in dataset:
        mesh_type = "MPAS"
    elif "maxNodePElement" in dataset.dims:
        mesh_type = "ESMF"
    elif all(key in dataset.sizes for key in ["nf", "YCdim", "XCdim"]):
        # expected dimensions for a GEOS cube sphere grid
        mesh_type = "GEOS-CS"
    elif "vertex_of_cell" in dataset:
        mesh_type = "ICON"
    elif "triag_nodes" in dataset:
        mesh_type = "FESOM2"
    elif _structured:
        mesh_type = "Structured"
        return mesh_type, lon_name, lat_name
    else:
        raise RuntimeError("Could not recognize dataset format.")

    return mesh_type, None, None


def _is_structured(dataset: xr.Dataset, tol: float = 1e-5) -> bool:
    """
    Determine if the dataset is structured based on the presence of
    coordinates with 'standard_name' attributes set to 'latitude' and 'longitude',
    and verify that the spacing between latitude and longitude cells is regular.

    Parameters:
    -----------
    dataset : xr.Dataset
        The xarray Dataset to check.
    tol : float, optional
        Tolerance for checking regular spacing. Default is 1e-5.

    Returns:
    --------
    bool
        True if the dataset is structured with regularly spaced latitude and longitude,
        False otherwise.
    """
    # Extract all 'standard_name' attributes in lower case
    standard_names = [
        coord.attrs.get("standard_name", "").lower()
        for coord in dataset.coords.values()
    ]

    # Check for presence of 'latitude' and 'longitude'
    has_latitude = "latitude" in standard_names
    has_longitude = "longitude" in standard_names

    if not (has_latitude and has_longitude):
        return False, None, None

    # Identify the names of latitude and longitude coordinates
    lat_name = None
    lon_name = None
    for coord_name, coord in dataset.coords.items():
        standard_name = coord.attrs.get("standard_name", "").lower()
        if standard_name == "latitude":
            lat_name = coord_name
        elif standard_name == "longitude":
            lon_name = coord_name

    # Extract latitude and longitude values
    lat = dataset.coords[lat_name].data
    lon = dataset.coords[lon_name].data

    # Ensure that latitude and longitude are one-dimensional
    if lat.ndim != 1 or lon.ndim != 1:
        print("Latitude and/or longitude coordinates are not one-dimensional.")
        return False, None, None

    # Calculate the differences between consecutive latitude and longitude values
    lat_diffs = np.diff(lat)
    lon_diffs = np.diff(lon)

    # Check if the differences are approximately constant within the tolerance
    lat_regular = np.all(np.abs(lat_diffs - lat_diffs[0]) <= tol)
    lon_regular = np.all(np.abs(lon_diffs - lon_diffs[0]) <= tol)

    if not lat_regular:
        print("Latitude coordinates are not regularly spaced.")
    if not lon_regular:
        print("Longitude coordinates are not regularly spaced.")

    return lat_regular and lon_regular, lon_name, lat_name


def _get_source_dims_dict(grid_ds, grid_spec):
    if grid_spec == "MPAS":
        return _mpas_to_ugrid_dims(grid_ds)
    elif grid_spec == "UGRID":
        _, dim_dict = _read_ugrid(grid_ds)
        return dim_dict
    elif grid_spec == "ICON":
        return _icon_to_ugrid_dims(grid_ds)
    elif grid_spec == "ESMF":
        return _esmf_to_ugrid_dims(grid_ds)

    else:
        return dict()
