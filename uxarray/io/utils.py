import numpy as np
import xarray as xr

from uxarray.io._esmf import _esmf_to_ugrid_dims
from uxarray.io._icon import _icon_to_ugrid_dims
from uxarray.io._mpas import _mpas_to_ugrid_dims
from uxarray.io._ugrid import _is_ugrid, _read_ugrid


def _is_exodus(dataset: xr.Dataset) -> bool:
    """Check whether a dataset looks like an Exodus mesh."""
    has_packed_coords = "coord" in dataset
    has_split_coords = {"coordx", "coordy"}.issubset(dataset.variables)
    has_connectivity = any(
        name.startswith("connect") for name in dataset.variables
    ) or any("num_nod_per_el" in dim for dim in dataset.dims)

    return has_connectivity and (has_packed_coords or has_split_coords)


def _is_scrip(dataset: xr.Dataset) -> bool:
    """Check whether a dataset looks like an unstructured SCRIP grid."""
    required_vars = {
        "grid_center_lon",
        "grid_center_lat",
        "grid_corner_lon",
        "grid_corner_lat",
    }
    unstructured_markers = {"grid_imask", "grid_rank", "grid_area"}

    return required_vars.issubset(dataset.variables) and any(
        marker in dataset for marker in unstructured_markers
    )


def _is_mpas(dataset: xr.Dataset) -> bool:
    """Check whether a dataset looks like an MPAS grid."""
    if "verticesOnCell" not in dataset:
        return False

    companion_groups = (
        {"nEdgesOnCell"},
        {"latCell", "lonCell"},
        {"latVertex", "lonVertex"},
        {"xCell", "yCell", "zCell"},
        {"xVertex", "yVertex", "zVertex"},
    )

    return any(group.issubset(dataset.variables) for group in companion_groups)


def _is_esmf(dataset: xr.Dataset) -> bool:
    """Check whether a dataset looks like an ESMF mesh."""
    return "maxNodePElement" in dataset.dims and "elementConn" in dataset


def _is_geos_cs(dataset: xr.Dataset) -> bool:
    """Check whether a dataset looks like a GEOS cube-sphere grid."""
    required_dims = {"nf", "YCdim", "XCdim"}
    required_vars = {"corner_lons", "corner_lats"}

    return required_dims.issubset(dataset.sizes) and required_vars.issubset(
        dataset.variables
    )


def _is_icon(dataset: xr.Dataset) -> bool:
    """Check whether a dataset looks like an ICON grid."""
    required_vars = {"vertex_of_cell", "clon", "clat", "vlon", "vlat"}
    return required_vars.issubset(dataset.variables)


def _is_fesom2(dataset: xr.Dataset) -> bool:
    """Check whether a dataset looks like a FESOM2 grid."""
    return "triag_nodes" in dataset


def _parse_grid_type(dataset):
    """Determine the grid type represented by an input dataset.

    Parameters
    ----------
    dataset : Xarray dataset
        Xarray dataset containing grid topology information.

    Returns
    -------
    tuple[str, str | None, str | None]
        A 3-tuple of ``(mesh_type, lon_name, lat_name)``. ``mesh_type`` is one
        of ``"Exodus"``, ``"Scrip"``, ``"UGRID"``, ``"MPAS"``, ``"ESMF"``,
        ``"GEOS-CS"``, ``"ICON"``, ``"FESOM2"``, or ``"Structured"``. The
        longitude and latitude coordinate names are only returned for structured
        grids and are otherwise ``None``.

    Raises
    ------
    RuntimeError
        If the dataset format cannot be recognized.
    """

    _structured, lon_name, lat_name = _is_structured(dataset)

    if _is_exodus(dataset):
        mesh_type = "Exodus"
    elif _is_scrip(dataset):
        mesh_type = "Scrip"
    elif _is_ugrid(dataset):
        mesh_type = "UGRID"
    elif _is_mpas(dataset):
        mesh_type = "MPAS"
    elif _is_esmf(dataset):
        mesh_type = "ESMF"
    elif _is_geos_cs(dataset):
        mesh_type = "GEOS-CS"
    elif _is_icon(dataset):
        mesh_type = "ICON"
    elif _is_fesom2(dataset):
        mesh_type = "FESOM2"
    elif _structured:
        mesh_type = "Structured"
        return mesh_type, lon_name, lat_name
    else:
        raise RuntimeError("Failed to parse uxgrid information from xarray.Dataset.")

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
