from typing import Any, Sequence

import numpy as np
import polars as pl
import xarray as xr

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.conventions import ugrid
from uxarray.grid.connectivity import _replace_fill_values


def _to_ugrid(in_ds, out_ds):
    """If input dataset (``in_ds``) file is an unstructured SCRIP file,
    function will reassign SCRIP variables to UGRID conventions in output file
    (``out_ds``).
    """

    source_dims_dict = {}

    # Check if any of imask, area or rank are present
    if any(key in in_ds for key in ["grid_imask", "grid_rank", "grid_area"]):
        # Create node_lon & node_lat variables from grid_corner_lat/lon
        # Turn latitude and longitude scrip arrays into 1D
        corner_lat = in_ds["grid_corner_lat"].values.ravel()
        corner_lon = in_ds["grid_corner_lon"].values.ravel()

        # Use Polars to find unique coordinate pairs
        df = pl.DataFrame({"lon": corner_lon, "lat": corner_lat}).with_row_count(
            "original_index"
        )

        # Get unique rows (first occurrence). This preserves the order in which they appear.
        unique_df = df.unique(subset=["lon", "lat"], keep="first")

        # unq_ind: The indices of the unique rows in the original array
        unq_ind = unique_df["original_index"].to_numpy().astype(INT_DTYPE)

        # To get the inverse index (unq_inv): map each original row back to its unique row index.
        # Add a unique_id to the unique_df which will serve as the "inverse" mapping.
        unique_df = unique_df.with_row_count("unique_id")

        # Join original df with unique_df to find out which unique_id corresponds to each row
        df_joined = df.join(
            unique_df.drop("original_index"), on=["lon", "lat"], how="left"
        )
        unq_inv = df_joined["unique_id"].to_numpy().astype(INT_DTYPE)

        # Extract unique lon and lat values using unq_ind
        unq_lon = corner_lon[unq_ind]
        unq_lat = corner_lat[unq_ind]

        # Reshape face nodes array into original shape for use in 'face_node_connectivity'
        unq_inv = np.reshape(unq_inv, (len(in_ds.grid_size), len(in_ds.grid_corners)))

        # Create node_lon & node_lat
        out_ds[ugrid.NODE_COORDINATES[0]] = xr.DataArray(
            unq_lon, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LON_ATTRS
        )

        out_ds[ugrid.NODE_COORDINATES[1]] = xr.DataArray(
            unq_lat, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LAT_ATTRS
        )

        # Create face_lon & face_lat from grid_center_lat/lon
        out_ds[ugrid.FACE_COORDINATES[0]] = xr.DataArray(
            in_ds["grid_center_lon"].values,
            dims=[ugrid.FACE_DIM],
            attrs=ugrid.FACE_LON_ATTRS,
        )

        out_ds[ugrid.FACE_COORDINATES[1]] = xr.DataArray(
            in_ds["grid_center_lat"].values,
            dims=[ugrid.FACE_DIM],
            attrs=ugrid.FACE_LAT_ATTRS,
        )

        # standardize fill values and data type face nodes
        face_nodes = _replace_fill_values(
            xr.DataArray(data=unq_inv),
            original_fill=-1,
            new_fill=INT_FILL_VALUE,
            new_dtype=INT_DTYPE,
        )

        # set the face nodes data compiled in "connect" section
        out_ds["face_node_connectivity"] = xr.DataArray(
            data=face_nodes.data,
            dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
            attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
        )

    else:
        raise Exception("Structured scrip files are not yet supported")

    # populate source dims
    source_dims_dict[in_ds["grid_center_lon"].dims[0]] = "n_face"

    return source_dims_dict


def _read_scrip(ext_ds):
    """Function to reassign lat/lon variables to node variables.

    Currently, supports unstructured SCRIP grid files following traditional SCRIP
    naming practices (grid_corner_lat, grid_center_lat, etc) and SCRIP files with
    UGRID conventions.

    Unstructured grid SCRIP files will have ``grid_rank=1`` and include variables
    ``grid_imask`` and ``grid_area`` in the dataset.

    More information on structured vs unstructured SCRIP files can be found here on the `Earth System Modeling Framework <https://earthsystemmodeling.org/docs/release/ESMF_6_2_0/ESMF_refdoc/node3.html>`_ website.

    Parameters
    ----------
    ext_ds : xarray.Dataset, required
        SCRIP datafile of interest

    Returns
    -------
    ds : xarray.Dataset
        ugrid aware :class:`xarray.Dataset`
    """
    ds = xr.Dataset()

    source_dims_dict = _to_ugrid(ext_ds, ds)

    return ds, source_dims_dict


def _encode_scrip(face_node_connectivity, node_lon, node_lat, face_areas):
    """Function to reassign UGRID formatted variables to SCRIP formatted
    variables.

    Currently, supports creating unstructured SCRIP grid files following traditional
    SCRIP naming practices (grid_corner_lat, grid_center_lat, etc).

    Unstructured grid SCRIP files typically have ``grid_rank=1`` and include variables
    ``grid_imask`` and ``grid_area`` in the dataset.

    More information on structured vs unstructured SCRIP files can be found here on the `Earth System Modeling Framework <https://earthsystemmodeling.org/docs/release/ESMF_6_2_0/ESMF_refdoc/node3.html>`_ website.

    Parameters
    ----------
    outfile : str
        Name of file to be created. Saved to working directory, or to
        specified location if full path to new file is provided.

    face_node_connectivity : xarray.DataArray
        Face-node connectivity. This variable should come from the ``Grid``
        object that calls this function

    node_lon : xarray.DataArray
        Nodes' x values. This variable should come from the ``Grid`` object
        that calls this function

    node_lat : xarray.DataArray
        Nodes' y values. This variable should come from the ``Grid`` object
        that calls this function

    face_areas : numpy.ndarray
        Face areas. This variable should come from the ``Grid`` object
        that calls this function

    Returns
    -------
    ds : xarray.Dataset
        Dataset to be returned by ``_encode_scrip``. The function returns
        the output dataset in SCRIP format for immediate use.
    """
    # Create empty dataset to put new scrip format data into
    ds = xr.Dataset()
    n_face = face_node_connectivity.shape[0]
    n_max_nodes = face_node_connectivity.shape[1]

    # --- Core logic enhanced with Implementation 2's robust method ---
    # Flatten the connectivity array to easily work with all node indices
    f_nodes_flat = face_node_connectivity.values.astype(int).ravel()

    # Create a mask to identify valid nodes vs. fill values
    valid_nodes_mask = f_nodes_flat != INT_FILL_VALUE

    # Create arrays to hold final lat/lon data, filled with NaN
    lat_nodes_flat = np.full(f_nodes_flat.shape, np.nan, dtype=np.float64)
    lon_nodes_flat = np.full(f_nodes_flat.shape, np.nan, dtype=np.float64)

    # Get the flattened indices of the valid nodes (where the mask is True)
    valid_indices = np.where(valid_nodes_mask)[0]
    # Get the actual node indices from the connectivity array for those valid positions
    valid_node_ids = f_nodes_flat[valid_indices]

    # Use the valid indices to populate the coordinate arrays correctly
    lon_nodes_flat[valid_indices] = node_lon.values[valid_node_ids]
    lat_nodes_flat[valid_indices] = node_lat.values[valid_node_ids]

    # Reshape the 1D arrays back to 2D
    reshp_lat = lat_nodes_flat.reshape((n_face, n_max_nodes))
    reshp_lon = lon_nodes_flat.reshape((n_face, n_max_nodes))
    # --- End of enhanced logic ---

    # Add data to new scrip output file
    ds["grid_corner_lat"] = xr.DataArray(
        data=reshp_lat,
        dims=["grid_size", "grid_corners"],
        attrs={"units": "degrees"},
    )
    ds["grid_corner_lon"] = xr.DataArray(
        data=reshp_lon,
        dims=["grid_size", "grid_corners"],
        attrs={"units": "degrees"},
    )

    # Create Grid rank, always 1 for unstructured grids
    ds["grid_rank"] = xr.DataArray(data=[1], dims=["grid_rank"])

    # FIX: Correctly set grid_dims to the number of faces
    ds["grid_dims"] = xr.DataArray(data=[n_face], dims=["grid_rank"])

    # Create grid_imask representing active cells
    ds["grid_imask"] = xr.DataArray(data=np.ones(n_face, dtype=int), dims=["grid_size"])

    ds["grid_area"] = xr.DataArray(data=face_areas, dims=["grid_size"])

    # Calculate and create grid center lat/lon using helper function
    center_lat, center_lon = grid_center_lat_lon(ds)

    ds["grid_center_lon"] = xr.DataArray(data=center_lon, dims=["grid_size"])
    ds["grid_center_lat"] = xr.DataArray(data=center_lat, dims=["grid_size"])

    return ds


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
    center_lat : array-like
        The calculated center latitudes of the grid box based on the corner
        points. Preserves chunking when inputs are backed by Dask arrays.
    center_lon : array-like
        The calculated center longitudes of the grid box based on the corner
        points. Preserves chunking when inputs are backed by Dask arrays.
    """

    # Calculate and create grid center lat/lon
    scrip_corner_lon = ds["grid_corner_lon"]
    scrip_corner_lat = ds["grid_corner_lat"]

    # convert to radians
    rad_corner_lon = np.deg2rad(scrip_corner_lon)
    rad_corner_lat = np.deg2rad(scrip_corner_lat)

    # get nodes per face
    nodes_per_face = rad_corner_lat.shape[1]

    # geographic center of each cell
    x = np.sum(np.cos(rad_corner_lat) * np.cos(rad_corner_lon), axis=1) / nodes_per_face
    y = np.sum(np.cos(rad_corner_lat) * np.sin(rad_corner_lon), axis=1) / nodes_per_face
    z = np.sum(np.sin(rad_corner_lat), axis=1) / nodes_per_face

    center_lon = np.rad2deg(np.arctan2(y, x))
    center_lat = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))

    # Normalize negative longitudes without forcing eager computation
    center_lon = center_lon.where(center_lon >= 0, center_lon + 360)

    return center_lat.data, center_lon.data


def _detect_multigrid(ds: xr.Dataset) -> tuple[str, dict[str, dict[str, Any]]]:
    """Detect whether a dataset follows single-grid or multi-grid SCRIP format.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to inspect.

    Returns
    -------
    tuple
        A tuple of (format_type, grids_dict). ``format_type`` is either
        ``\"single_scrip\"`` or ``\"multi_scrip\"``. ``grids_dict`` maps grid
        names to their variable metadata when multi-grid files are detected.
    """

    # Quick exit for canonical single-grid SCRIP files
    if {"grid_corner_lat", "grid_corner_lon"}.issubset(set(ds.variables)):
        return "single_scrip", {"grid": {}}

    # Collect candidate grids from dimension names
    grids: dict[str, dict[str, Any]] = {}
    for dim_name in ds.dims:
        if dim_name.startswith("nc_"):
            info = grids.setdefault(dim_name[3:], {})
            info.setdefault("cell_dims", []).append(dim_name)
            info.setdefault("cell_dim", dim_name)
        elif dim_name.startswith("nv_"):
            grids.setdefault(dim_name[3:], {})["corner_dim"] = dim_name

    def _infer_corner_dim_from_dims(dims: Sequence[str]) -> str | None:
        for dim in dims:
            dim_lower = dim.lower()
            if dim_lower.startswith(("nv", "nvertex", "corner", "corn", "crn")):
                return dim
        return dims[-1] if dims else None

    def _update_grid_dim_metadata(info: dict[str, Any], dims: Sequence[str]) -> None:
        if not dims:
            return

        corner_dim = info.get("corner_dim")
        inferred_corner_dim = _infer_corner_dim_from_dims(dims)
        if corner_dim is None or corner_dim not in dims:
            corner_dim = inferred_corner_dim
        if corner_dim is not None:
            info["corner_dim"] = corner_dim

        cell_dims = [dim for dim in dims if dim != corner_dim]
        if not cell_dims:
            cell_dims = list(dims)

        info["cell_dims"] = list(cell_dims)
        info["cell_dim"] = cell_dims[0]

    corner_lat_suffixes = {"cla", "corner_lat", "cornlat"}
    corner_lon_suffixes = {"clo", "corner_lon", "cornlon"}
    center_lat_suffixes = {"center_lat", "cenlat", "gclat", "clat"}
    center_lon_suffixes = {"center_lon", "cenlon", "gclon", "clon"}

    # Parse OASIS-style <grid>.<var> names
    for var_name in ds.data_vars:
        if "." not in var_name:
            continue
        grid_name, suffix = var_name.split(".", 1)
        info = grids.setdefault(grid_name, {})
        suffix_lower = suffix.lower()

        if suffix_lower in corner_lat_suffixes:
            info["corner_lat"] = var_name
            dims = ds[var_name].dims
            if len(dims) >= 2:
                _update_grid_dim_metadata(info, dims)
        elif suffix_lower in corner_lon_suffixes:
            info["corner_lon"] = var_name
            dims = ds[var_name].dims
            if len(dims) >= 2:
                _update_grid_dim_metadata(info, dims)
        elif suffix_lower in center_lat_suffixes:
            info["center_lat"] = var_name
        elif suffix_lower in center_lon_suffixes:
            info["center_lon"] = var_name

    # Keep only grids that have the required corner variables
    parsed_grids = {
        name: meta
        for name, meta in grids.items()
        if "corner_lat" in meta and "corner_lon" in meta
    }

    if parsed_grids:
        return "multi_scrip", parsed_grids

    return "single_scrip", {}


def _resolve_cell_dims(
    metadata: dict[str, Any],
    data_dims: Sequence[str],
    corner_dim: str | None = None,
) -> list[str]:
    """Determine which dimensions describe cells for a grid variable."""

    dims_from_meta = metadata.get("cell_dims")
    cell_dims: list[str] = []
    if isinstance(dims_from_meta, (list, tuple)):
        cell_dims = [dim for dim in dims_from_meta if dim in data_dims]
    elif "cell_dim" in metadata and metadata["cell_dim"] in data_dims:
        cell_dims = [metadata["cell_dim"]]
    if not cell_dims:
        cell_dims = [dim for dim in data_dims]

    if corner_dim is not None:
        cell_dims = [dim for dim in cell_dims if dim != corner_dim]

    if not cell_dims:
        cell_dims = [dim for dim in data_dims if dim != corner_dim]

    if not cell_dims:
        raise ValueError("Unable to determine cell dimensions for grid variable.")

    return cell_dims


def _stack_cell_dims(
    data_array: xr.DataArray, cell_dims: Sequence[str], new_dim: str
) -> xr.DataArray:
    """Stack one or more cell dimensions into a single new dimension."""

    dims_in_array = [dim for dim in cell_dims if dim in data_array.dims]
    if not dims_in_array:
        if new_dim in data_array.dims:
            return data_array
        raise ValueError(
            f"Unable to stack dimensions {cell_dims}; none are present in {data_array.dims}"
        )

    if len(dims_in_array) == 1:
        dim = dims_in_array[0]
        if dim == new_dim:
            return data_array
        return data_array.rename({dim: new_dim})

    stacked = data_array.stack({new_dim: dims_in_array})
    # Remove MultiIndex so grid_size behaves like a standard dimension
    stacked = stacked.reset_index(new_dim, drop=True)
    # Ensure the new dimension is the leading axis for consistency
    remaining_dims = [dim for dim in stacked.dims if dim != new_dim]
    return stacked.transpose(new_dim, *remaining_dims)


def _extract_single_grid(
    ds: xr.Dataset, grid_name: str, metadata: dict[str, Any]
) -> xr.Dataset:
    """Extract a single grid from a multi-grid SCRIP dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The source multi-grid dataset.
    grid_name : str
        Name of the grid to extract.
    metadata : dict
        Mapping that describes the variable and dimension names for the grid.

    Returns
    -------
    xr.Dataset
        Dataset encoded in standard SCRIP single-grid format.
    """

    if "corner_lat" not in metadata or "corner_lon" not in metadata:
        raise ValueError(f"Grid '{grid_name}' is missing corner variables.")

    corner_lat = ds[metadata["corner_lat"]]
    corner_lon = ds[metadata["corner_lon"]]

    dims = list(corner_lat.dims)
    if len(dims) < 2:
        raise ValueError(f"Corner variable for grid '{grid_name}' must be at least 2D.")

    corner_dim = metadata.get("corner_dim", dims[-1])
    cell_dims = _resolve_cell_dims(metadata, dims, corner_dim)

    grid_corner_lat = _stack_cell_dims(corner_lat, cell_dims, "grid_size")
    grid_corner_lon = _stack_cell_dims(corner_lon, cell_dims, "grid_size")

    if corner_dim != "grid_corners":
        grid_corner_lat = grid_corner_lat.rename({corner_dim: "grid_corners"})
        grid_corner_lon = grid_corner_lon.rename({corner_dim: "grid_corners"})

    grid_corner_lat = grid_corner_lat.copy()
    grid_corner_lon = grid_corner_lon.copy()

    result = xr.Dataset()
    result["grid_corner_lat"] = grid_corner_lat
    result["grid_corner_lon"] = grid_corner_lon

    n_cells = grid_corner_lat.sizes["grid_size"]

    # Center coordinates: use supplied variables if available, otherwise compute
    center_lat = metadata.get("center_lat")
    center_lon = metadata.get("center_lon")

    computed_lat_lon = None

    if center_lat and center_lat in ds:
        center_lat_da = _stack_cell_dims(
            ds[center_lat],
            _resolve_cell_dims(metadata, ds[center_lat].dims),
            "grid_size",
        ).copy()
    else:
        if computed_lat_lon is None:
            computed_lat_lon = grid_center_lat_lon(result)
        center_lat_da = xr.DataArray(computed_lat_lon[0], dims=["grid_size"])

    if center_lon and center_lon in ds:
        center_lon_da = _stack_cell_dims(
            ds[center_lon],
            _resolve_cell_dims(metadata, ds[center_lon].dims),
            "grid_size",
        ).copy()
    else:
        if computed_lat_lon is None:
            computed_lat_lon = grid_center_lat_lon(result)
        center_lon_da = xr.DataArray(computed_lat_lon[1], dims=["grid_size"])

    result["grid_center_lat"] = center_lat_da
    result["grid_center_lon"] = center_lon_da

    # Provide minimal auxiliary variables required by the reader
    result["grid_imask"] = xr.DataArray(
        np.ones(n_cells, dtype=np.int32), dims=["grid_size"]
    )
    result["grid_dims"] = xr.DataArray(
        np.array([n_cells], dtype=np.int32), dims=["grid_rank"]
    )

    # Provide a placeholder grid area to satisfy downstream checks
    result["grid_area"] = xr.DataArray(
        np.ones(n_cells, dtype=np.float64), dims=["grid_size"]
    )

    result.attrs.update(ds.attrs)
    result.attrs["grid_name"] = grid_name

    return result
