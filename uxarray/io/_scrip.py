import xarray as xr
import numpy as np

from uxarray.grid.connectivity import _replace_fill_values
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


def _to_ugrid(in_ds, out_ds):
    """If input dataset (``in_ds``) file is an unstructured SCRIP file,
    function will reassign SCRIP variables to UGRID conventions in output file
    (``out_ds``).

    Parameters
    ----------
    in_ds : xarray.Dataset
        Original scrip dataset of interest being used

    out_ds : xarray.Variable
        file to be returned by ``_populate_scrip_data``, used as an empty placeholder file
        to store reassigned SCRIP variables in UGRID conventions
    """

    source_dims_dict = {}

    if in_ds['grid_area'].all():

        # Create node_lon & node_lat variables from grid_corner_lat/lon
        # Turn latitude scrip array into 1D instead of 2D
        corner_lat = in_ds['grid_corner_lat'].values.ravel()

        # Repeat above steps with longitude data instead
        corner_lon = in_ds['grid_corner_lon'].values.ravel()

        # Combine flat lat and lon arrays
        corner_lon_lat = np.vstack((corner_lon, corner_lat)).T

        # Run numpy unique to determine which rows/values are actually unique
        _, unq_ind, unq_inv = np.unique(corner_lon_lat,
                                        return_index=True,
                                        return_inverse=True,
                                        axis=0)

        # Now, calculate unique lon and lat values to account for 'node_lon' and 'node_lat'
        unq_lon = corner_lon_lat[unq_ind, :][:, 0]
        unq_lat = corner_lon_lat[unq_ind, :][:, 1]

        # Reshape face nodes array into original shape for use in 'face_node_connectivity'
        unq_inv = np.reshape(unq_inv,
                             (len(in_ds.grid_size), len(in_ds.grid_corners)))

        # Create node_lon & node_lat from unsorted, unique grid_corner_lat/lon
        out_ds['node_lon'] = xr.DataArray(
            unq_lon,
            dims=["n_node"],
            attrs={
                "standard_name": "longitude",
                "long_name": "longitude of mesh nodes",
                "units": "degrees_east",
            })

        out_ds['node_lat'] = xr.DataArray(
            unq_lat,
            dims=["n_node"],
            attrs={
                "standard_name": "latitude",
                "long_name": "latitude of mesh nodes",
                "units": "degrees_north",
            })

        # Create face_lon & face_lat from grid_center_lat/lon
        out_ds['face_lon'] = in_ds['grid_center_lon']
        out_ds['face_lat'] = in_ds['grid_center_lat']

        # standardize fill values and data type face nodes
        face_nodes = _replace_fill_values(unq_inv,
                                          original_fill=-1,
                                          new_fill=INT_FILL_VALUE,
                                          new_dtype=INT_DTYPE)

        # set the face nodes data compiled in "connect" section
        out_ds["face_node_connectivity"] = xr.DataArray(
            data=face_nodes,
            dims=["n_face", "n_max_face_nodes"],
            attrs={
                "cf_role":
                    "face_node_connectivity",
                "_FillValue":
                    INT_FILL_VALUE,
                "start_index":
                    INT_DTYPE(
                        0
                    )  # NOTE: This might cause an error if numbering has holes
            })

    else:
        raise Exception("Structured scrip files are not yet supported")

    # populate source dims
    source_dims_dict[in_ds['grid_center_lon'].dims[0]] = "n_face"

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

    try:
        # If not ugrid compliant, translates scrip to ugrid conventions
        source_dims_dict = _to_ugrid(ext_ds, ds)
    except:
        print(
            "Variables not in recognized SCRIP form. Please refer to",
            "https://earthsystemmodeling.org/docs/release/ESMF_6_2_0/ESMF_refdoc/node3.html#SECTION03024000000000000000",
            "for more information on SCRIP Grid file formatting")

    return ds, source_dims_dict


def _encode_scrip(face_node_connectivity, node_lon, node_lat, face_areas):
    """Function to reassign UGRID formatted variables to SCRIP formatted
    variables.

    Currently, supports creating unstructured SCRIP grid files following traditional
    SCRIP naming practices (grid_corner_lat, grid_center_lat, etc).

    Unstructured grid SCRIP files will have ``grid_rank=1`` and include variables
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

    # Make grid corner lat/lon
    f_nodes = face_node_connectivity.values.astype(INT_DTYPE).ravel()

    # Create arrays to hold lat/lon data
    lat_nodes = node_lat[f_nodes].values
    lon_nodes = node_lon[f_nodes].values

    # Reshape arrays to be 2D instead of 1D
    reshp_lat = np.reshape(
        lat_nodes,
        [face_node_connectivity.shape[0], face_node_connectivity.shape[1]])
    reshp_lon = np.reshape(
        lon_nodes,
        [face_node_connectivity.shape[0], face_node_connectivity.shape[1]])

    # Add data to new scrip output file
    ds['grid_corner_lat'] = xr.DataArray(data=reshp_lat,
                                         dims=["grid_size", 'grid_corners'])

    ds['grid_corner_lon'] = xr.DataArray(data=reshp_lon,
                                         dims=["grid_size", 'grid_corners'])

    # Create Grid rank, always 1 for unstructured grids
    ds["grid_rank"] = xr.DataArray(data=[1], dims=["grid_rank"])

    # Create grid_dims value of len grid_size
    ds["grid_dims"] = xr.DataArray(data=[len(lon_nodes)], dims=["grid_rank"])

    # Create grid_imask representing fill values
    ds["grid_imask"] = xr.DataArray(data=np.ones(len(reshp_lon), dtype=int),
                                    dims=["grid_size"])

    ds["grid_area"] = xr.DataArray(data=face_areas, dims=["grid_size"])

    # Calculate and create grid center lat/lon using helper function
    center_lat, center_lon = grid_center_lat_lon(ds)

    ds['grid_center_lon'] = xr.DataArray(data=center_lon, dims=["grid_size"])

    ds['grid_center_lat'] = xr.DataArray(data=center_lat, dims=["grid_size"])

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
