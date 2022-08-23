import xarray as xr
import numpy as np
import uxarray as ux
from .helpers import grid_center_lat_lon


def _to_ugrid(in_ds, out_ds):
    """If input dataset (ds) file is an unstructured SCRIP file, function will
    reassign SCRIP variables to UGRID conventions in output file (outfile).

    Parameters
    ----------
    in_ds : :class:`xarray.Dataset`
        Original scrip dataset of interest being used

    out_ds : :class:`xarray.Variable`
        file to be returned by _populate_scrip_data, used as an empty placeholder file
        to store reassigned SCRIP variables in UGRID conventions
    """

    if in_ds['grid_area'].all():

        # Create Mesh2_node_x/y variables from grid_corner_lat/lon
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

        # Now, calculate unique lon and lat values to account for 'Mesh2_node_x' and 'Mesh2_node_y'
        unq_lon = corner_lon_lat[unq_ind, :][:, 0]
        unq_lat = corner_lon_lat[unq_ind, :][:, 1]

        # Reshape face nodes array into original shape for use in 'Mesh2_face_nodes'
        unq_inv = np.reshape(unq_inv,
                             (len(in_ds.grid_size), len(in_ds.grid_corners)))

        # Create Mesh2_node_x/y from unsorted, unique grid_corner_lat/lon
        out_ds['Mesh2_node_x'] = unq_lon
        out_ds['Mesh2_node_y'] = unq_lat

        # Create Mesh2_face_x/y from grid_center_lat/lon
        out_ds['Mesh2_face_x'] = in_ds['grid_center_lon']
        out_ds['Mesh2_face_y'] = in_ds['grid_center_lat']

        # set the face nodes data compiled in "connect" section
        out_ds["Mesh2_face_nodes"] = xr.DataArray(
            data=unq_inv,
            dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
            attrs={
                "cf_role":
                    "face_node_connectivity",
                "_FillValue":
                    -1,
                "start_index":
                    np.int32(
                        0
                    )  # NOTE: This might cause an error if numbering has holes
            })

    else:
        raise Exception("Structured scrip files are not yet supported")

    return out_ds


def _read_scrip(ext_ds):
    """Function to reassign lat/lon variables to mesh2_node variables.

    Currently, supports unstructured SCRIP grid files following traditional SCRIP
    naming practices (grid_corner_lat, grid_center_lat, etc) and SCRIP files with
    UGRID conventions.

    Unstructured grid SCRIP files will have 'grid_rank=1' and include variables
    "grid_imask" and "grid_area" in the dataset.

    More information on structured vs unstructured SCRIP files can be found here:
    https://earthsystemmodeling.org/docs/release/ESMF_6_2_0/ESMF_refdoc/node3.html

    Parameters: xarray.Dataset, required
    Returns: ugrid aware xarray.Dataset
    """
    ds = xr.Dataset()

    try:
        # If not ugrid compliant, translates scrip to ugrid conventions
        _to_ugrid(ext_ds, ds)

        # Add necessary UGRID attributes to new dataset
        ds["Mesh2"] = xr.DataArray(
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Topology data of 2D unstructured mesh",
                "topology_dimension": 2,
                "node_coordinates": "Mesh2_node_x Mesh2_node_y Mesh2_node_z",
                "node_dimension": "nMesh2_node",
                "face_node_connectivity": "Mesh2_face_nodes",
                "face_dimension": "nMesh2_face"
            })

    except:
        print(
            "Variables not in recognized SCRIP form. Please refer to",
            "https://earthsystemmodeling.org/docs/release/ESMF_6_2_0/ESMF_refdoc/node3.html#SECTION03024000000000000000",
            "for more information on SCRIP Grid file formatting")

    return ds


def _write_scrip(ext_ds, outfile):
    """Function to reassign UGRID formatted variables to SCRIP formatted
    variables.

    Currently, supports creating unstructured SCRIP grid files following traditional
    SCRIP naming practices (grid_corner_lat, grid_center_lat, etc).

    Unstructured grid SCRIP files will have 'grid_rank=1' and include variables
    "grid_imask" and "grid_area" in the dataset.

    More information on structured vs unstructured SCRIP files can be found here:
    https://earthsystemmodeling.org/docs/release/ESMF_6_2_0/ESMF_refdoc/node3.html

    Parameters
    ----------
    ext_ds : :class:`xarray.Dataset`
        Original UGRID dataset of interest being used

    outfile : :class:`string`
        Name of file to be created. Saved to working directory, or to specified location if full path
        to new file is provided.

    Returns
    -------
    ds : :class:`xarray.Dataset`
        File to be returned by _write_scrip. Saved both as an independent file and as an active
        dataset for immediate use.
    """
    # Create empty dataset to put new scrip format data into
    ds = xr.Dataset()

    # Create grid instance of input ugrid file for later use
    grid = ux.open_dataset(ext_ds)

    # Use Xarray to open dataset
    ext_ds = xr.open_dataset(ext_ds, decode_times=False, engine='netcdf4')

    # Make grid corner lat/lon
    f_nodes = ext_ds['Mesh2_face_nodes'].values.ravel()

    # Extract lat/lon node data
    y_val = ext_ds['Mesh2_node_y']
    x_val = ext_ds['Mesh2_node_x']

    # Create empty arrays to hold lat/lon data
    lat_nodes = np.zeros_like(f_nodes)
    lon_nodes = np.zeros_like(f_nodes)

    for i in range(len(f_nodes)):
        lat_nodes[i] = y_val[int(f_nodes[i])]
        lon_nodes[i] = x_val[int(f_nodes[i])]

    # Reshape arrays to be 2D instead of 1D
    reshp_lat = np.reshape(lat_nodes, [ext_ds['Mesh2_face_nodes'].shape[0], 4])
    reshp_lon = np.reshape(lon_nodes, [ext_ds['Mesh2_face_nodes'].shape[0], 4])

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

    # Create grid_area using Grid class functions
    f_area = grid.compute_face_areas(quadrature_rule='gaussian')

    ds["grid_area"] = xr.DataArray(data=f_area, dims=["grid_size"])

    # Calculate and create grid center lat/lon using helper function
    center_lat, center_lon = grid_center_lat_lon(ds)

    ds['grid_center_lon'] = xr.DataArray(data=center_lon, dims=["grid_size"])

    ds['grid_center_lat'] = xr.DataArray(data=center_lat, dims=["grid_size"])

    # Create and save new scrip file
    ds.to_netcdf(outfile)

    return ds
