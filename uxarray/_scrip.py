import xarray as xr
import numpy as np


def _to_ugrid(in_ds):
    """If input dataset (ds) file is an unstructured SCRIP file, function will
    reassign SCRIP variables to UGRID conventions in output file (outfile).

    Parameters
    ----------
    in_ds : :class:`xarray.Dataset`
        Original scrip dataset of interest being used

    Returns
    -------
    out_ds : :class:`xarray.Dataset`
        file to be returned by _populate_scrip_data, used as an empty placeholder file
        to store reassigned SCRIP variables in UGRID conventions
    """
    out_ds = xr.Dataset()
    if in_ds['grid_area'].all():

        # Create Mesh2_node_x/y variables from grid_corner_lat/lon
        # Turn latitude scrip array into 1D instead of 2D
        corner_lat_xr = in_ds['grid_corner_lat']
        corner_lat = corner_lat_xr.values.ravel()

        # Return only the index of unique values of latitude data
        lat_idx = np.unique(corner_lat, return_index=True)[1]
        sort_lat = sorted(lat_idx)

        # Repeat above steps with longitude data instead
        corner_lon_xr = in_ds['grid_corner_lon']
        corner_lon = corner_lon_xr.values.ravel()

        lon_idx = np.unique(corner_lon, return_index=True)[1]
        sort_lon = sorted(lon_idx)

        # Combine lists of indexes to be used between both lat and lon coords
        indexes = np.concatenate((sort_lon, sort_lat))
        sort_idx = sorted(indexes)
        unique_idx = np.unique(sort_idx)

        # Create array for the sorted unique lat/lon values
        unique_lon = corner_lon[unique_idx]
        unique_lat = corner_lat[unique_idx]

        # Create Mesh2_node_x/y from unsorted, unique grid_corner_lat/lon
        out_ds['Mesh2_node_x'] = unique_lon
        out_ds['Mesh2_node_y'] = unique_lat

        # Create Mesh2_face_x/y from grid_center_lat/lon
        out_ds['Mesh2_face_x'] = in_ds['grid_center_lon']
        out_ds['Mesh2_face_y'] = in_ds['grid_center_lat']

        # Create array using matching grid corners to create face nodes
        # find max face nodes
        max_face_nodes = 0
        for dim in in_ds.dims:
            if "grid_corners" in dim:
                if in_ds.dims[dim] > max_face_nodes:
                    max_face_nodes = in_ds.dims[dim]

        # create an empty conn array for storing all blk face_nodes_data
        conn = np.empty((len(unique_lon), max_face_nodes))

        # set the face nodes data compiled in "connect" section
        out_ds["Mesh2_face_nodes"] = xr.DataArray(
            data=(conn[:] - 1),
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


def _read_scrip(file_path):
    """Function to reassign lat/lon variables to mesh2_node variables.

    Currently, supports unstructured SCRIP grid files following traditional SCRIP
    naming practices (grid_corner_lat, grid_center_lat, etc) and SCRIP files with
    UGRID conventions.

    Unstructured grid SCRIP files will have 'grid_rank=1' and include variables
    "grid_imask" and "grid_area" in the dataset.

    More information on structured vs unstructured SCRIP files can be found here:
    https://earthsystemmodeling.org/docs/release/ESMF_6_2_0/ESMF_refdoc/node3.html

    Parameters
    ----------
    file_path : :class:`string`
        location of SCRIP dataset of interest in format:
        "path/to/file"

    Returns
    --------
    out_ds : :class:`xarray.Dataset`
    """
    ext_ds = xr.open_dataset(file_path, decode_times=False, engine='netcdf4')

    try:
        # If not ugrid compliant, translates scrip to ugrid conventions
        ds = _to_ugrid(ext_ds)

    except KeyError:
        print(
            "Variables not in recognized SCRIP form. Please refer to",
            "https://earthsystemmodeling.org/docs/release/ESMF_6_2_0/ESMF_refdoc/node3.html#SECTION03024000000000000000",
            "for more information on SCRIP Grid file formatting")

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
    return ds
