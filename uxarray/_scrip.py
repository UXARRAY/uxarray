import xarray as xr
import numpy as np


def _to_ugrid(in_ds, outfile):
    """If input dataset (ds) file is an unstructured SCRIP file, function will
    reassign SCRIP variables to UGRID conventions in output file (outfile).

    Parameters
    ----------
    in_ds : :class:`xarray.Dataset`
        Original scrip dataset of interest being used

    outfile : :class:`xarray.Variable`
        file to be returned by _populate_scrip_data, used as an empty placeholder file
        to store reassigned SCRIP variables in UGRID conventions
    """

    if in_ds['grid_area'].all():

        # Create Mesh2_node_x/y variables from grid_corner_lat/lon
        # Turn latitude scrip array into 1D instead of 2D
        corner_lat = in_ds['grid_corner_lat'].values
        corner_lat = corner_lat.flatten()

        # Return only the index of unique values to preserve order
        lat_idx = np.unique(corner_lat, return_index=True)[1]
        sort_lat = sorted(lat_idx)

        # Create placeholder array for the sorted unique values
        unique_lat = np.zeros(len(sort_lat))
        for i in range(len(sort_lat) - 1):
            unique_lat[i] = (corner_lat[sort_lat[i]])

        # Repeat above steps with longitude data instead
        corner_lon = in_ds['grid_corner_lon'].values
        corner_lon = corner_lon.flatten()
        lon_idx = np.unique(corner_lon, return_index=True)[1]
        sort_lon = sorted(lon_idx)

        unique_lon = np.zeros(len(sort_lon))
        for i in range(len(sort_lon) - 1):
            unique_lon[i] = (corner_lon[sort_lon[i]])

        # Create Mesh2_node_x/y from unsorted, unique grid_corner_lat/lon
        outfile['Mesh2_node_x'] = unique_lon
        outfile['Mesh2_node_y'] = unique_lat

        # Create Mesh2_face_x/y from grid_center_lat/lon
        outfile['Mesh2_face_x'] = in_ds['grid_center_lon']
        outfile['Mesh2_face_y'] = in_ds['grid_center_lat']

        # Create array using matching grid corners to create face nodes
        face_arr = []
        for i in range(len(in_ds['grid_corner_lat'] - 1)):
            x = in_ds['grid_corner_lon'][i].values
            y = in_ds['grid_corner_lat'][i].values
            face = np.hstack([x[:, np.newaxis], y[:, np.newaxis]])
            face_arr.append(face)

        face_node = np.asarray(face_arr)

        outfile['Mesh2_face_nodes'] = xr.DataArray(
            face_node, dims=['grid_size', 'grid_corners', 'lat/lon'])
    else:
        raise Exception("Structured scrip files are not yet supported")

    return outfile


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
    out_ds : :class:`xarray.DataArray`
    """
    ext_ds = xr.open_dataset(file_path, decode_times=False, engine='netcdf4')
    ds = xr.Dataset()
    try:
        # If not ugrid compliant, translates scrip to ugrid conventions
        _to_ugrid(ext_ds, ds)

    except KeyError:
        if ext_ds['Mesh2']:
            # If is ugrid compliant, returns the dataset unchanged
            ds = ext_ds
            return ds

        else:
            # If not ugrid or scrip, returns error
            raise Exception("Variables not in recognized form (SCRIP or UGRID)")

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