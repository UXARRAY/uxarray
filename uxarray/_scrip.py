import xarray as xr
import numpy as np


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
