import xarray as xr


def _is_scrip(in_ds, out_ds):
    """If in file is an unstructured SCRIP file, function will reassign SCRIP
    variables to UGRID conventions in out_ds.

    Parameters
    ----------
    in_ds : :class:`xarray.Dataset`
        Original scrip dataset of interest being used

    out_ds : :class:`xarray.Variable`
        file to be returned by _populate_scrip_data, used as an empty placeholder file
        to store reassigned SCRIP variables in UGRID conventions
    """

    if in_ds['grid_area'].all(
    ):  # Presence indicates if SCRIP file is unstructured grid
        out_ds['Mesh2_node_x'] = in_ds['grid_corner_lon']
        out_ds['Mesh2_node_y'] = in_ds['grid_corner_lat']

        # Create array using matching grid corners to create face nodes
        face_arr = []
        for i in range(len(in_ds['grid_corner_lat'] - 1)):
            x = in_ds['grid_corner_lon'][i].values
            y = in_ds['grid_corner_lat'][i].values
            face = np.hstack([x[:, np.newaxis], y[:, np.newaxis]])
            face_arr.append(face)

        face_node = np.asarray(face_arr)

        out_ds['Mesh2_face_nodes'] = xr.DataArray(
            face_node, dims=['grid_size', 'grid_corners', 'lat/lon'])
    else:
        raise Exception("Structured scrip files are not yet supported")


def _populate_scrip_data(in_ds):
    """Function to reassign lat/lon variables to mesh2_node variables.

    Currently supports unstructured SCRIP grid files following traditional SCRIP
    naming practices (grid_corner_lat, grid_center_lat, etc) and SCRIP files with
    UGRID conventions.

    Unstructured grid SCRIP files will have 'grid_rank=2' and include variables
    "grid_imask" and "grid_area" in the dataset.

    More information on structured vs unstructured SCRIP files can be found here:
    https://earthsystemmodeling.org/docs/release/ESMF_6_2_0/ESMF_refdoc/node3.html

    Parameters
    ----------
    in_ds : :class:`xarray.Dataset`
        Scrip dataset of interest being used

    Returns
    --------
    out_ds : :class:`xarray.Variable`
        SCRIP file with UGRID conventions for 2D flexible mesh topology
    """
    out_ds = xr.Dataset()
    try:
        # If not ugrid compliant, translates scrip to ugrid conventions
        _is_scrip(in_ds, out_ds)

    except KeyError:
        if in_ds['Mesh2']:
            # If is ugrid compliant, returns the dataset unchanged
            try:
                out_ds = in_ds
                return out_ds
            except:
                # If not ugrid or scrip, returns error
                raise Exception(
                    "Variables not in recognized form (SCRIP or UGRID)")

    out_ds["Mesh2"] = xr.DataArray(
        attrs={
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 2D unstructured mesh",
            "topology_dimension": 2,
            "node_coordinates": "Mesh2_node_x Mesh2_node_y Mesh2_node_z",
            "node_dimension": "nMesh2_node",
            "face_node_connectivity": "Mesh2_face_nodes",
            "face_dimension": "nMesh2_face"
        })

    return out_ds


# ESMF files https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/share/meshes/
# ESMF/SCRIP info: https://earthsystemmodeling.org/docs/release/ESMF_6_2_0/ESMF_refdoc/node3.html
