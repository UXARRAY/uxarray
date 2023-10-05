from uxarray.io._ugrid import _is_ugrid


def _parse_grid_type(dataset):
    """Checks input and contents to determine grid type. Supports detection of
    UGrid, SCRIP, Exodus and shape file.

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
    # exodus with coord or coordx
    if "coord" in dataset:
        mesh_type = "Exodus"
    elif "coordx" in dataset:
        mesh_type = "Exodus"
    # scrip with grid_center_lon
    elif "grid_center_lon" in dataset:
        mesh_type = "Scrip"
    # ugrid topology
    elif _is_ugrid(dataset):
        mesh_type = "UGRID"
    elif "verticesOnCell" in dataset:
        mesh_type = "MPAS"
    else:
        raise RuntimeError(f"Could not recognize dataset format.")
    return mesh_type

    # check mesh topology and dimension
    # try:
    #     standard_name = lambda v: v is not None
    #     # getkeys_filter_by_attribute(filepath, attr_name, attr_val)
    #     # return type KeysView
    #     ext_ds = xr.open_dataset(filepath, mask_and_scale=False)
    #     node_coords_dv = ext_ds.filter_by_attrs(
    #         node_coordinates=standard_name).keys()
    #     face_conn_dv = ext_ds.filter_by_attrs(
    #         face_node_connectivity=standard_name).keys()
    #     topo_dim_dv = ext_ds.filter_by_attrs(
    #         topology_dimension=standard_name).keys()
    #     mesh_topo_dv = ext_ds.filter_by_attrs(cf_role="mesh_topology").keys()
    #     if list(mesh_topo_dv)[0] != "" and list(topo_dim_dv)[0] != "" and list(
    #             face_conn_dv)[0] != "" and list(node_coords_dv)[0] != "":
    #         mesh_type = "ugrid"
    #     else:
    #         raise ValueError(
    #             "cf_role is other than mesh_topology, the input NetCDF file is not UGRID format"
    #         )
    # except KeyError as e:
    #     msg = str(e) + ': {}'.format(filepath)
    # except (TypeError, AttributeError) as e:
    #     msg = str(e) + ': {}'.format(filepath)
    # except (RuntimeError, OSError) as e:
    #     # check if this is a shp file
    #     # we won't use xarray to load that file
    #     if file_extension == ".shp":
    #         mesh_type = "shp"
    #     else:
    #         msg = str(e) + ': {}'.format(filepath)
    # except ValueError as e:
    #     # check if this is a shp file
    #     # we won't use xarray to load that file
    #     if file_extension == ".shp":
    #         mesh_type = "shp"
    #     else:
    #         msg = str(e) + ': {}'.format(filepath)
    # finally:
    #     if msg != "":
    #         msg = "Unable to determine file type, mesh file not supported" + ': {}'.format(
    #             filepath)
    #         raise ValueError(msg)
    #
    # return mesh_type
