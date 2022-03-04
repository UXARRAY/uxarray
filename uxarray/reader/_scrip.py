# Create list of names that can be used for node coords
_X_NODES = ("lat", "latitude", "Lat", "Latitude", "X", "x", "grid_corner_lat",
            "grid_corner_latitude")

_Y_NODES = ("lon", "longitude", "Lon", "Longitude", "Y", "y", "grid_corner_lon",
            "grid_corner_longitude")


def viewkeys(d):
    """Return either the keys or viewkeys method for a dictionary.

    Args:
        d (:obj:`dict`): A dictionary.
    Returns:
        view method: Either the keys or viewkeys method.
    """

    func = getattr(d, "viewkeys", None)
    if func is None:
        func = d.keys
    return func()


def _get_nodes(ds):
    """Outter wrapper function for datasets that are not cf compliant.

    :param ds: xarray.Dataset
    :return: xarray.Dataset with cf compliant naming of lat/lon variables
    """

    NODE_X = None
    NODE_Y = None

    for name in _X_NODES:
        if name in viewkeys(ds):
            ds['Mesh2_node_x'] = ds[name]
            NODE_X = ds['Mesh2_node_x']

            break

    for name in _Y_NODES:
        if name in viewkeys(ds):
            ds['Mesh2_node_y'] = ds[name]
            NODE_Y = ds['Mesh2_node_y']

            break

    return NODE_X, NODE_Y


def populate_scrip_data(ds, is_cf=True):
    """Function to reassign lat/lon variables to mesh2_node variables. Could be
    possible to expand this to include:

    - Error message if variable name is not in current dictionary
    - Further capability to have a user input for variables not in dictionary

    :param ds: xarray.Dataset used
    :param cf: Bool tells code if dataset is cf compliant or not, default True
    :return: Reassigns data variables to be mesh2_node_x and mesh2_node_y
    """

    NODE_X = None
    NODE_Y = None

    if is_cf == True:
        try:
            NODE_X = ds['Mesh2_node_x']
            NODE_Y = ds['Mesh2_node_y']
        except:
            raise Exception(
                "Variables not in form 'Mesh2_node_x' or 'Mesh2_node_y' please specify cf=False"
            )

    else:
        _get_nodes(ds)

    return NODE_X, NODE_Y
