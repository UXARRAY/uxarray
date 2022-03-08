# Create list of names that can be used for node coords
_X_NODES = ("lat", "latitude", "Lat", "Latitude", "X", "x", "grid_corner_lat",
            "grid_corner_latitude")

_Y_NODES = ("lon", "longitude", "Lon", "Longitude", "Y", "y", "grid_corner_lon",
            "grid_corner_longitude")


def _viewkeys(d):
    """Return either the keys or viewkeys method for a dictionary.
    Parameters
    ----------
    Args:
        d :obj:`dict`: A dictionary.
    Returns:
        view method: Either the keys or viewkeys method.
    """

    func = getattr(d, "viewkeys", None)
    if func is None:
        func = d.keys
    return func()


def _get_nodes(ds):
    """Outer wrapper function for datasets that are not cf compliant.

    ds : :class:`xarray.Dataset`
        Scrip dataset of interest being used

    return : :class:`xarray.Dataset`
        Dataset updated with cf compliant naming of lat/lon variables
    """

    node_x = None
    node_y = None

    for name in _X_NODES:
        if name in _viewkeys(ds):
            ds['Mesh2_node_x'] = ds[name]
            node_x = ds['Mesh2_node_x']

            break

    for name in _Y_NODES:
        if name in _viewkeys(ds):
            ds['Mesh2_node_y'] = ds[name]
            node_y = ds['Mesh2_node_y']

            break

    return node_x, node_y


def _populate_scrip_data(ds, is_cf=True):
    """Function to reassign lat/lon variables to mesh2_node variables. Could be
    possible to expand this to include:

    - Error message if variable name is not in current dictionary
    - Further capability to have a user input for variables not in dictionary

    Parameters
    ----------
    ds : :class:`xarray.Dataset`
        Scrip dataset of interest being used

    cf : :class:`bool` optional
        User defined if dataset is known as cf compliant or not, default True

    Returns
    --------
    node_x : :class:`xarray.Variable`
        Reassigns data variables to be mesh2_node_x

    node_y : :class:`xarray.Variable`
        Reassigns data variables to be mesh2_node_y
    """

    node_x = None
    node_y = None

    if is_cf == True:
        try:
            node_x = ds['Mesh2_node_x']
            node_y = ds['Mesh2_node_y']
        except:
            raise Exception(
                "Variables not in form 'Mesh2_node_x' or 'Mesh2_node_y' please specify cf=False"
            )

    else:
        _get_nodes(ds)

    return node_x, node_y
