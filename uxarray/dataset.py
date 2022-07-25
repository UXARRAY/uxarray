"""uxarray dataset module."""

from .grid import *
from .helpers import parse_grid_type


def open_dataset(grid_filename, *args, **kw):
    """Given a grid file and/or other files with corresponding data.
    This function merges them to output a xarray dataset object.
    Parameters
    ----------

    grid_filename : string, required
        Grid file name is the first argument.
    *args : string, optional
        datafile name(s) corresponding to the grid_filename

    Returns
    -------

    object : xarray Dataset
        uxarray grid object: Contains the grid and corresponding data

    Examples
    --------

    Open grid file only
    >>> mesh = ux.open_dataset("grid_filename.g")


    Open grid file along with data
    >>> mesh_and_data = ux.open_dataset("grid_filename.g", "grid_filename_vortex.nc")
    """
    mesh_filetype, dataset = parse_grid_type(grid_filename, **kw)
    ux_grid = Grid(dataset=dataset, mesh_filetype=mesh_filetype)

    if len(args) > 0:
        # load all the datafiles using mfdataset
        all_data = xr.open_mfdataset(args)
        # merge data with grid ds
        ux_grid.ds = xr.merge([ux_grid.ds, all_data])

    # return grid with data
    return ux_grid
