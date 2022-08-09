"""UXarray dataset module."""

from .grid import *
from .helpers import parse_grid_type


def open_dataset(grid_file, *args, **kw):
    """Creates a UXarray Grid object, given a single grid file with or without
    grid data file(s) with corresponding data. This function merges all those
    files into the Grid object that includes a Xarray dataset object.

    Parameters
    ----------

    grid_file : string, required
        Grid file is the first argument, which should be the file that
        houses the unstructured grid definition. It should be comapatible to
        be opened with xarray.open_dataset (e.g. path to a file in the local
        storage, OpenDAP URL, etc).
    *args : string, optional
        Data file(s) corresponding to the grid_file. They should be comapatible
        to be opened with xarray.open_dataset (e.g. path to a file in the local
        storage, OpenDAP URL, etc).

    Returns
    -------

    object : uxarray Grid
        UXarray Grid object that contains the grid definition and corresponding
        data.

    Examples
    --------

    Open grid file only
    >>> mesh = ux.open_dataset("grid_filename.g")


    Open grid file along with data
    >>> mesh_and_data = ux.open_dataset("grid_filename.g", "grid_filename_vortex.nc")
    """
    mesh_filetype, dataset = parse_grid_type(grid_file, **kw)

    # Determine the source data sets regarding args
    source_datasets = np.array(args) if len(args) > 0 else None

    # Construct the Grid object
    ux_grid = Grid(dataset=dataset,
                   mesh_filetype=mesh_filetype,
                   source_grid=grid_file,
                   source_datasets=source_datasets)

    # If there are additional data file(s) corresponding to this grid, merge them
    # into the dataset
    if len(args) > 0:
        # load all the datafiles using mfdataset
        all_data = xr.open_mfdataset(args)
        # merge data with grid ds
        ux_grid.ds = xr.merge([ux_grid.ds, all_data])

    return ux_grid
