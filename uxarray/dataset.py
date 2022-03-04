"""uxarray dataset module."""

from .grid import *
from pathlib import PurePath


def open_dataset(filename, *args):
    """A class for uxarray dataset object returns an xarray object with Mesh2
    construct representing the grid.
    args: takes mesh file name first argument
          and datafile names corresponding to the mesh file as args
    output: returns xarray object with grid and data info combined

    Examples
    ========
    import uxarray as ux
    # open an exodus file with Grid object
    mesh = ux.open_dataset("filename.g")

    mesh_data = ux.open_dataset("filename.g", "filename_vortex.nc")
    """
    print("Loading initial grid from file: ", filename)
    ux_grid = Grid(str(filename))

    # open all the datafiles with xarrays
    xr_datafile = [None] * len(args)
    i = 0

    for datafile in args:
        print("Opening data from file: ", filename)
        xr_datafile[i] = xr.open_dataset(datafile)
        try:
            print("Merging grid and data")
            ux_grid.in_ds = xr.merge([ux_grid.in_ds, xr_datafile[i]])
        except:
            raise RuntimeError("unable to merge grid and datafile", datafile)
        i += 1

    if (ux_grid.in_ds):
        return ux_grid.in_ds
    else:
        raise RuntimeError("unable to get uxarray grid object from file:" +
                           filename)
