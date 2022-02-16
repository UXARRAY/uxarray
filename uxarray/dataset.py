"""uxarray dataset module."""

from logging import raiseExceptions
import xarray as xr
import numpy as np
from pathlib import PurePath
import os
from .grid import *


def open_dataset(filename, type):
    """A class for uxarray dataset object returns an xarray object with Mesh2
    construct representing the grid.

    Examples
    ========
    import uxarray as ux
    # open an exodus file with Grid object
    mesh = ux.open_dataset("filename.g")
    """
    print("opening dataset: ", filename)
    if type == "exo" or "ugrid" or "scrip" or "shp":
        ux_grid = Grid(str(filename))
    else:
        print("Meshfile type is not supported")
    return ux_grid.in_ds
