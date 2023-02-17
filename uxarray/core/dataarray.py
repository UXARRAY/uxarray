import numpy as np
import xarray as xr

from typing import Optional

from uxarray.core.grid import Grid


class UxDataArray(xr.DataArray):

    _uxgrid = None

    def __init__(self, *args, uxgrid: Grid = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.uxgrid = uxgrid
        # TODO: Weird that below if-check leads miscreation of UxDataArray object
        # if uxgrid is None:
        #     raise RuntimeError("uxgrid cannot be None")
        # else:
        #     self.uxgrid = uxgrid

    @property
    def uxgrid(self):
        return self._uxgrid

    # a setter function
    @uxgrid.setter
    def uxgrid(self, ugrid):

        self._uxgrid = ugrid

    # You can add custom methods to the class here
    def custom_method(self):
        print("Custom method for the class")
