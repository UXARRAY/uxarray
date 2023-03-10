import numpy as np
import xarray as xr

from typing import Optional

from uxarray.core.grid import Grid


class UxDataArray(xr.DataArray):
    __slots__ = ("source_datasets")
    _uxgrid = None

    def __init__(self, *args, uxgrid: Grid = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.uxgrid = uxgrid

        if uxgrid is None:
            raise RuntimeError("uxgrid cannot be None")
        else:
            self.uxgrid = uxgrid

    # def __slots__ = ()
    #     # print("slots")
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
