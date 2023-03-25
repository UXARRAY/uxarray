import numpy as np
import xarray as xr

from typing import Optional

from uxarray.core.grid import Grid


class UxDataArray(xr.DataArray):
    # expected instance attributes, required for subclassing with xarray (as of v0.13.0)
    __slots__ = ("_uxgrid",)

    def __init__(self, *args, uxgrid: Grid = None, **kwargs):
        super().__init__(*args, **kwargs)

        self._uxgrid = uxgrid

        if uxgrid is None or not isinstance(uxgrid, Grid):
            raise RuntimeError(
                "uxarray.core.UxDataArray.__init__: uxgrid cannot be None. It needs to "
                "be an instance of the uxarray.core.Grid class")
        else:
            self.uxgrid = uxgrid

    @property
    def uxgrid(self):
        return self._uxgrid

    # a setter function
    @uxgrid.setter
    def uxgrid(self, grid_obj):
        self._uxgrid = grid_obj

    # You can add custom methods to the class here.
    def custom_method(self):
        print("Custom method for the class")
