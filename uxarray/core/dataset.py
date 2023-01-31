import xarray as xr

class UxDataset(xr.Dataset):

    _uxgrid = None

    def __init__(self, uxgrid, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.uxgrid = uxgrid

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
