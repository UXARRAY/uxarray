import numpy as np
import xarray as xr

from collections.abc import Hashable

from typing import Optional

from uxarray.core.dataarray import UxDataArray
from uxarray.core.grid import Grid


class UxDataset(xr.Dataset):

    _uxgrid = None

    def __init__(self,
                 *args,
                 uxgrid: Grid = None,
                 source_datasets: Optional[str] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        setattr(self, 'source_datasets', source_datasets)

        self.uxgrid = uxgrid

        if uxgrid is None or not isinstance(uxgrid, Grid):
            raise RuntimeError(
                "uxgrid cannot be None or it needs to "
                "be of an instance of the uxarray.core.Grid class")
        else:
            self.uxgrid = uxgrid

    @property
    def uxgrid(self):
        return self._uxgrid

    # a setter function
    @uxgrid.setter
    def uxgrid(self, ugrid):

        self._uxgrid = ugrid

    def _construct_dataarray(self, name) -> UxDataArray:
        """Override to check if the result is an instance of xarray.DataArray.

        If so, convert to UxDataArray.
        """
        xarr = super()._construct_dataarray(name)

        return UxDataArray(xarr, uxgrid=self.uxgrid)

    def __getitem__(self, key):
        """Override to check if the result is an instance of xarray.DataArray.

        If so, convert to UxDataArray.
        """
        xarr = super().__getitem__(key)

        if isinstance(xarr, xr.DataArray):
            return UxDataArray(xarr, uxgrid=self.uxgrid)
        else:
            return xarr

    def __setitem__(self, key, value):
        """Override to check if the value being set is an instance of
        xarray.DataArray.

        If so, convert to UxDataArray.
        """
        if isinstance(value, xr.DataArray):
            value = UxDataArray(value)
            # works with the value below also.
            # value = value.to_dataarray()

        super().__setitem__(key, value)

    @classmethod
    def from_dataframe(cls, dataframe):
        """Convert to a UxDataset instead of an xarray.Dataset."""
        return cls(
            {
                col: ('index', dataframe[col].values)
                for col in dataframe.columns
            },
            coords={'index': dataframe.index})

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Convert to a UxDataset instead of an xarray.Dataset."""
        return cls({key: ('index', val) for key, val in data.items()},
                   coords={'index': range(len(next(iter(data.values()))))},
                   **kwargs)

    # You can add custom methods to the class here
    def custom_method(self):
        print("Custom method for the class")

    def integrate(self, quadrature_rule="triangular", order=4):
        """Integrates over all the faces of the given mesh.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        Calculated integral : float

        Examples
        --------
        Open a Uxarray dataset

        >>> import uxarray as ux
        >>> uxds = ux.open_dataset("grid.ug", "centroid_pressure_data_ug")

        # Compute the integral
        >>> integral = uxds.integrate()
        """
        integral = 0.0

        # call function to get area of all the faces as a np array
        face_areas = self.uxgrid.compute_face_areas(quadrature_rule, order)

        # TODO: Fix this requirement. It should be applicable to
        # TODO: either all variables of dataset or a dataarray instead.
        var_key = list(self.keys())
        if len(var_key) > 1:
            # warning: print message
            print(
                "WARNING: The dataset has more than one variable, using the first variable for integration"
            )

        var_key = var_key[0]
        face_vals = self[var_key].to_numpy()
        integral = np.dot(face_areas, face_vals)

        return integral
