import numpy as np
import xarray as xr

from typing import Optional

class UxDataset(xr.Dataset):

    _uxgrid = None

    def __init__(self,
                 uxgrid,
                 *args,
                 source_datasets: Optional[str] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        setattr(self, 'source_datasets', source_datasets)

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
        >>> uxds = ux.open_dataset("centroid_pressure_data_ug", "grid.ug")

        # Compute the integral
        >>> integral = uxds.integrate()
        """
        integral = 0.0

        # call function to get area of all the faces as a np array
        face_areas = self.uxgrid.compute_face_areas(quadrature_rule, order)

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

