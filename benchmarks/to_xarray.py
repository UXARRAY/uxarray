"""Benchmarks for to_xarray() methods,
like UxDataset.to_xarray() and UxDataArray.to_xarray().
"""

import numpy as np
import uxarray as ux


def _assign_coords(obj, n_coords):
    n_face = obj.sizes["n_face"]
    return obj.assign_coords(
        {f"c{i}": ("n_face", np.arange(n_face, dtype=float)) for i in range(n_coords)}
    )


class UxDatasetToXarray:
    """Benchmark ``UxDataset.to_xarray()``, which is dominated by per-variable overhead."""

    params = [[(100, 10), (1000, 100)]]
    param_names = ["n_vars_and_n_coords"]

    def setup(self, n_vars_and_n_coords):
        n_vars, n_coords = n_vars_and_n_coords
        uxds = ux.tutorial.open_dataset("outCSne30-vortex")
        uxds = uxds.assign({f"v{i}": uxds["psi"]+i for i in range(n_vars-1)})
        self.uxds = _assign_coords(uxds, n_coords)

    def time_to_xarray(self, n_vars_and_n_coords):
        self.uxds.to_xarray()


class UxDataArrayToXarray:
    """Benchmark ``UxDataArray.to_xarray()`` with many (n_face-sized) coords."""

    params = [[10]]
    param_names = ["n_coords"]

    def setup(self, n_coords):
        uxds = ux.tutorial.open_dataset("outCSne30-vortex")
        self.uxda = _assign_coords(uxds["psi"], n_coords)

    def time_to_xarray(self, n_coords):
        self.uxda.to_xarray()
