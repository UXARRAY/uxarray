"""Opening synthetic ensemble data with a rebuilt or reusable grid."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

import uxarray as ux


class OpenDatasetGridReuse:
    param_names = ["grid_input", "data_input"]
    params = [["file", "dataset", "grid"], ["file", "dataset"]]

    def setup(self, grid_input, data_input):
        """Prepare synthetic inputs outside the timed member-opening path."""
        self.directory = TemporaryDirectory()
        self.grid = ux.Grid.from_topology(
            node_lon=np.array([0.0, 1.0, 1.0, 0.0]),
            node_lat=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        )
        self.grid_ds = self.grid.to_xarray()
        self.data_ds = xr.Dataset(
            {"temperature": (("time", "n_face"), np.arange(20).reshape(10, 2))}
        )
        directory = Path(self.directory.name)
        grid_path = directory / "grid.nc"
        data_path = directory / "member.nc"
        self.grid_ds.to_netcdf(grid_path)
        self.data_ds.to_netcdf(data_path)
        self.grid_input = {
            "file": grid_path,
            "dataset": self.grid_ds,
            "grid": self.grid,
        }[grid_input]
        self.data_input = data_path if data_input == "file" else self.data_ds

    def teardown(self, grid_input, data_input):
        """Remove the generated files after measurement."""
        self.directory.cleanup()

    def time_open_dataset(self, grid_input, data_input):
        """Time opening one member, retaining the Grid when already available."""
        result = ux.open_dataset(self.grid_input, self.data_input)
        result.close()
        if grid_input == "file":
            result.uxgrid._ds.close()
