import os

from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]

base_path = current_path / "test" / "meshfiles" / "mpas" / "QU"

file_path_dict = {"480km": [base_path / "oQU480.grid.zarr", base_path / "oQU480.data.zarr"],
                  "120km": [base_path / "oQU120.grid.zarr", base_path / "oQU120.data.zarr"]}

data_var = "bottomDepth"


class Gradient:

    param_names = ['resolution']
    params = ['480km', '120km']


    def setup(self, resolution):
        self.uxds = ux.open_dataset(file_path_dict[resolution][0], file_path_dict[resolution][1],
                                    engine="zarr",
                                    grid_kwargs={"engine": "zarr"})

    def teardown(self, resolution):
        del self.uxds

    def time_gradient(self, resolution):
        self.uxds[data_var].gradient()

    def peakmem_gradient(self, resolution):
        grad = self.uxds[data_var].gradient()

class Integrate:

    param_names = ['resolution']
    params = ['480km', '120km']


    def setup(self, resolution):
        self.uxds = ux.open_dataset(file_path_dict[resolution][0], file_path_dict[resolution][1],
                                    engine="zarr",
                                    grid_kwargs={"engine": "zarr"})

    def teardown(self, resolution):
        del self.uxds

    def time_integrate(self, resolution):
        self.uxds[data_var].integrate()

    def peakmem_integrate(self, resolution):
        integral = self.uxds[data_var].integrate()

class GeoDataFrame:

    param_names = ['resolution', 'exclude_antimeridian']
    params = [['480km', '120km'],
              [True, False]]


    def setup(self, resolution, exclude_antimeridian):
        self.uxds = ux.open_dataset(file_path_dict[resolution][0], file_path_dict[resolution][1],
                                    engine="zarr",
                                    grid_kwargs={"engine": "zarr"})

    def teardown(self, resolution, exclude_antimeridian):
        del self.uxds

    def time_to_geodataframe(self, resolution, exclude_antimeridian):
        self.uxds[data_var].to_geodataframe(exclude_antimeridian=exclude_antimeridian)

    def peakmem_to_geodataframe(self, resolution, exclude_antimeridian):
        gdf = self.uxds[data_var].to_geodataframe(exclude_antimeridian=exclude_antimeridian)


class ConnectivityConstruction:

    param_names = ['resolution']
    params = ['480km', '120km']


    def setup(self, resolution):
        self.uxds = ux.open_dataset(file_path_dict[resolution][0], file_path_dict[resolution][1],
                                    engine="zarr",
                                    grid_kwargs={"engine": "zarr"})

    def teardown(self, resolution):
        del self.uxds

    def time_n_nodes_per_face(self, resolution):
        self.uxds.uxgrid.n_nodes_per_face
