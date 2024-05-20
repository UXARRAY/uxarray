import os
import urllib.request
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

data_var = 'bottomDepth'

grid_filename_480 = "oQU480.grid.nc"
data_filename_480 = "oQU480.data.nc"

grid_filename_120 = "oQU120.grid.nc"
data_filename_120 = "oQU120.data.nc"

filenames = [grid_filename_480, data_filename_480, grid_filename_120, data_filename_120]

for filename in filenames:
    if not os.path.isfile(current_path / filename):
        # downloads the files from Cookbook repo, if they haven't been downloaded locally yet
        url = f"https://github.com/ProjectPythia/unstructured-grid-viz-cookbook/raw/main/meshfiles/{filename}"
        _, headers = urllib.request.urlretrieve(url, filename=current_path / filename)


file_path_dict = {"480km": [current_path / grid_filename_480, current_path / data_filename_480],
                  "120km": [current_path / grid_filename_120, current_path / data_filename_120]}


class Gradient:

    param_names = ['resolution']
    params = ['480km', '120km']


    def setup(self, resolution):
        self.uxds = ux.open_dataset(file_path_dict[resolution][0], file_path_dict[resolution][1])

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
        self.uxds = ux.open_dataset(file_path_dict[resolution][0], file_path_dict[resolution][1])

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
        self.uxds = ux.open_dataset(file_path_dict[resolution][0], file_path_dict[resolution][1])

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
        self.uxds = ux.open_dataset(file_path_dict[resolution][0], file_path_dict[resolution][1])

    def teardown(self, resolution):
        del self.uxds

    def time_n_nodes_per_face(self, resolution):
        self.uxds.uxgrid.n_nodes_per_face
