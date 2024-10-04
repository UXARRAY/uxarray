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



class DatasetBenchmark:
    """Class used as a template for benchmarks requiring a ``UxDataset`` in
    this module across both resolutions."""
    param_names = ['resolution',]
    params = [['480km', '120km'],]


    def setup(self, resolution, *args, **kwargs):


        self.uxds = ux.open_dataset(file_path_dict[resolution][0], file_path_dict[resolution][1])

    def teardown(self, resolution, *args, **kwargs):
        del self.uxds

class GridBenchmark:
    """Class used as a template for benchmarks requiring a ``Grid`` in this
    module across both resolutions."""
    param_names = ['resolution', ]
    params = [['480km', '120km'], ]

    def setup(self, resolution, *args, **kwargs):

        self.uxgrid = ux.open_grid(file_path_dict[resolution][0])

    def teardown(self, resolution, *args, **kwargs):
        del self.uxgrid



class Gradient(DatasetBenchmark):
    def time_gradient(self, resolution):
        self.uxds[data_var].gradient()

    def peakmem_gradient(self, resolution):
        grad = self.uxds[data_var].gradient()

class Integrate(DatasetBenchmark):
    def time_integrate(self, resolution):
        self.uxds[data_var].integrate()

    def peakmem_integrate(self, resolution):
        integral = self.uxds[data_var].integrate()


class GeoDataFrame(DatasetBenchmark):
    param_names = DatasetBenchmark.param_names + ['exclude_antimeridian']
    params = DatasetBenchmark.params + [[True, False]]


    def time_to_geodataframe(self, resolution, exclude_antimeridian):
        self.uxds[data_var].to_geodataframe(exclude_antimeridian=exclude_antimeridian)



class ConnectivityConstruction(DatasetBenchmark):
    def time_n_nodes_per_face(self, resolution):
        self.uxds.uxgrid.n_nodes_per_face

    def time_face_face_connectivity(self, resolution):
        ux.grid.connectivity._populate_face_face_connectivity(self.uxds.uxgrid)


class MatplotlibConversion(DatasetBenchmark):
    param_names = DatasetBenchmark.param_names + ['periodic_elements']
    params = DatasetBenchmark.params + [['include', 'exclude', 'split']]

    def time_dataarray_to_polycollection(self, resolution, periodic_elements):
        self.uxds[data_var].to_polycollection()


class ConstructTreeStructures(DatasetBenchmark):
    def time_kd_tree(self, resolution):
        self.uxds.uxgrid.get_kd_tree()

    def time_ball_tree(self, resolution):
        self.uxds.uxgrid.get_ball_tree()


class RemapDownsample:

    def setup(self):
        self.uxds_120 = ux.open_dataset(file_path_dict['120km'][0], file_path_dict['120km'][1])
        self.uxds_480 = ux.open_dataset(file_path_dict['480km'][0], file_path_dict['480km'][1])

    def teardown(self):
        del self.uxds_120, self.uxds_480

    def time_nearest_neighbor_remapping(self):
        self.uxds_120["bottomDepth"].remap.nearest_neighbor(self.uxds_480.uxgrid)

    def time_inverse_distance_weighted_remapping(self):
        self.uxds_120["bottomDepth"].remap.inverse_distance_weighted(self.uxds_480.uxgrid)


class RemapUpsample:

    def setup(self):
        self.uxds_120 = ux.open_dataset(file_path_dict['120km'][0], file_path_dict['120km'][1])
        self.uxds_480 = ux.open_dataset(file_path_dict['480km'][0], file_path_dict['480km'][1])

    def teardown(self):
        del self.uxds_120, self.uxds_480

    def time_nearest_neighbor_remapping(self):
        self.uxds_480["bottomDepth"].remap.nearest_neighbor(self.uxds_120.uxgrid)

    def time_inverse_distance_weighted_remapping(self):
        self.uxds_480["bottomDepth"].remap.inverse_distance_weighted(self.uxds_120.uxgrid)

class HoleEdgeIndices(DatasetBenchmark):
    def time_construct_hole_edge_indices(self, resolution):
        ux.grid.geometry._construct_hole_edge_indices(self.uxds.uxgrid.edge_face_connectivity)

class CheckNorm:
    param_names = ['resolution']
    params = ['480km', '120km']

    def setup(self, resolution):
        self.uxgrid = ux.open_grid(file_path_dict[resolution][0])

    def teardown(self, resolution):
        del self.uxgrid

    def time_check_norm(self, resolution):
        from uxarray.grid.validation import _check_normalization
        _check_normalization(self.uxgrid)
