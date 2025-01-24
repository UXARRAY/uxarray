import os

from asv_runner.benchmarks.mark import skip_benchmark_if, timeout_class_at

import uxarray as ux

# Paths to grid files on Glade
grid_path_dict = {"30km": "/glade/campaign/cisl/vast/uxarray/data/dyamond/30km/grid.nc",
                  "15km": "/glade/campaign/cisl/vast/uxarray/data/dyamond/15km/grid.nc",
                  "7.5km": "/glade/campaign/cisl/vast/uxarray/data/dyamond/7.5km/grid.nc",
                  "3.75km": "/glade/campaign/cisl/vast/uxarray/data/dyamond/3.75km/grid.nc"}



# Determines if all file paths exist and are accesible
all_paths_exist = True
for file_path in grid_path_dict.values():
    all_paths_exist = all_paths_exist and os.path.exists(file_path)


class BaseGridBenchmark:
    """Base class for Grid Benchmarks across the four supported resolutions
    (30km, 15km, 7.5km, 3.75km)"""
    param_names = ['resolution']
    params = [['30km', '15km', '7.5km', '3.75km'], ]

    def setup(self, resolution, **kwargs):
        self.uxgrid = ux.open_grid(grid_path_dict[resolution])

    def teardown(self, resolution, **kwargs):
        del self.uxgrid

@timeout_class_at(1200)
class OpenGrid:
    param_names = ['resolution']
    params = [['30km', '15km', '7.5km', '3.75km'], ]

    @skip_benchmark_if(not all_paths_exist)
    def time_open_grid(self, resolution):
        _ = ux.open_grid(grid_path_dict[resolution])


@timeout_class_at(1200)
class Bounds(BaseGridBenchmark):
    @skip_benchmark_if(not all_paths_exist)
    def time_bounds(self, resolution):
        _ = self.uxgrid.bounds

@timeout_class_at(1200)
class GeoDataFrame(BaseGridBenchmark):
    @skip_benchmark_if(not all_paths_exist)
    def time_to_geodataframe(self, resolution):
        self.uxgrid.to_geodataframe(exclude_antimeridian=True)
