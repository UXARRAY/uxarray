from asv_runner.benchmarks.mark import skip_benchmark_if

import uxarray as ux
import os


# 30km, 15km, 7.5km, 3.75km

file_path_dict = {"30km": ["/glade/campaign/cisl/vast/uxarray/data/dyamond/30km/grid.nc", "/glade/"],
                  "15km": ["/glade/campaign/cisl/vast/uxarray/data/dyamond/15km/grid.nc", "/glade/"],
                  "7.5km":["/glade/campaign/cisl/vast/uxarray/data/dyamond/7.5km/grid.nc", "/glade/"],
                  "3.75km":["/glade/campaign/cisl/vast/uxarray/data/dyamond/3.75km/grid.nc", "/glade/"]}

all_paths_exist = True
for file_paths in file_path_dict.values():
    for file_path in file_paths:
        all_paths_exist = all_paths_exist and os.path.exists(file_path)

@skip_benchmark_if(not all_paths_exist)
class MatplotlibConversion:
    param_names = ['resolution', 'periodic_elements']
    params = (['30km', '15km', '7.5km', '3.75km'], ['include', 'exclude', 'split'])

    def setup(self, resolution, periodic_elements):
        self.uxgrid = ux.open_grid(file_path_dict[resolution[0]])

    def teardown(self, resolution, periodic_elements):
        del self.uxgrid

    def time_grid_to_polycollection(self, resolution, periodic_elements):
        self.uxgrid.to_polycollection()
