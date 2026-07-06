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

# Paths to grid files on Glade
dyamond_path_dict = {"30km": "/glade/campaign/cisl/vast/uxarray/data/dyamond/30km/grid.nc",
                  "15km": "/glade/campaign/cisl/vast/uxarray/data/dyamond/15km/grid.nc",
                  "7.5km": "/glade/campaign/cisl/vast/uxarray/data/dyamond/7.5km/grid.nc",
                  "3.75km": "/glade/campaign/cisl/vast/uxarray/data/dyamond/3.75km/grid.nc"}

oQU_path_dict = {"480km": [current_path / grid_filename_480, current_path / data_filename_480],
                  "120km": [current_path / grid_filename_120, current_path / data_filename_120]}

file_path_dict = oQU_path_dict | dyamond_path_dict


class GridBenchmark:
    """Class used as a template for benchmarks requiring a ``Grid`` in this
    module across both resolutions."""
    param_names = ['resolution', ]
    params = [['480km', '120km', '30km', '15km', '7.5km', '3.75km'], ]

    def setup(self, resolution, *args, **kwargs):
        self.uxgrid = ux.open_grid(file_path_dict[resolution][0])

    def teardown(self, resolution, *args, **kwargs):
        del self.uxgrid

class Connectivity(GridBenchmark):

    def time_face_node(self, resolution):
        _ = self.uxgrid.face_node_connectivity

    def time_edge_node(self, resolution):
        _ = self.uxgrid.edge_node_connectivity

#   TODO: Not yet supported?
#   def time_node_node(self, resolution):
#       _ = self.uxgrid.node_node_connectivity

    def time_face_edge(self, resolution):
        _ = self.uxgrid.face_edge_connectivity

#   TODO: Not yet supported?
#   def time_edge_edge(self, resolution):
#        _ = self.uxgrid.edge_edge_connectivity

    def time_node_edge(self, resolution):
        _ = self.uxgrid.node_edge_connectivity

    def time_face_face(self, resolution):
        _ = self.uxgrid.face_face_connectivity

    def time_edge_face(self, resolution):
        _ = self.uxgrid.edge_face_connectivity

    def time_node_face(self, resolution):
        _ = self.uxgrid.node_face_connectivity
