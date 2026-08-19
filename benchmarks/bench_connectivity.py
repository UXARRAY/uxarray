import os
import urllib.request
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

grid_filename_480 = "oQU480.grid.nc"
grid_filename_120 = "oQU120.grid.nc"
filenames = [grid_filename_480, grid_filename_120]

for filename in filenames:
    if not os.path.isfile(current_path / filename):
        # downloads the files from Cookbook repo, if they haven't been downloaded locally yet
        url = f"https://github.com/ProjectPythia/unstructured-grid-viz-cookbook/raw/main/meshfiles/{filename}"
        _, headers = urllib.request.urlretrieve(url, filename=current_path / filename)

oQU_path_dict = {"480km": current_path / grid_filename_480,
                  "120km": current_path / grid_filename_120}

# Paths to grid files on Glade
dyamond_path_dict = {"30km": "/glade/campaign/cisl/vast/uxarray/data/dyamond/30km/grid.nc",
                  "15km": "/glade/campaign/cisl/vast/uxarray/data/dyamond/15km/grid.nc",
                  "7.5km": "/glade/campaign/cisl/vast/uxarray/data/dyamond/7.5km/grid.nc",
                  "3.75km": "/glade/campaign/cisl/vast/uxarray/data/dyamond/3.75km/grid.nc"}

# Determines if all file paths exist and are accesible
all_paths_exist = True
for file_path in dyamond_path_dict.values():
    all_paths_exist = all_paths_exist and os.path.exists(file_path)

file_path_dict = oQU_path_dict
if all_paths_exist:
    file_path_dict = file_path_dict | dyamond_path_dict


class GridBenchmark:
    """Class used as a template for benchmarks requiring a ``Grid`` in this
    module across both resolutions."""
    param_names = ['resolution', ]

    # Conditionally available; could get annoying if there are downstream tools relying on it.
    if all_paths_exist:
        params = [['480km', '120km', '30km', '15km', '7.5km', '3.75km'], ]
    else:
        params = [['480km', '120km'], ]

    def setup(self, resolution, *args, **kwargs):
        self.uxgrid = ux.open_grid(file_path_dict[resolution])

    def teardown(self, resolution, *args, **kwargs):
        del self.uxgrid

CONNECTIVITY_NAMES = [
    "n_nodes_per_face",
    "face_node_connectivity",
    "edge_node_connectivity",
    "face_edge_connectivity",
    "node_edge_connectivity",
    "face_face_connectivity",
    "edge_face_connectivity",
    "node_face_connectivity",
]

_numba_warmed_up = False

def _warmup(uxgrid):
    """Compiles the Numba kernels backing each connectivity variable.

    ``_build_node_edge_connectivity`` is not disk-cached, so a fresh benchmark
    process would otherwise charge ~240ms of JIT compilation to whichever sample
    happened to touch it first.
    """
    global _numba_warmed_up
    if _numba_warmed_up:
        return
    for name in CONNECTIVITY_NAMES:
        getattr(uxgrid, name)
    _numba_warmed_up = True


class Connectivity(GridBenchmark):
    # Each connectivity variable is cached in ``Grid._ds`` once constructed, so a
    # sample may only contain a single call; otherwise every call but the first
    # would time a dictionary lookup.
    number = 1

    def setup(self, resolution, *args, **kwargs):
        # The benchmark grids are MPAS meshes, which carry every connectivity
        # variable on disk. Reading one would time the MPAS parser rather than
        # the construction routines, so reduce the grid down to the minimal
        # UGRID topology and let each variable be built on demand.
        source_grid = ux.open_grid(file_path_dict[resolution])
        self.topology = (
            source_grid.node_lon.data,
            source_grid.node_lat.data,
            source_grid.face_node_connectivity.data,
        )

        _warmup(self.minimal_grid())
        self.uxgrid = self.minimal_grid()

    def minimal_grid(self):
        return ux.Grid.from_topology(*self.topology)

    def teardown(self, resolution, *args, **kwargs):
        del self.uxgrid
        del self.topology

    def time_n_nodes_per_face(self, resolution):
        _ = self.uxgrid.n_nodes_per_face.compute()

    def time_face_node(self, resolution):
        _ = self.uxgrid.face_node_connectivity.compute()

    def time_edge_node(self, resolution):
        _ = self.uxgrid.edge_node_connectivity.compute()

#   TODO: Not yet supported?
#   def time_node_node(self, resolution):
#       _ = self.uxgrid.node_node_connectivity

    def time_face_edge(self, resolution):
        _ = self.uxgrid.face_edge_connectivity.compute()

#   TODO: Not yet supported?
#   def time_edge_edge(self, resolution):
#        _ = self.uxgrid.edge_edge_connectivity

    def time_node_edge(self, resolution):
        _ = self.uxgrid.node_edge_connectivity.compute()

    def time_face_face(self, resolution):
        _ = self.uxgrid.face_face_connectivity.compute()

    def time_edge_face(self, resolution):
        _ = self.uxgrid.edge_face_connectivity.compute()

    def time_node_face(self, resolution):
        _ = self.uxgrid.node_face_connectivity.compute()
