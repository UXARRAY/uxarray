import os
import urllib.request
from pathlib import Path

import uxarray as ux

from .helpers._peakmem import peak_allocated

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

# Direct prerequisites only, read off the ``_populate_*`` functions in
# ``uxarray/grid/connectivity.py``; accessing one builds its own in turn.
CONNECTIVITY_PREREQUISITES = {
    "n_nodes_per_face": (),
    "face_node_connectivity": (),
    "edge_node_connectivity": ("n_nodes_per_face",),
    # ``_populate_edge_node_connectivity`` writes this one out too, so it costs
    # nothing once that has run.
    "face_edge_connectivity": ("edge_node_connectivity",),
    "node_edge_connectivity": ("edge_node_connectivity",),
    "face_face_connectivity": ("edge_face_connectivity",),
    "edge_face_connectivity": ("face_edge_connectivity",),
    "node_face_connectivity": (),
}


def _build_prerequisites(uxgrid, connectivity):
    """Builds ``connectivity``'s prerequisites, so what follows measures one
    construction routine rather than the whole chain rooted at it."""
    for prerequisite in CONNECTIVITY_PREREQUISITES[connectivity]:
        getattr(uxgrid, prerequisite)
    return uxgrid


_numba_warmed_up = False

def _warmup():
    """Compiles the Numba kernels backing each connectivity variable.

    Every kernel in ``uxarray/grid/connectivity.py`` is ``@njit(cache=True)``, so
    this carries across processes through Numba's on-disk cache -- what makes it
    usable from ``setup_cache``. Loading from that cache still allocates, so it
    matters for ``track_peakmem_*`` too, not just timing.
    """
    global _numba_warmed_up
    if _numba_warmed_up:
        return
    # Resolution affects how long the kernels run, not which signatures compile.
    uxgrid = ux.Grid.from_topology(*_source_topology(GridBenchmark.params[0][0]))
    for name in CONNECTIVITY_NAMES:
        getattr(uxgrid, name)
    _numba_warmed_up = True


_topology_cache = {}


def _source_topology(resolution):
    """The minimal UGRID topology for ``resolution``, read once per process.

    The benchmark grids are MPAS meshes carrying every connectivity variable on
    disk; reading one would measure the MPAS parser rather than the construction
    routines, so each variable is left to be built on demand.

    Cached because asv re-runs ``setup`` between repeats, and at dyamond
    resolutions re-reading the source grid dwarfs the sample it precedes.
    """
    if resolution not in _topology_cache:
        source_grid = ux.open_grid(file_path_dict[resolution])
        _topology_cache[resolution] = (
            source_grid.node_lon.data,
            source_grid.node_lat.data,
            source_grid.face_node_connectivity.data,
        )
    return _topology_cache[resolution]


class MinimalGridBenchmark(GridBenchmark):
    """Template for benchmarks that construct connectivity variables on demand.

    Holds a ``Grid`` carrying nothing but the minimal UGRID topology, plus the
    topology needed to mint further ones, and leaves the Numba kernels compiled.
    """

    # Handover slot for ``_prerequisite_setup``; see its docstring.
    active_grid = None

    # asv's 60s default is not enough to build a connectivity variable at 3.75km.
    timeout = 1800

    def setup(self, resolution, *args, **kwargs):
        self.topology = _source_topology(resolution)

        _warmup()
        self.uxgrid = self.minimal_grid()
        MinimalGridBenchmark.active_grid = self.uxgrid

    def minimal_grid(self):
        """Mints a ``Grid`` holding nothing beyond the minimal UGRID topology."""
        return ux.Grid.from_topology(*self.topology)

    def teardown(self, resolution, *args, **kwargs):
        # Cleared so a per-benchmark setup raises rather than quietly measuring
        # a stale grid.
        MinimalGridBenchmark.active_grid = None
        del self.uxgrid
        del self.topology


def _prerequisite_setup(connectivity):
    """Builds a per-benchmark ``setup`` that puts ``connectivity``'s
    prerequisites in place before the clock starts.

    asv collects ``setup`` from the benchmark function as well as the class and
    runs the class one first, but passes neither the instance, hence the handover
    through ``MinimalGridBenchmark.active_grid``.
    """

    def setup(resolution, *args, **kwargs):
        _build_prerequisites(MinimalGridBenchmark.active_grid, connectivity)

    return setup


class Connectivity(MinimalGridBenchmark):
    """Time to construct each connectivity variable.

    Prerequisites are built during ``setup``, so a sample times the one routine
    that produces that variable rather than the whole chain rooted at it --
    matching how :class:`ConnectivityTracemalloc` attributes memory.
    """

    number = 1
    warmup_time = 0

    def time_n_nodes_per_face(self, resolution):
        _ = self.uxgrid.n_nodes_per_face.compute()

    time_n_nodes_per_face.setup = _prerequisite_setup("n_nodes_per_face")

    def time_face_node(self, resolution):
        _ = self.uxgrid.face_node_connectivity.compute()

    time_face_node.setup = _prerequisite_setup("face_node_connectivity")

    def time_edge_node(self, resolution):
        _ = self.uxgrid.edge_node_connectivity.compute()

    time_edge_node.setup = _prerequisite_setup("edge_node_connectivity")

#   TODO: Not yet supported?
#   def time_node_node(self, resolution):
#       _ = self.uxgrid.node_node_connectivity

    def time_face_edge(self, resolution):
        _ = self.uxgrid.face_edge_connectivity.compute()

    time_face_edge.setup = _prerequisite_setup("face_edge_connectivity")

#   TODO: Not yet supported?
#   def time_edge_edge(self, resolution):
#        _ = self.uxgrid.edge_edge_connectivity

    def time_node_edge(self, resolution):
        _ = self.uxgrid.node_edge_connectivity.compute()

    time_node_edge.setup = _prerequisite_setup("node_edge_connectivity")

    def time_face_face(self, resolution):
        _ = self.uxgrid.face_face_connectivity.compute()

    time_face_face.setup = _prerequisite_setup("face_face_connectivity")

    def time_edge_face(self, resolution):
        _ = self.uxgrid.edge_face_connectivity.compute()

    time_edge_face.setup = _prerequisite_setup("edge_face_connectivity")

    def time_node_face(self, resolution):
        _ = self.uxgrid.node_face_connectivity.compute()

    time_node_face.setup = _prerequisite_setup("node_face_connectivity")


class ConnectivityTracemalloc(MinimalGridBenchmark):
    """Peak memory of each connectivity routine on its own.

    The transient high-water allocation of the construction routine, with the
    ~245MB the process already holds excluded.
    """

    unit = "bytes"

    def _peak_building(self, name):
        """Peak allocation of ``name``'s own construction routine."""
        uxgrid = _build_prerequisites(self.minimal_grid(), name)
        return peak_allocated(lambda: getattr(uxgrid, name).compute())

    def track_peakmem_n_nodes_per_face(self, resolution):
        return self._peak_building("n_nodes_per_face")

    def track_peakmem_face_node(self, resolution):
        return self._peak_building("face_node_connectivity")

    def track_peakmem_edge_node(self, resolution):
        return self._peak_building("edge_node_connectivity")

    def track_peakmem_face_edge(self, resolution):
        return self._peak_building("face_edge_connectivity")

    def track_peakmem_node_edge(self, resolution):
        return self._peak_building("node_edge_connectivity")

    def track_peakmem_face_face(self, resolution):
        return self._peak_building("face_face_connectivity")

    def track_peakmem_edge_face(self, resolution):
        return self._peak_building("edge_face_connectivity")

    def track_peakmem_node_face(self, resolution):
        return self._peak_building("node_face_connectivity")


class ConnectivityChainTracemalloc(MinimalGridBenchmark):
    """Peak memory of the whole chain rooted at each connectivity variable.

    Same instrument as :class:`ConnectivityTracemalloc` -- what the build
    allocates, with the ~245MB the process already holds excluded -- but wider
    in scope: no prerequisites are put in place beforehand, so a sample covers
    everything the variable pulls in, not just the routine that produces it.

    The two series coincide for ``n_nodes_per_face``, ``face_node_connectivity``
    and ``node_face_connectivity``, which build straight off the minimal
    topology; elsewhere the gap between them is what the prerequisites cost.
    """

    unit = "bytes"

    def _peak_chain(self, name):
        """Peak allocation of building ``name`` and everything it rests on."""
        uxgrid = self.minimal_grid()
        return peak_allocated(lambda: getattr(uxgrid, name).compute())

    def track_peakmem_n_nodes_per_face(self, resolution):
        return self._peak_chain("n_nodes_per_face")

    def track_peakmem_face_node(self, resolution):
        return self._peak_chain("face_node_connectivity")

    def track_peakmem_edge_node(self, resolution):
        return self._peak_chain("edge_node_connectivity")

    def track_peakmem_face_edge(self, resolution):
        return self._peak_chain("face_edge_connectivity")

    def track_peakmem_node_edge(self, resolution):
        return self._peak_chain("node_edge_connectivity")

    def track_peakmem_face_face(self, resolution):
        return self._peak_chain("face_face_connectivity")

    def track_peakmem_edge_face(self, resolution):
        return self._peak_chain("edge_face_connectivity")

    def track_peakmem_node_face(self, resolution):
        return self._peak_chain("node_face_connectivity")
