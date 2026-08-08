import os
import tracemalloc
import urllib.request
from pathlib import Path

import numpy as np

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

# What each variable needs in place before its own construction routine can run,
# read off the ``_populate_*`` functions in ``uxarray/grid/connectivity.py``.
# Only direct prerequisites are listed: accessing one builds its own in turn.
CONNECTIVITY_PREREQUISITES = {
    "n_nodes_per_face": (),
    "face_node_connectivity": (),
    "edge_node_connectivity": ("n_nodes_per_face",),
    # ``_populate_edge_node_connectivity`` writes ``face_edge_connectivity`` out
    # alongside its own variable, so this one costs nothing once it has run.
    "face_edge_connectivity": ("edge_node_connectivity",),
    "node_edge_connectivity": ("edge_node_connectivity",),
    "face_face_connectivity": ("edge_face_connectivity",),
    "edge_face_connectivity": ("face_edge_connectivity",),
    "node_face_connectivity": (),
}


def _build_prerequisites(uxgrid, connectivity):
    """Puts ``connectivity``'s prerequisites in place, so that what follows
    measures the one construction routine rather than the whole chain rooted at
    it."""
    for prerequisite in CONNECTIVITY_PREREQUISITES[connectivity]:
        getattr(uxgrid, prerequisite)
    return uxgrid


_numba_warmed_up = False

def _warmup():
    """Compiles the Numba kernels backing each connectivity variable.

    ``_build_node_edge_connectivity`` is not disk-cached, so a fresh benchmark
    process would otherwise charge ~240ms of JIT compilation to whichever sample
    happened to touch it first.
    """
    global _numba_warmed_up
    if _numba_warmed_up:
        return
    # The resolution only affects how long the kernels run, not which signatures get
    # compiled, so the coarsest grid is enough to warm any of them.
    uxgrid = ux.Grid.from_topology(*_source_topology(GridBenchmark.params[0][0]))
    for name in CONNECTIVITY_NAMES:
        getattr(uxgrid, name)
    _numba_warmed_up = True


_topology_cache = {}


def _source_topology(resolution):
    """The minimal UGRID topology for ``resolution``, read once per process.

    The benchmark grids are MPAS meshes, which carry every connectivity variable
    on disk. Reading one would measure the MPAS parser rather than the
    construction routines, so the grid is reduced to the minimal UGRID topology
    and each variable is left to be built on demand.

    Cached because asv re-runs ``setup`` between timing repeats: at the dyamond
    resolutions re-reading the source grid costs orders of magnitude more than
    the sample it precedes.
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

    # The default ``benchmark_timeout`` is not enough to build a connectivity
    # variable on a 3.75km grid.
    timeout = 1200

    def setup(self, resolution, *args, **kwargs):
        self.topology = _source_topology(resolution)

        _warmup()
        self.uxgrid = self.minimal_grid()
        MinimalGridBenchmark.active_grid = self.uxgrid

    def minimal_grid(self):
        """Mints a ``Grid`` holding nothing beyond the minimal UGRID topology."""
        return ux.Grid.from_topology(*self.topology)

    def teardown(self, resolution, *args, **kwargs):
        # Cleared so a per-benchmark setup can never reach a stale grid: it
        # would quietly measure the wrong thing, where this raises instead.
        MinimalGridBenchmark.active_grid = None
        del self.uxgrid
        del self.topology


def _prerequisite_setup(connectivity):
    """Builds a per-benchmark ``setup`` that puts ``connectivity``'s
    prerequisites in place before the clock starts.

    asv collects ``setup`` from the benchmark function as well as from the
    class, and runs the class one first, but it calls neither with the instance
    -- only with the parameters. Hence the handover through
    ``MinimalGridBenchmark.active_grid``, which the class ``setup`` has just
    filled in. One benchmark runs per process, so there is nothing to collide
    with.
    """

    def setup(resolution, *args, **kwargs):
        _build_prerequisites(MinimalGridBenchmark.active_grid, connectivity)

    return setup


class Connectivity(MinimalGridBenchmark):
    """Time to construct each connectivity variable.

    Each variable's prerequisites are built during ``setup``, so a sample times
    the one construction routine that produces that variable rather than the
    whole chain rooted at it -- matching how
    :class:`ConnectivityPeakAlloc` attributes memory.
    """

    # Each connectivity variable is cached in ``Grid._ds`` once constructed, so a
    # sample may only contain a single call; otherwise every call but the first
    # would time a dictionary lookup.
    number = 1

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


def _peak_allocated(build):
    """Bytes held at the high-water point of ``build``, counting only what it
    allocated itself."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        build()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


class ConnectivityPeakAlloc(MinimalGridBenchmark):
    """Peak memory of each connectivity routine on its own.

    Reports the transient high-water allocation of the construction routine,
    with the ~245MB the process is already holding subtracted out.
    """

    # Applies to every ``track_*`` in the class; asv resolves benchmark
    # attributes from the instance when the function does not carry them.
    unit = "bytes"

    def _peak_building(self, name):
        """Peak allocation of ``name``'s own construction routine."""
        uxgrid = _build_prerequisites(self.minimal_grid(), name)
        return _peak_allocated(lambda: getattr(uxgrid, name).compute())

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


def _save_topology(uxgrid, npz_path):
    """Writes the minimal UGRID topology of ``uxgrid`` out to ``npz_path``."""
    np.savez(
        npz_path,
        node_lon=uxgrid.node_lon.data,
        node_lat=uxgrid.node_lat.data,
        face_node_connectivity=uxgrid.face_node_connectivity.data,
    )


def _load_topology(npz_path):
    """Builds a ``Grid`` holding nothing beyond the minimal UGRID topology."""
    with np.load(npz_path) as topology:
        return ux.Grid.from_topology(
            topology["node_lon"],
            topology["node_lat"],
            topology["face_node_connectivity"],
        )


class ConnectivityPeakMem:
    """Peak resident memory of the process while constructing each connectivity
    variable.

    Unlike :class:`ConnectivityPeakAlloc`, no prerequisites are built during
    ``setup``, so a sample covers the whole chain rooted at that variable --
    which is what the resident footprint of asking for it actually costs.
    """

    # Declared rather than inherited from ``MinimalGridBenchmark`` -- only the
    # parameterization is shared, not the ``setup`` that opens a grid in the
    # process being measured.
    param_names = GridBenchmark.param_names
    params = GridBenchmark.params
    timeout = 1200

    def setup_cache(self):
        # asv runs this in its own process and passes the return value back as
        # the leading argument of ``setup`` and of each benchmark, so nothing
        # allocated here counts towards the samples.
        topology_paths = {}
        for resolution in self.params[0]:
            npz_path = os.path.abspath(f"topology_{resolution}.npz")
            _save_topology(ux.open_grid(file_path_dict[resolution]), npz_path)
            topology_paths[resolution] = npz_path

        # Being a separate process, this only reaches the benchmarks through
        # Numba's on-disk cache -- which is the point, since a compilation it
        # saves is one that would otherwise land inside a measured region.
        _warmup()

        return topology_paths

    # Reading every grid in ``file_path_dict`` exceeds the default
    # ``benchmark_timeout`` once the Glade paths are available.
    setup_cache.timeout = 1800

    def setup(self, topology_paths, resolution):
        # Each connectivity variable is cached in ``Grid._ds`` once constructed,
        # so the measured call needs a ``Grid`` that does not hold it yet.
        self.uxgrid = _load_topology(topology_paths[resolution])

    def teardown(self, topology_paths, resolution):
        del self.uxgrid

    def peakmem_n_nodes_per_face(self, topology_paths, resolution):
        _ = self.uxgrid.n_nodes_per_face.compute()

    def peakmem_face_node(self, topology_paths, resolution):
        _ = self.uxgrid.face_node_connectivity.compute()

    def peakmem_edge_node(self, topology_paths, resolution):
        _ = self.uxgrid.edge_node_connectivity.compute()

    def peakmem_face_edge(self, topology_paths, resolution):
        _ = self.uxgrid.face_edge_connectivity.compute()

    def peakmem_node_edge(self, topology_paths, resolution):
        _ = self.uxgrid.node_edge_connectivity.compute()

    def peakmem_face_face(self, topology_paths, resolution):
        _ = self.uxgrid.face_face_connectivity.compute()

    def peakmem_edge_face(self, topology_paths, resolution):
        _ = self.uxgrid.edge_face_connectivity.compute()

    def peakmem_node_face(self, topology_paths, resolution):
        _ = self.uxgrid.node_face_connectivity.compute()
