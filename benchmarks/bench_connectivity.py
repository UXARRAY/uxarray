import uxarray as ux

from .helpers._fixtures import (
    ALL_RESOLUTIONS,
    GRIDS_BY_RESOLUTION,
    CachedFixtures,
    cached_topology,
    preload_topologies,
)
from .helpers._peakmem import numba_threads, peak_allocated
from .helpers._warmup import warm_in_parent

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

    Run once in the parent by ``_warm_parent`` below, so every forked benchmark
    inherits the compiled dispatchers. Loading them off Numba's on-disk cache
    would otherwise charge both time and allocations to whichever sample touched
    a kernel first, which matters for ``track_peakmem_*`` as much as timing.
    """
    global _numba_warmed_up
    if _numba_warmed_up:
        return
    # Resolution affects how long the kernels run, not which signatures compile.
    uxgrid = ux.Grid.from_topology(*cached_topology(GRIDS_BY_RESOLUTION[ALL_RESOLUTIONS[0]]))
    for name in CONNECTIVITY_NAMES:
        getattr(uxgrid, name)
    _numba_warmed_up = True


class MinimalGridBenchmark(CachedFixtures):
    """Template for benchmarks that construct connectivity variables on demand.

    Holds a ``Grid`` carrying nothing but the minimal UGRID topology, plus the
    topology needed to mint further ones, and leaves the Numba kernels compiled.

    Never touches the full cached grid: the topology fixture is all it needs,
    and it is preloaded in the parent so each fork inherits it rather than
    reading it again.
    """

    param_names = ['resolution', ]
    params = [ALL_RESOLUTIONS, ]

    # Handover slot for ``_prerequisite_setup``; see its docstring.
    active_grid = None

    # asv's 60s default is not enough to build a connectivity variable at 3.75km.
    timeout = 1800

    def setup(self, resolution, *args, **kwargs):
        # The benchmark grids are MPAS meshes, which carry every connectivity
        # variable on disk. Reading one would measure the MPAS parser rather than
        # the construction routines, so this takes the minimal UGRID topology
        # fixture and lets each variable be built on demand.
        self.topology = self.cached_topology(GRIDS_BY_RESOLUTION[resolution])

        # A no-op once the module-level warm below has run; kept so the class is
        # still correct if that ever goes away.
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

    The transient high-water allocation of the construction routine, with
    whatever the process already holds -- the inherited topology fixtures
    included -- excluded.

    ``edge_node_connectivity`` and ``face_edge_connectivity`` are built by
    ``parallel=True`` kernels, hence the pinning -- see
    :func:`~benchmarks.helpers._peakmem.numba_threads`.
    """

    unit = "bytes"

    def _peak_building(self, name):
        """Peak allocation of ``name``'s own construction routine."""
        uxgrid = _build_prerequisites(self.minimal_grid(), name)
        with numba_threads(1):
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
    allocates, with what the process already holds excluded -- but wider
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
        with numba_threads(1):
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


# Compiled at import rather than in ``setup``. ASV imports the suite once and forks
# every benchmark from that parent, so kernels compiled here are inherited by all
# of them. Only safe while the connectivity kernels are serial
def _warm_parent():
    _warmup()
    # And the topologies themselves...
    preload_topologies(GRIDS_BY_RESOLUTION[res] for res in ALL_RESOLUTIONS)


warm_in_parent(_warm_parent, "the connectivity kernels")
