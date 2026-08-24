import uxarray as ux

from .helpers._fixtures import (
    ALL_RESOLUTIONS,
    GRIDS_BY_RESOLUTION,
    CachedFixtures,
    cached_topology,
    preload_topologies,
)
from .helpers._warmup import warm_in_parent


class GridBenchmark(CachedFixtures):
    """Class used as a template for benchmarks requiring a ``Grid`` in this
    module across both resolutions."""

    param_names = ['resolution', ]
    params = [ALL_RESOLUTIONS, ]
    timeout = 1200

    def setup(self, resolution, *args, **kwargs):
        self.uxgrid = self.cached_grid(GRIDS_BY_RESOLUTION[resolution])

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

def _warmup():
    """Compiles the Numba kernels backing each connectivity variable.

    ``_build_node_edge_connectivity`` is ``@njit`` without ``cache=True``, so a
    fresh benchmark process would otherwise charge ~240ms of JIT compilation to
    whichever sample happened to touch it first.
    """
    global _numba_warmed_up
    if _numba_warmed_up:
        return
    uxgrid = ux.Grid.from_topology(*cached_topology(GRIDS_BY_RESOLUTION[ALL_RESOLUTIONS[0]]))
    for name in CONNECTIVITY_NAMES:
        getattr(uxgrid, name)
    _numba_warmed_up = True


class Connectivity(GridBenchmark):
    # connectivity is cached in ``Grid._ds`` on construction, so only run them once
    number = 1

    def setup(self, resolution, *args, **kwargs):
        # The benchmark grids are MPAS meshes, which carry every connectivity
        # variable on disk. Reading one would time the MPAS parser rather than
        # the construction routines, so this takes the minimal UGRID topology
        # fixture and lets each variable be built on demand.
        self.topology = self.cached_topology(GRIDS_BY_RESOLUTION[resolution])

        # A no-op once the module-level warm below has run; kept so the class is
        # still correct if that ever goes away.
        _warmup()
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


# Compiled at import rather than in ``setup``. ASV imports the suite once and forks
# every benchmark from that parent, so kernels compiled here are inherited by all
# of them. Only safe while the connectivity kernels are serial
def _warm_parent():
    _warmup()
    # And, if asked, the topologies themselves...
    preload_topologies(GRIDS_BY_RESOLUTION[res] for res in ALL_RESOLUTIONS)


warm_in_parent(_warm_parent, "the connectivity kernels")
