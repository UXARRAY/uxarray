"""Regional subsetting of a grid that has not had its bounds computed.

``Grid.subset.bounding_box`` computes face bounds for the faces whose nodes
lie inside the box rather than for the whole mesh, so its cost should follow
the size of the region, not the grid (#1778). Each sample starts from a fresh
``Grid`` so that nothing cached by an earlier call is measured.
"""

import uxarray as ux

from .helpers._fixtures import OQU_GRIDS, OQU_RESOLUTIONS, CachedFixtures
from .helpers._peakmem import numba_threads, peak_allocated

# Texas, roughly: 0.3% of the sphere.
LON_BOUNDS = (-106.6, -93.5)
LAT_BOUNDS = (25.8, 36.5)


def _warm_kernels():
    """Compile the njit kernels on a small grid before anything is measured."""
    ux.Grid.from_healpix(2).subset.bounding_box(LON_BOUNDS, LAT_BOUNDS)


class BoundingBox(CachedFixtures):
    param_names = ["resolution"]
    params = [OQU_RESOLUTIONS]

    number = 1
    warmup_time = 0

    def setup(self, resolution):
        _warm_kernels()
        self.uxgrid = self.cached_grid(OQU_GRIDS[resolution])

    def teardown(self, resolution):
        del self.uxgrid

    def time_bounding_box(self, resolution):
        self.uxgrid.subset.bounding_box(LON_BOUNDS, LAT_BOUNDS)

    def track_peakmem_bounding_box(self, resolution):
        """Transient high-water allocation of the subset.

        Pinned to one numba thread because the bounds kernel is
        ``parallel=True``; see :func:`~benchmarks.helpers._peakmem.numba_threads`.
        """
        with numba_threads(1):
            return peak_allocated(
                lambda: self.uxgrid.subset.bounding_box(LON_BOUNDS, LAT_BOUNDS)
            )

    track_peakmem_bounding_box.unit = "bytes"


class BoundingBoxHealpix:
    """The same subset on HEALPix grids, whose size grows 4x per zoom level.

    Built in memory, so the mesh can be far larger than any file in the
    repository: zoom 8 is 786,432 faces.
    """

    param_names = ["zoom"]
    params = [[6, 8]]

    number = 1
    warmup_time = 0

    def setup(self, zoom):
        _warm_kernels()
        self.uxgrid = ux.Grid.from_healpix(zoom)
        # Materialize what the subset reads, so only the subset is measured.
        self.uxgrid.face_node_connectivity.values
        self.uxgrid.node_lon.values
        self.uxgrid.node_lat.values

    def teardown(self, zoom):
        del self.uxgrid

    def time_bounding_box(self, zoom):
        self.uxgrid.subset.bounding_box(LON_BOUNDS, LAT_BOUNDS)

    def track_peakmem_bounding_box(self, zoom):
        with numba_threads(1):
            return peak_allocated(
                lambda: self.uxgrid.subset.bounding_box(LON_BOUNDS, LAT_BOUNDS)
            )

    track_peakmem_bounding_box.unit = "bytes"
