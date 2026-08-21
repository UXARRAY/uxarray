import uxarray as ux

from .helpers._fixtures import GRIDS_BY_FORMAT, CachedFixtures
from .helpers._memsize import grid_nbytes
from .helpers._peakmem import numba_threads, peak_allocated, subprocess_peak_rss

# One grid per reader, from the shared registry. ``mpas`` here is the same mesh
# the oQU ``480km`` benchmarks use, through the copy in the repo.
grid_quad_hex = GRIDS_BY_FORMAT["ugrid-quad-hexagon"]
grid_geoflow = GRIDS_BY_FORMAT["ugrid-geoflow"]
grid_scrip = GRIDS_BY_FORMAT["scrip-outCSne8"]
grid_mpas = GRIDS_BY_FORMAT["mpas-oQU480"]

class FaceBounds(CachedFixtures):

    params = [grid_quad_hex, grid_geoflow, grid_scrip, grid_mpas]

    number = 1
    warmup_time = 0

    def setup(self, grid_path):
        # Warmed on the smallest grid in ``params`` so the njit kernels are
        # compiled before anything is measured. ``track_peakmem_*`` would
        # otherwise charge the first sample for loading them off numba's disk
        # cache, which inflates the reported peak by ~3%.
        self.cached_grid(grid_quad_hex).bounds
        self.uxgrid = self.cached_grid(grid_path)

    def teardown(self, n):
        del self.uxgrid

    def time_face_bounds(self, grid_path):
        """Time to obtain ``Grid.face_bounds``"""
        self.uxgrid.bounds

    def track_nbytes_face_bounds(self, grid_path):
        """Size of the materialized ``Grid.face_bounds`` array."""
        return self.uxgrid.bounds.nbytes

    track_nbytes_face_bounds.unit = "bytes"

    def track_nbytes_grid_with_bounds(self, grid_path):
        """Grid footprint after populating bounds -- catches cached arrays that
        ``bounds`` adds to the ``Grid`` beyond the returned array itself."""
        self.uxgrid.bounds
        return grid_nbytes(self.uxgrid)

    track_nbytes_grid_with_bounds.unit = "bytes"

    def track_peakmem_face_bounds(self, grid_path):
        """Transient high-water allocation of populating ``Grid.face_bounds``.

        The kernel behind ``bounds`` is ``parallel=True``, hence the pinning --
        see :func:`~benchmarks.helpers._peakmem.numba_threads`.
        """
        with numba_threads(1):
            return peak_allocated(lambda: self.uxgrid.bounds)

    track_peakmem_face_bounds.unit = "bytes"


class FaceBoundsColdStartRss:
    """Peak memory of a cold start: import uxarray, open a grid, get its bounds.

    Whole-process peak resident memory, not tracemalloc -- the ~250MB uxarray
    import is part of the number by design, because the cold start is the
    subject. For the cost of ``bounds`` alone see
    ``FaceBounds.track_peakmem_face_bounds``, which runs one to three orders of
    magnitude lower.

    Measured in a subprocess of its own rather than through asv's ``peakmem_*``,
    which reports ``ru_maxrss`` for the benchmark process. Under
    ``launch_method: forkserver`` that process is forked from an interpreter that
    has already imported the suite, so a ``peakmem_*`` here would be reporting a
    warm start plus whatever the parent was holding. A fresh interpreter is the
    only way to keep measuring what this benchmark is named for.
    """

    params = FaceBounds.params
    param_names = ["grid_path"]

    def setup_cache(self):
        """Compile the njit kernels before anything is measured.

        The subprocess inherits numba's on-disk cache, not this process's
        memory, so this keeps compilation out of the measured cold start.
        """
        for grid_path in self.params:
            ux.open_grid(grid_path).bounds

    setup_cache.timeout = 1800

    def track_peakmem_open_and_bounds(self, grid_path):
        return subprocess_peak_rss(
            f"import uxarray as ux; ux.open_grid({str(grid_path)!r}).bounds"
        )

    track_peakmem_open_and_bounds.unit = "bytes"
