from asv_runner.benchmarks.mark import skip_benchmark_if, timeout_class_at

import uxarray as ux

from .helpers._fixtures import DYAMOND_AVAILABLE, DYAMOND_GRIDS, CachedFixtures

# Paths, and the question of whether this machine can see them, both come from
# ``helpers._fixtures`` -- ``bench_connectivity`` asks for the same four grids.
grid_path_dict = DYAMOND_GRIDS


class BaseGridBenchmark(CachedFixtures):
    """Base class for Grid Benchmarks across the four supported resolutions
    (30km, 15km, 7.5km, 3.75km)"""
    param_names = ['resolution']
    params = [list(DYAMOND_GRIDS), ]

    def setup(self, resolution, **kwargs):
        # The cached grid, not a fresh read: what these benchmarks measure is
        # ``bounds`` and ``to_geodataframe``, not the MPAS reader. ``OpenGrid``
        # below is the one that measures reading, and it opens the real file.
        self.uxgrid = self.cached_grid(grid_path_dict[resolution])

    def teardown(self, resolution, **kwargs):
        del self.uxgrid

@timeout_class_at(1200)
class OpenGrid:
    param_names = ['resolution']
    params = [list(DYAMOND_GRIDS), ]

    @skip_benchmark_if(not DYAMOND_AVAILABLE)
    def time_open_grid(self, resolution):
        _ = ux.open_grid(grid_path_dict[resolution])


@timeout_class_at(1200)
class Bounds(BaseGridBenchmark):
    @skip_benchmark_if(not DYAMOND_AVAILABLE)
    def time_bounds(self, resolution):
        _ = self.uxgrid.bounds

@timeout_class_at(1200)
class GeoDataFrame(BaseGridBenchmark):
    @skip_benchmark_if(not DYAMOND_AVAILABLE)
    def time_to_geodataframe(self, resolution):
        self.uxgrid.to_geodataframe(exclude_antimeridian=True)
