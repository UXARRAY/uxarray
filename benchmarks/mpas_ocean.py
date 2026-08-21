import numpy as np

import uxarray as ux

from .helpers._fixtures import (
    OQU_DATASETS,
    OQU_GRIDS,
    OQU_RESOLUTIONS,
    CachedFixtures,
)
from .helpers._memsize import grid_nbytes
from .helpers._peakmem import numba_threads, peak_allocated, subprocess_peak_rss

data_var = 'bottomDepth'

# Paths, and fetching the files in the first place, both live in
# ``helpers._fixtures`` now -- ``bench_connectivity`` draws the same grids from it.
file_path_dict = OQU_DATASETS


class DatasetBenchmark(CachedFixtures):
    """Class used as a template for benchmarks requiring a ``UxDataset`` in
    this module across both resolutions.

    The dataset comes from the fixture cache rather than a fresh
    ``open_dataset``: every benchmark below measures an algorithm over the mesh,
    not the reader that produced it. The fixture is what the reader produced,
    connectivity and ``face_areas`` included, so nothing here silently starts
    measuring construction that used to come off disk.
    """
    param_names = ['resolution', ]
    params = [OQU_RESOLUTIONS, ]

    def setup(self, resolution, *args, **kwargs):
        self.uxds = self.cached_dataset(*file_path_dict[resolution])

    def teardown(self, resolution, *args, **kwargs):
        del self.uxds


class GridBenchmark(CachedFixtures):
    """Class used as a template for benchmarks requiring a ``Grid`` in this
    module across both resolutions."""
    param_names = ['resolution', ]
    params = [OQU_RESOLUTIONS, ]

    def setup(self, resolution, *args, **kwargs):
        self.uxgrid = self.cached_grid(file_path_dict[resolution][0])

    def teardown(self, resolution, *args, **kwargs):
        del self.uxgrid


class FaceAreas(GridBenchmark):
    number = 1
    warmup_time = 0

    def setup(self, resolution, *args, **kwargs):
        # The coarsest grid, purely to compile the njit kernel
        _ = self.cached_grid(OQU_GRIDS[OQU_RESOLUTIONS[0]]).face_areas
        super().setup(resolution, *args, **kwargs)
        # MPAS meshes carry ``face_areas`` on disk and the fixture keeps it, so
        # it is dropped here to leave the computation to be measured. Safe to do
        # to a fixture: each handout is a fresh ``Grid`` over a shallow copy.
        self.uxgrid._ds = self.uxgrid._ds.drop_vars("face_areas", errors="ignore")

    def time_face_areas(self, resolution):
        _ = self.uxgrid.face_areas

    def track_nbytes_face_areas(self, resolution):
        """Size of the materialized ``Grid.face_areas`` array."""
        return self.uxgrid.face_areas.nbytes

    track_nbytes_face_areas.unit = "bytes"

    def track_peakmem_face_areas(self, resolution):
        """Transient high-water allocation of computing ``Grid.face_areas``."""
        with numba_threads(1):
            return peak_allocated(lambda: self.uxgrid.face_areas)

    track_peakmem_face_areas.unit = "bytes"


class Gradient(DatasetBenchmark):
    def setup(self, resolution, *args, **kwargs):
        super().setup(resolution, *args, **kwargs)
        # Compiles the gradient kernels on the coarsest grid
        _ = self.cached_dataset(*file_path_dict[OQU_RESOLUTIONS[0]])[data_var].gradient()

    def time_gradient(self, resolution):
        self.uxds[data_var].gradient()

    def track_nbytes_gradient(self, resolution):
        """Size of the gradient result."""
        return self.uxds[data_var].gradient().nbytes

    track_nbytes_gradient.unit = "bytes"

    def track_peakmem_gradient(self, resolution):
        """Transient high-water allocation of taking a gradient.

        The kernel behind ``gradient`` is ``parallel=True``, hence the pinning
        """
        with numba_threads(1):
            return peak_allocated(lambda: self.uxds[data_var].gradient())

    track_peakmem_gradient.unit = "bytes"


class Integrate(DatasetBenchmark):

    def time_integrate(self, resolution):
        self.uxds[data_var].integrate()

    def track_nbytes_integrate(self, resolution):
        """Grid footprint after integrating."""
        self.uxds[data_var].integrate()
        return grid_nbytes(self.uxds.uxgrid)

    track_nbytes_integrate.unit = "bytes"


class GradientColdStartRss:
    """Peak memory of a cold start: import uxarray, open a dataset, take a gradient.

    Whole-process peak resident memory, not tracemalloc -- the ~226MB uxarray
    import is part of the number by design, because the cold start is the
    subject. For the gradient's own transient cost see
    ``Gradient.track_peakmem_gradient``, which runs one to three orders of
    magnitude lower.

    Measured in a subprocess of its own rather than through asv's ``peakmem_*``,
    which reports ``ru_maxrss`` for the benchmark process. Under
    ``launch_method: forkserver`` that process is forked from an interpreter
    that has already imported the suite, so ``peakmem_*`` would report a warm
    start plus whatever the parent held. A fresh interpreter is the only way to
    keep measuring the thing this benchmark is named for.
    """

    param_names = ["resolution"]
    params = [OQU_RESOLUTIONS]

    def setup_cache(self):
        """Compile the njit kernels before anything is measured.

        The subprocess inherits numba's on-disk cache rather than this process's
        memory, so this keeps compilation out of the measured cold start.
        """
        for resolution in self.params[0]:
            grid, data = file_path_dict[resolution]
            ux.open_dataset(grid, data)[data_var].gradient()

    setup_cache.timeout = 1800

    def track_peakmem_gradient(self, resolution):
        grid, data = file_path_dict[resolution]
        return subprocess_peak_rss(
            "import uxarray as ux\n"
            f"uxds = ux.open_dataset({str(grid)!r}, {str(data)!r})\n"
            f"uxds[{data_var!r}].gradient()\n"
        )

    track_peakmem_gradient.unit = "bytes"


class GeoDataFrame(DatasetBenchmark):
    param_names = DatasetBenchmark.param_names + ['exclude_antimeridian']
    params = DatasetBenchmark.params + [[True, False]]

    def time_to_geodataframe(self, resolution, exclude_antimeridian):
        self.uxds[data_var].to_geodataframe(exclude_antimeridian=exclude_antimeridian)


class ConnectivityConstruction(DatasetBenchmark):
    def time_n_nodes_per_face(self, resolution):
        _ = self.uxds.uxgrid.n_nodes_per_face

    def time_face_face_connectivity(self, resolution):
        ux.grid.connectivity._populate_face_face_connectivity(self.uxds.uxgrid)


class MatplotlibConversion(DatasetBenchmark):
    param_names = DatasetBenchmark.param_names + ['periodic_elements']
    params = DatasetBenchmark.params + [['include', 'exclude', 'split']]

    def time_dataarray_to_polycollection(self, resolution, periodic_elements):
        self.uxds[data_var].to_polycollection()


class ConstructTreeStructures(DatasetBenchmark):

    def time_kd_tree(self, resolution):
        self.uxds.uxgrid.get_kd_tree()

    def time_ball_tree(self, resolution):
        self.uxds.uxgrid.get_ball_tree()


class RemapDownsample(CachedFixtures):

    def setup(self):
        self.uxds_120 = self.cached_dataset(*file_path_dict['120km'])
        self.uxds_480 = self.cached_dataset(*file_path_dict['480km'])

    def teardown(self):
        del self.uxds_120, self.uxds_480

    def time_nearest_neighbor_remapping(self):
        self.uxds_120["bottomDepth"].remap.nearest_neighbor(self.uxds_480.uxgrid)

    def time_inverse_distance_weighted_remapping(self):
        self.uxds_120["bottomDepth"].remap.inverse_distance_weighted(self.uxds_480.uxgrid)

    def time_bilinear_remapping(self):
        self.uxds_120["bottomDepth"].remap.bilinear(self.uxds_480.uxgrid)

class RemapUpsample(CachedFixtures):

    def setup(self):
        self.uxds_120 = self.cached_dataset(*file_path_dict['120km'])
        self.uxds_480 = self.cached_dataset(*file_path_dict['480km'])

    def teardown(self):
        del self.uxds_120, self.uxds_480

    def time_nearest_neighbor_remapping(self):
        self.uxds_480["bottomDepth"].remap.nearest_neighbor(self.uxds_120.uxgrid)

    def time_inverse_distance_weighted_remapping(self):
        self.uxds_480["bottomDepth"].remap.inverse_distance_weighted(self.uxds_120.uxgrid)

    def time_bilinear_remapping(self):
        self.uxds_480["bottomDepth"].remap.bilinear(self.uxds_120.uxgrid)


class HoleEdgeIndices(DatasetBenchmark):
    def time_construct_hole_edge_indices(self, resolution):
        ux.grid.geometry._construct_boundary_edge_indices(self.uxds.uxgrid.edge_face_connectivity)


class DualMesh(DatasetBenchmark):
    def time_dual_mesh_construction(self, resolution):
        self.uxds.uxgrid.get_dual()


class ConstructFaceLatLon(GridBenchmark):
    def time_welzl(self, resolution):
        self.uxgrid.construct_face_centers(method='welzl')

    def time_cartesian_averaging(self, resolution):
        self.uxgrid.construct_face_centers(method='cartesian average')


class CheckNorm(CachedFixtures):
    param_names = ['resolution']
    params = OQU_RESOLUTIONS

    def setup(self, resolution):
        self.uxgrid = self.cached_grid(file_path_dict[resolution][0])

    def teardown(self, resolution):
        del self.uxgrid

    def time_check_norm(self, resolution):
        from uxarray.grid.validation import _check_normalization
        _check_normalization(self.uxgrid)

class CrossSections(DatasetBenchmark):
    param_names = DatasetBenchmark.param_names + ['n_lat']
    params = DatasetBenchmark.params + [[1, 2, 4]]

    def setup(self, resolution, lat_step):
        self.uxgrid = self.cached_grid(file_path_dict[resolution][0])
        self.uxgrid.normalize_cartesian_coordinates()
        self.lats = np.arange(-45, 45, lat_step)
        _ = self.uxgrid.bounds

    def teardown(self, resolution, lat_step):
        del self.uxgrid

    def time_const_lat(self, resolution, lat_step):
        for lat in self.lats:
            self.uxgrid.cross_section.constant_latitude(lat)


class PointInPolygon(CachedFixtures):
    param_names = ['resolution']
    params = OQU_RESOLUTIONS

    def setup(self, resolution):
        self.uxgrid = self.cached_grid(file_path_dict[resolution][0])
        self.uxgrid.normalize_cartesian_coordinates()

        # Construct variables needed to ensure that the benchmark doesn't measure construction time
        _ = self.uxgrid.face_edge_connectivity
        _ = self.uxgrid.face_x.values
        _ = self.uxgrid.face_lon.values

        point = np.array([0.0, 0.0, 1.0])
        res = self.uxgrid.get_faces_containing_point(point)

        self.point_xyz = np.array([self.uxgrid.face_x[0].values, self.uxgrid.face_y[0].values, self.uxgrid.face_z[0].values], dtype=np.float64)
        self.point_lonlat = np.array([self.uxgrid.face_lon[0].values, self.uxgrid.face_lat.values[0]], dtype=np.float64)

    def teardown(self, resolution):
        del self.uxgrid

    def time_face_search_xyz(self, resolution):
        self.uxgrid.get_faces_containing_point(self.point_xyz)

    def time_face_search_lonlat(self, resolution):
        self.uxgrid.get_faces_containing_point(self.point_lonlat)


class ZonalAverage(DatasetBenchmark):
    def setup(self, resolution, *args, **kwargs):
        super().setup(resolution, *args, **kwargs)
        bounds = self.uxds.uxgrid.bounds

    def time_zonal_average(self, resolution):
        lat_step = 10
        self.uxds['bottomDepth'].zonal_mean(lat=(-45, 45, lat_step))


class ZonalAveragePeakMem:
    """Peak memory of a cold-start non-conservative zonal-mean sweep.

    A fresh interpreter per sample, for the reason spelled out in
    :class:`GradientColdStartRss`: the cold start is the subject, and a forked
    benchmark process no longer has one.
    """

    param_names = ["resolution"]
    params = [OQU_RESOLUTIONS]

    def setup_cache(self):
        """Compile the njit kernels before anything is measured."""
        for resolution in self.params[0]:
            grid, data = file_path_dict[resolution]
            uxds = ux.open_dataset(grid, data)
            uxds.uxgrid.bounds
            uxds[data_var].zonal_mean(lat=(-45, 45, 10))

    setup_cache.timeout = 1800

    def track_peakmem_zonal_average(self, resolution):
        grid, data = file_path_dict[resolution]
        return subprocess_peak_rss(
            "import uxarray as ux\n"
            f"uxds = ux.open_dataset({str(grid)!r}, {str(data)!r})\n"
            "uxds.uxgrid.bounds\n"
            f"uxds[{data_var!r}].zonal_mean(lat=(-45, 45, 10))\n"
        )

    track_peakmem_zonal_average.unit = "bytes"


class CrossSectionsPeakMem:
    """Peak memory of a cold-start constant-latitude cross-section sweep.

    A fresh interpreter per sample, for the reason spelled out in
    :class:`GradientColdStartRss`.
    """

    param_names = ["resolution", "lat_step"]
    params = [OQU_RESOLUTIONS, [1, 2, 4]]

    def setup_cache(self):
        """Compile the njit kernels before anything is measured."""
        for resolution in self.params[0]:
            uxgrid = ux.open_grid(file_path_dict[resolution][0])
            uxgrid.normalize_cartesian_coordinates()
            uxgrid.bounds
            uxgrid.cross_section.constant_latitude(0.0)

    setup_cache.timeout = 1800

    def track_peakmem_const_lat(self, resolution, lat_step):
        grid = file_path_dict[resolution][0]
        return subprocess_peak_rss(
            "import numpy as np, uxarray as ux\n"
            f"uxgrid = ux.open_grid({str(grid)!r})\n"
            "uxgrid.normalize_cartesian_coordinates()\n"
            "uxgrid.bounds\n"
            f"for lat in np.arange(-45, 45, {lat_step}):\n"
            "    uxgrid.cross_section.constant_latitude(lat)\n"
        )

    track_peakmem_const_lat.unit = "bytes"
