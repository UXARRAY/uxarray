import os
import urllib.request
import warnings
from pathlib import Path

import numpy as np

import uxarray as ux
from uxarray.grid.neighbors import Neighborhood, _get_element_coords

from .helpers._memsize import grid_nbytes
from .helpers._peakmem import numba_threads, peak_allocated

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


file_path_dict = {"480km": [current_path / grid_filename_480, current_path / data_filename_480],
                  "120km": [current_path / grid_filename_120, current_path / data_filename_120]}



class DatasetBenchmark:
    """Class used as a template for benchmarks requiring a ``UxDataset`` in
    this module across both resolutions."""
    param_names = ['resolution', ]
    params = [['480km', '120km'], ]

    def setup(self, resolution, *args, **kwargs):
        self.uxds = ux.open_dataset(file_path_dict[resolution][0], file_path_dict[resolution][1])

    def teardown(self, resolution, *args, **kwargs):
        del self.uxds


class GridBenchmark:
    """Class used as a template for benchmarks requiring a ``Grid`` in this
    module across both resolutions."""
    param_names = ['resolution', ]
    params = [['480km', '120km'], ]

    def setup(self, resolution, *args, **kwargs):
        self.uxgrid = ux.open_grid(file_path_dict[resolution][0])

    def teardown(self, resolution, *args, **kwargs):
        del self.uxgrid


class FaceAreas(GridBenchmark):
    number = 1
    warmup_time = 0

    def setup(self, resolution, *args, **kwargs):
        # The coarsest grid, purely to compile the njit kernel
        warmup_grid = ux.open_grid(file_path_dict[self.params[0][0]][0])
        _ = warmup_grid.face_areas
        super().setup(resolution, *args, **kwargs)
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
        grid, data = file_path_dict[self.params[0][0]]
        _ = ux.open_dataset(grid, data)[data_var].gradient()

    def time_gradient(self, resolution):
        self.uxds[data_var].gradient()

    def track_nbytes_gradient(self, resolution):
        """Size of the gradient result."""
        return self.uxds[data_var].gradient().nbytes

    track_nbytes_gradient.unit = "bytes"

    def track_peakmem_gradient(self, resolution):
        """Transient high-water allocation of taking a gradient."""
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

    Whole-process ``ru_maxrss``, not tracemalloc -- the ~250MB uxarray import is
    part of the number by design, because the cold start is the subject. For the
    gradient's own transient cost see ``Gradient.track_peakmem_gradient``, which
    runs one to three orders of magnitude lower.
    """

    param_names = ["resolution"]
    params = [["480km", "120km"]]

    def setup_cache(self):
        """Compile the njit kernels before anything is measured."""
        for resolution in self.params[0]:
            grid, data = file_path_dict[resolution]
            ux.open_dataset(grid, data)[data_var].gradient()

    setup_cache.timeout = 1800

    def peakmem_gradient(self, resolution):
        grid, data = file_path_dict[resolution]
        ux.open_dataset(grid, data)[data_var].gradient()


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


class RemapDownsample:

    def setup(self):
        self.uxds_120 = ux.open_dataset(file_path_dict['120km'][0], file_path_dict['120km'][1])
        self.uxds_480 = ux.open_dataset(file_path_dict['480km'][0], file_path_dict['480km'][1])

    def teardown(self):
        del self.uxds_120, self.uxds_480

    def time_nearest_neighbor_remapping(self):
        self.uxds_120["bottomDepth"].remap.nearest_neighbor(self.uxds_480.uxgrid)

    def time_inverse_distance_weighted_remapping(self):
        self.uxds_120["bottomDepth"].remap.inverse_distance_weighted(self.uxds_480.uxgrid)

    def time_bilinear_remapping(self):
        self.uxds_120["bottomDepth"].remap.bilinear(self.uxds_480.uxgrid)

class RemapUpsample:

    def setup(self):
        self.uxds_120 = ux.open_dataset(file_path_dict['120km'][0], file_path_dict['120km'][1])
        self.uxds_480 = ux.open_dataset(file_path_dict['480km'][0], file_path_dict['480km'][1])

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


class CheckNorm:
    param_names = ['resolution']
    params = ['480km', '120km']

    def setup(self, resolution):
        self.uxgrid = ux.open_grid(file_path_dict[resolution][0])

    def teardown(self, resolution):
        del self.uxgrid

    def time_check_norm(self, resolution):
        from uxarray.grid.validation import _check_normalization
        _check_normalization(self.uxgrid)

class CrossSections(DatasetBenchmark):
    param_names = DatasetBenchmark.param_names + ['n_lat']
    params = DatasetBenchmark.params + [[1, 2, 4]]

    def setup(self, resolution, lat_step):
        self.uxgrid = ux.open_grid(file_path_dict[resolution][0])
        self.uxgrid.normalize_cartesian_coordinates()
        self.lats = np.arange(-45, 45, lat_step)
        _ = self.uxgrid.bounds

    def teardown(self, resolution, lat_step):
        del self.uxgrid

    def time_const_lat(self, resolution, lat_step):
        for lat in self.lats:
            self.uxgrid.cross_section.constant_latitude(lat)


class PointInPolygon:
    param_names = ['resolution']
    params = ['480km', '120km']

    def setup(self, resolution):
        self.uxgrid = ux.open_grid(file_path_dict[resolution][0])
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
        self.uxds = ux.open_dataset(file_path_dict[resolution][0], file_path_dict[resolution][1])
        bounds = self.uxds.uxgrid.bounds

    def time_zonal_average(self, resolution):
        lat_step = 10
        self.uxds['bottomDepth'].zonal_mean(lat=(-45, 45, lat_step))


class ZonalAveragePeakMem:
    """Peak memory of a cold-start non-conservative zonal-mean sweep."""

    param_names = ["resolution"]
    params = [["480km", "120km"]]

    def setup_cache(self):
        """Compile the njit kernels before anything is measured."""
        for resolution in self.params[0]:
            grid, data = file_path_dict[resolution]
            uxds = ux.open_dataset(grid, data)
            uxds.uxgrid.bounds
            uxds[data_var].zonal_mean(lat=(-45, 45, 10))

    def peakmem_zonal_average(self, resolution):
        grid, data = file_path_dict[resolution]
        uxds = ux.open_dataset(grid, data)
        uxds.uxgrid.bounds
        uxds[data_var].zonal_mean(lat=(-45, 45, 10))


class CrossSectionsPeakMem:
    """Peak memory of a cold-start constant-latitude cross-section sweep."""

    param_names = ["resolution", "lat_step"]
    params = [["480km", "120km"], [1, 2, 4]]

    def setup_cache(self):
        """Compile the njit kernels before anything is measured."""
        for resolution in self.params[0]:
            uxgrid = ux.open_grid(file_path_dict[resolution][0])
            uxgrid.normalize_cartesian_coordinates()
            uxgrid.bounds
            uxgrid.cross_section.constant_latitude(0.0)

    def peakmem_const_lat(self, resolution, lat_step):
        uxgrid = ux.open_grid(file_path_dict[resolution][0])
        uxgrid.normalize_cartesian_coordinates()
        uxgrid.bounds
        for lat in np.arange(-45, 45, lat_step):
            uxgrid.cross_section.constant_latitude(lat)


class NeighborhoodBuild(DatasetBenchmark):
    """Construction cost of a ``Neighborhood``, split into its three stages.

    ``Neighborhood`` claims the neighbor query costs more than any reduction run
    on it. ``r`` is a great-circle radius in degrees, against
    a mean element spacing of roughly 4.3 degrees at 480km and 1.1 at 120km, so
    the smallest radius here is near self-only on the coarser mesh.
    """

    param_names = DatasetBenchmark.param_names + ['r']
    params = DatasetBenchmark.params + [[1.0, 5.0, 15.0]]

    def setup(self, resolution, r):
        super().setup(resolution)
        self.uxgrid = self.uxds.uxgrid
        # Build the coordinates, and the njit paths behind them, here -- so the
        # timings below are the query rather than lat/lon construction.
        self.coords = _get_element_coords(self.uxgrid, "face centers", "spherical")
        self.tree = self.uxgrid.get_ball_tree(coordinates="face centers",
                                              coordinate_system="spherical",
                                              distance_metric="haversine")

    def time_query_radius(self, resolution, r):
        self.tree.query_radius(self.coords, r=r)

    def time_build(self, resolution, r):
        """Query plus CSR flatten; ``setup`` has already cached the tree."""
        Neighborhood(self.uxgrid, r=r, on="face centers")

    def track_nbytes_neighbors(self, resolution, r):
        """Size of the CSR structure a ``Neighborhood`` holds onto."""
        nb = Neighborhood(self.uxgrid, r=r, on="face centers")
        return nb._flat.nbytes + nb._starts.nbytes + nb._counts.nbytes

    track_nbytes_neighbors.unit = "bytes"

    def track_peakmem_build(self, resolution, r):
        """Transient high-water allocation of building a ``Neighborhood``."""
        return peak_allocated(
            lambda: Neighborhood(self.uxgrid, r=r, on="face centers"))

    track_peakmem_build.unit = "bytes"

    def track_mean_neighbors(self, resolution, r):
        """Mean neighborhood size -- the ``k`` behind every timing here."""
        nb = Neighborhood(self.uxgrid, r=r, on="face centers")
        return round(float(nb.n_neighbors.mean()), 2)

    track_mean_neighbors.unit = "elements"


class NeighborhoodReduce(DatasetBenchmark):
    """Reduction cost, and what reusing one neighbor query saves.

    One reduction, measured three ways: on a neighborhood built in ``setup``,
    which is the compiled kernel alone; through ``UxDataArray.neighborhood``,
    which pays for a query per call; and through ``UxDataset.neighborhood``,
    which shares one query per grid location across every variable. The first
    two bracket how much of a call is the query, and the third says whether
    ``DatasetNeighborhood`` actually shares one.

    ``mean`` is linear in the neighborhood, while ``median`` partitions it.
    """

    param_names = DatasetBenchmark.param_names + ['reduction']
    params = DatasetBenchmark.params + [['mean', 'median']]

    radius = 15.0

    @staticmethod
    def _run(neighborhood, reduction):
        """Calls ``reduction`` on an already-bound neighborhood."""
        if reduction == 'percentile':
            return neighborhood.percentile(90)
        if reduction == 'std':
            return neighborhood.std(ddof=1)
        return getattr(neighborhood, reduction)()

    def setup(self, resolution, reduction):
        super().setup(resolution)
        uxgrid = self.uxds.uxgrid

        # There is one compiled kernel per reduction, so warm the one under
        # test on the coarsest grid. ``cache=True`` is an on-disk cache and asv
        # builds a fresh environment per commit, so the first call still
        # compiles.
        grid, data = file_path_dict[self.params[0][0]]
        warmup = ux.open_dataset(grid, data)[data_var].neighborhood(r=1.0)
        _ = self._run(warmup, reduction)

        # A second face-centered variable, so the dataset case has something to
        # share a query with, and one variable at each of the other two
        # locations, so it has to build more than one.
        self.uxds['depth_squared'] = self.uxds[data_var] ** 2
        self.uxds['node_var'] = ux.UxDataArray(
            np.ones(uxgrid.n_node), dims=('n_node',), uxgrid=uxgrid)
        self.uxds['edge_var'] = ux.UxDataArray(
            np.ones(uxgrid.n_edge), dims=('n_edge',), uxgrid=uxgrid)
        # Edge coordinates pull in edge_node_connectivity; build all three
        # locations now so the first timed call is not the one that pays.
        _, _, _ = uxgrid.node_lon, uxgrid.edge_lon, uxgrid.face_lon

        self.nb = self.uxds[data_var].neighborhood(r=self.radius)

    def time_reduce(self, resolution, reduction):
        """The kernel alone: the query was paid for in ``setup``."""
        self._run(self.nb, reduction)

    def time_neighborhood_reduce(self, resolution, reduction):
        """A query per call, which is what reuse is meant to avoid."""
        self._run(self.uxds[data_var].neighborhood(r=self.radius), reduction)

    def time_dataset_reduce(self, resolution, reduction):
        """Four variables across three grid locations, sharing three queries."""
        self._run(self.uxds.neighborhood(r=self.radius), reduction)

    def track_peakmem_reduce(self, resolution, reduction):
        """Transient allocation of the kernel, with the query already paid for.

        Held to one numba thread: the kernels are ``target="parallel"``, and
        tracing serializes them on tracemalloc's allocator lock.
        """
        with numba_threads(1):
            return peak_allocated(lambda: self._run(self.nb, reduction))

    track_peakmem_reduce.unit = "bytes"


class NeighborhoodDask(DatasetBenchmark):
    """A reduction over lazy input, chunked three ways."""

    param_names = DatasetBenchmark.param_names + ['chunking']
    params = DatasetBenchmark.params + [['numpy', 'time_chunks', 'grid_chunks']]

    n_time = 12
    radius = 5.0

    def setup(self, resolution, chunking):
        super().setup(resolution)
        grid, data = file_path_dict[self.params[0][0]]
        _ = ux.open_dataset(grid, data)[data_var].neighborhood(r=1.0).mean()

        base = self.uxds[data_var].values
        stacked = np.broadcast_to(base, (self.n_time,) + base.shape).copy()
        uxda = ux.UxDataArray(stacked, dims=('time', 'n_face'),
                              uxgrid=self.uxds.uxgrid)
        if chunking == 'time_chunks':
            uxda = uxda.chunk({'time': 1})
        elif chunking == 'grid_chunks':
            uxda = uxda.chunk({'time': 1,
                               'n_face': uxda.sizes['n_face'] // 4})

        # Built here, so these measure the reduction and the graph it runs
        # through rather than the query.
        self.nb = uxda.neighborhood(r=self.radius)

        # One reduction here too, to warm the dask graph path -- and, for
        # 'grid_chunks', to let the rechunk warning through exactly once...
        _ = self.nb.mean().compute()

        # ...then silence the repeats.
        warnings.filterwarnings('ignore', category=UserWarning,
                                message='Rechunking')

    def time_mean(self, resolution, chunking):
        _ = self.nb.mean().compute()

    def track_peakmem_mean(self, resolution, chunking):
        """High-water allocation of the reduction, tree query excluded."""
        with numba_threads(1):
            return peak_allocated(lambda: self.nb.mean().compute())

    track_peakmem_mean.unit = "bytes"
