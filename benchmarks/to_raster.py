import cartopy.crs as ccrs
import matplotlib
matplotlib.use("Agg")  # default backend causes ASV to crash on MacOS.
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import uxarray as ux



from .helpers._memsize import dataset_nbytes, grid_nbytes
from .helpers._peakmem import numba_threads, peak_allocated


def make_sphere_grid(n_faces, area_ratio, *, seed=0):
    """
    Returns a variable-resolution triangular grid on the unit sphere.

    n_faces    : target number of triangular faces (result is approximate)
    area_ratio : (face area at poles) / (face area at equator)
    seed       : RNG seed for the per-ring longitude offsets

    Returns
    -------
    node_lon, node_lat : (n_nodes,) degrees (lon in [-180, 180))
    face_node_connectivity : (n_faces, 3) int array, counter-clockwise
                             when viewed from outside the sphere.
    """
    # note: Claude made this function, aside from some small edits by hand.
    # The goal was to generate ANY "reasonable-looking" grid with varying face areas,
    # because to_raster() performance was bad for grids with significant area variations.

    # You can check it produces reasonable-looking results by doing something like:
    # areas = make_sphere_grid(1e4, 100).compute_face_areas(as_uxarray=True)
    # areas.plot()    # to view the actual areas
    # (areas*0 + np.arange(len(areas.n_face))).plot()   # to view face indices.

    edge_ratio = np.sqrt(area_ratio)          # edge length ~ sqrt(area)

    def h_of(lat, h_pole):                    # target edge length vs latitude
        return (h_pole / edge_ratio) * edge_ratio ** (np.abs(lat) / (np.pi / 2))

    def layout(h_pole):
        """Ring latitudes (all, south to north) and nodes per ring."""
        lats = [0.0]
        while True:
            nxt = lats[-1] + 0.866 * h_of(lats[-1], h_pole)
            if nxt > np.pi / 2 - 0.6 * h_pole:   # leave room for the pole node
                break
            lats.append(nxt)
        lats = np.array(lats)
        lats = np.concatenate([-lats[:0:-1], lats])
        counts = np.maximum(
            3, np.round(2 * np.pi * np.cos(lats) / h_of(lats, h_pole))
        ).astype(int)
        return lats, counts

    # --- choose h_pole so that the face count is close to n_faces
    # analytic estimate: faces ~ C / h_pole^2 (equilateral triangles)
    phi = np.linspace(-np.pi / 2, np.pi / 2, 20001)
    C = np.sum(2 * np.pi * np.cos(phi) / ((np.sqrt(3) / 4) * h_of(phi, 1.0) ** 2)) \
        * (phi[1] - phi[0])
    h0 = np.sqrt(C / n_faces)

    def faces_for(h_pole):                    # triangles = 2*nodes - 4 on a sphere
        return 2 * (layout(h_pole)[1].sum() + 2) - 4

    lo, hi = np.log(h0 / 2), np.log(h0 * 2)   # refine with the true ring counts
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if faces_for(np.exp(mid)) > n_faces:
            lo = mid                          # too many faces -> larger h
        else:
            hi = mid
    h_pole = np.exp(0.5 * (lo + hi))

    # --- nodes on each ring
    rng = np.random.default_rng(seed)
    lats, counts = layout(h_pole)
    lon_list, lat_list = [], []
    for lat, n in zip(lats, counts):
        lon_list.append(rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.arange(n) / n)
        lat_list.append(np.full(n, lat))
    lon_list += [np.array([0.0]), np.array([0.0])]            # poles
    lat_list += [np.array([np.pi / 2]), np.array([-np.pi / 2])]

    lon = np.concatenate(lon_list)
    lat = np.concatenate(lat_list)

    # --- spherical Delaunay via 3D convex hull
    xyz = np.column_stack([np.cos(lat) * np.cos(lon),
                           np.cos(lat) * np.sin(lon),
                           np.sin(lat)])
    faces = ConvexHull(xyz).simplices.copy()

    # orient counter-clockwise as seen from outside
    a, b, c = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), a + b + c) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]

    node_lon = (np.degrees(lon) + 180.0) % 360.0 - 180.0
    node_lat = np.degrees(lat)
    connectivity = faces.astype(np.int64)
    return ux.Grid.from_topology(node_lon, node_lat, connectivity)



class ToRaster:
    """Benchmark the time it takes to call to_raster(),
    on a variable-resolution grid.
    """
    param_names = ["n_faces_and_area_ratio",]
    params = [[
        (1e3, 10.0),
        (1e3, 100.0),
        # (1e3, 1000.0) isn't properly resolved;
        # it would just make a bunch of faces stretching from pole to equator.
        (1e4, 10.0),
        #(1e4, 100.0),
        #(1e4, 1000.0),
        #(1e5, 10.0),
        #(1e5, 100.0),
        #(1e5, 1000.0),
    ]]

    # if any combination takes longer than 3 mins, give up.
    timeout = 180

    def _warmup(self):
        # warm up numba, using a small example
        grid = make_sphere_grid(1e3, 10.0)
        data = grid.compute_face_areas(as_uxarray=True)
        # data could be any values; let's use areas for fun :)
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
        ax.set_global()
        data.to_raster(ax=ax)
        plt.close()

    def setup(self, n_faces_and_area_ratio):
        n_faces, area_ratio = n_faces_and_area_ratio
        grid = make_sphere_grid(n_faces, area_ratio)
        self.data = grid.compute_face_areas(as_uxarray=True)
        self._warmup()
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
        self.ax.set_global()

    def teardown(self, n_faces_and_area_ratio):
        del self.data
        del self.fig
        del self.ax
        plt.close()

    def time_to_raster(self, n_faces_and_area_ratio):
        """Time to call to_raster() on a variable-resolution grid."""
        self.data.to_raster(ax=self.ax)

    def track_peakmem(self, n_faces_and_area_ratio):
        """High-water allocation of to_raster()"""
        with numba_threads(1):
            return peak_allocated(lambda: self.data.to_raster(ax=self.ax))

    track_peakmem.unit = "bytes"
