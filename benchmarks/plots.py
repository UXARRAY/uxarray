"""benchmarks related to plotting"""

import cartopy.crs as ccrs
import holoviews as hv
import matplotlib
matplotlib.use("Agg")  # default backend causes ASV to crash on MacOS.
import matplotlib.pyplot as plt
import numpy as np
import uxarray as ux
import xarray as xr

from .helpers._peakmem import numba_threads, peak_allocated

# define only once, before benchmarks run.
RENDERER = hv.renderer("matplotlib")


class PlotHealpix:
    """Benchmark calls to plotting methods for data on a HEALPix grid.
    HEALPix size scales by a factor of 4 per zoom level, with ~50k faces at zoom 6.
    """

    param_names = ["zoom"]
    params = [[6, 8]]

    def make_array(self, zoom):
        """returns UxDataArray on a HEALPix grid with the given zoom level,
        filled with some example data: np.arange(n_face).
        """
        grid = ux.Grid.from_healpix(zoom)
        xarr = xr.DataArray(np.arange(grid.n_face), dims="n_face")
        return ux.UxDataArray(xarr, uxgrid=grid)

    def _warmup(self):
        # warm up numba and plotting imports, using a small example.
        arr = self.make_array(zoom=2)
        plot_obj = arr.plot()
        RENDERER.get_plot(plot_obj)

    def setup(self, zoom):
        self._warmup()
        self.data = self.make_array(zoom)

    def teardown(self, zoom):
        del self.data
        plt.close()

    def _doit(self, **kwargs):
        """helper to call arr.plot() and render the result, but first removing any cached gdf,
        to ensure the plot is from a "cold" start for the given zoom level.
        ("cold" i.e. no cached gdf, but not fully cold; numba routines have been compiled,
        and holoviews imports have occurred. E.g., this corresponds to calling plot not for the
        first time ever in a notebook, but for the first time on the given array.)
        """
        self.data.uxgrid._gdf_cached_parameters["gdf"] = None  # defeat the cache
        plot_obj = self.data.plot(**kwargs)
        RENDERER.get_plot(plot_obj)

    def time_default_plot(self, zoom):
        """time taken to call arr.plot() with default inputs"""
        self._doit()

    def track_peakmem_default_plot(self, zoom):
        """High-water allocation of calling arr.plot() with default inputs"""
        with numba_threads(1):
            return peak_allocated(lambda: self._doit())

    track_peakmem_default_plot.units = "bytes"
