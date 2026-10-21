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

    def time_default_plot(self, zoom):
        """time taken to call arr.plot() with default inputs"""
        plot_obj = self.data.plot()
        RENDERER.get_plot(plot_obj)

    def track_peakmem_default_plot(self, zoom):
        """High-water allocation of calling arr.plot() with default inputs"""
        with numba_threads(1):
            return peak_allocated(lambda: RENDERER.get_plot(self.data.plot()))

    track_peakmem_default_plot.units = "bytes"
