import os
from pathlib import Path

import uxarray as ux
from .helpers._memsize import grid_nbytes

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]

grid_quad_hex = current_path / "test" / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"
grid_geoflow = current_path / "test" / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
grid_scrip = current_path / "test" / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
grid_mpas= current_path / "test" / "meshfiles" / "mpas" / "QU" / "oQU480.231010.nc"

class FaceBounds:

    params = [grid_quad_hex, grid_geoflow, grid_scrip, grid_mpas]


    def setup(self, grid_path):
        self.uxgrid = ux.open_grid(grid_path)

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


class FaceBoundsPeakMem:
    """Peak memory of a cold start: import uxarray, open a grid, get its bounds.

    Deliberately defines no ``setup``. asv counts setup memory towards
    ``peakmem_*``, and asv gives each parameter its own process, so with the JIT
    warmed in ``setup_cache`` the measured process does exactly the work named in
    the benchmark and nothing else.
    """

    params = FaceBounds.params
    param_names = ["grid_path"]

    def setup_cache(self):
        """Compile the njit kernels before anything is measured.

        asv runs this in its own process, so LLVM's few hundred MB stays out of
        the measured ones -- they load from numba's on-disk cache instead.
        Without it the first measured process compiles and the rest do not, which
        is what made the old ``peakmem_*`` swing ~2x with run order. All params
        are warmed; the grids do not reach the same njit signatures.
        """
        for grid_path in self.params:
            ux.open_grid(grid_path).bounds

    def peakmem_open_and_bounds(self, grid_path):
        ux.open_grid(grid_path).bounds
