import os
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]

grid_quad_hex = current_path / "test" / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"
grid_geoflow = current_path / "test" / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
grid_scrip = current_path / "test" / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
grid_mpas = current_path / "test" / "meshfiles" / "mpas" / "QU" / "oQU480.231010.nc"


class FaceFaceConnectivity:
    params = [grid_quad_hex, grid_geoflow, grid_scrip, grid_mpas]

    def setup(self, grid_path):
        self.uxgrid = ux.open_grid(grid_path)

    def teardown(self, n):
        del self.uxgrid

    def time_construct(self, grid_path):
        """Time to populate ``face_face_connectivity``"""
        self.face_face_connectivity
