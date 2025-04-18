import uxarray as ux

import numpy.testing as nt
import os
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

CSne30_data_path = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"
quad_hex_grid_path = current_path / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"


def test_quad_hex_face_centers():
    """Tests a face center query into the RTree, which expects the same index
    to be returned."""
    uxgrid = ux.open_grid(quad_hex_grid_path)
    rt = uxgrid.get_rtree()

    for i in range(uxgrid.n_face):
        x = uxgrid.face_x[i].values
        y = uxgrid.face_y[i].values
        z = uxgrid.face_z[i].values
        res = rt.intersects((x, y, z, x, y, z))
        assert len(res) == 1
        assert res[0] == i
