from uxarray.io._mpas import _replace_padding, _replace_zeros, _to_zero_index
from uxarray.io._mpas import _read_mpas
import uxarray as ux
import xarray as xr
from unittest import TestCase
import numpy as np
import os
from pathlib import Path

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

from uxarray.core.gradient import _calculate_abs_edge_grad
from uxarray.grid.neighbors import _populate_edge_node_distances, _populate_edge_face_distances


class TestGrad(TestCase):
    mpas_atmo_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'
    mpas_ocean_path = current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc'

    def test_grad_mpas(self):
        uxds = ux.open_dataset(self.mpas_atmo_path, self.mpas_atmo_path)
        uxgrid = uxds.uxgrid

        edge_face_d = _populate_edge_face_distances(
            uxgrid.Mesh2_face_x.values, uxgrid.Mesh2_face_y.values,
            uxgrid.Mesh2_edge_faces.values)

        grad = _calculate_abs_edge_grad(uxds['lonCell'].values,
                                        uxgrid.Mesh2_edge_faces.values,
                                        uxgrid.Mesh2_edge_node_distances.values,
                                        uxgrid.nMesh2_edge)

        pass
