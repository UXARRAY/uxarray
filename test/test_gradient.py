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

from uxarray.core.gradient import _calculate_edge_grad
from uxarray.grid.neighbors import _populate_edge_node_distances


class TestGrad(TestCase):
    mpas_grid_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'

    def test_grad_mpas(self):
        uxds = ux.open_dataset(self.mpas_grid_path, self.mpas_grid_path)
        uxgrid = uxds.uxgrid
        grad = _calculate_edge_grad(uxds['lonCell'].values,
                                    uxgrid.Mesh2_edge_faces.values,
                                    uxgrid.Mesh2_edge_node_distances.values,
                                    uxgrid.nMesh2_edge)

        e_d = _populate_edge_node_distances(uxgrid.Mesh2_node_x.values,
                                            uxgrid.Mesh2_node_y.values,
                                            uxgrid.Mesh2_edge_nodes.values)

        pass
