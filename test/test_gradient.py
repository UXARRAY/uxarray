from uxarray.io._mpas import _replace_padding, _replace_zeros, _to_zero_index
from uxarray.io._mpas import _read_mpas
import uxarray as ux
import xarray as xr
from unittest import TestCase
import numpy as np
import numpy.testing as nt
import os
from pathlib import Path

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

from uxarray.core.gradient import _calculate_grad_on_edge
from uxarray.grid.neighbors import _construct_edge_node_distances, _construct_edge_face_distances

import xarray as xr


class TestGrad(TestCase):
    mpas_atmo_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'
    mpas_ocean_path = current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc'

    CSne30_grid_path = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    CSne30_data_path = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"

    def test_grad_mpas(self):
        """TODO:"""
        uxds = ux.open_dataset(self.mpas_ocean_path, self.mpas_ocean_path)
        uxgrid = uxds.uxgrid

        xrds = xr.open_dataset(self.mpas_ocean_path)

        edge_face_d = _construct_edge_face_distances(
            uxgrid.Mesh2_face_x.values, uxgrid.Mesh2_face_y.values,
            uxgrid.Mesh2_edge_faces.values)

        grad = _calculate_grad_on_edge(uxds['areaCell'].values,
                                       uxgrid.Mesh2_edge_faces.values,
                                       uxgrid.Mesh2_edge_face_distances.values,
                                       uxgrid.nMesh2_edge)

        mpas_d = uxgrid.Mesh2_edge_face_distances

        pass

    def test_grad_uniform_data(self):
        """Computes the gradient on meshes with uniform data, with the expected
        gradient being zero."""
        for grid_path in [
                self.mpas_atmo_path, self.mpas_ocean_path, self.CSne30_grid_path
        ]:
            uxgrid = ux.open_grid(grid_path)

            uxda_zeros = ux.UxDataArray(data=np.zeros(uxgrid.nMesh2_face),
                                        uxgrid=uxgrid,
                                        name="zeros")

            zero_grad = uxda_zeros.gradient(normalize=False)

            nt.assert_array_equal(zero_grad.values,
                                  np.zeros(uxgrid.nMesh2_edge))

            uxda_ones = ux.UxDataArray(data=np.ones(uxgrid.nMesh2_face),
                                       uxgrid=uxgrid,
                                       name="zeros")

            one_grad = uxda_ones.gradient(normalize=False)

            nt.assert_array_equal(one_grad.values, np.zeros(uxgrid.nMesh2_edge))
