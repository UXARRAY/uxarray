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




class TestGrad(TestCase):
    mpas_atmo_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'
    mpas_ocean_path = current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc'

    CSne30_grid_path = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    CSne30_data_path = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"

    quad_hex_grid_path = current_path / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"
    quad_hex_data_path = current_path / "meshfiles" / "ugrid" / "quad-hexagon" / "data.nc"

    geoflow_grid_path = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
    geoflow_data_path = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"

    def test_uniform_data(self):
        """Computes the gradient on meshes with uniform data, with the expected
        gradient being zero on all edges."""
        for grid_path in [
                self.mpas_atmo_path, self.mpas_ocean_path, self.CSne30_grid_path, self.quad_hex_grid_path,
        ]:
            uxgrid = ux.open_grid(grid_path)

            uxda_zeros = ux.UxDataArray(data=np.zeros(uxgrid.n_face),
                                        uxgrid=uxgrid,
                                        name="zeros",
                                        dims=['n_face'])

            zero_grad = uxda_zeros.gradient()

            nt.assert_array_equal(zero_grad.values, np.zeros(uxgrid.n_edge))

            uxda_ones = ux.UxDataArray(data=np.ones(uxgrid.n_face),
                                       uxgrid=uxgrid,
                                       name="zeros",
                                       dims=['n_face'])

            one_grad = uxda_ones.gradient()

            nt.assert_array_equal(one_grad.values, np.zeros(uxgrid.n_edge))


    def test_quad_hex(self):
        """Computes the gradient on a mesh of 4 hexagons."""

        uxds = ux.open_dataset(self.quad_hex_grid_path, self.quad_hex_data_path)

        grad = uxds['t2m'].gradient()

        for i, edge in enumerate(uxds.uxgrid.edge_face_connectivity.values):

            if INT_FILL_VALUE in edge:
                # an edge not saddled by two faces has a grad of 0
                assert grad.values[i] == 0
            else:
                # a non zero gradient for edges sadled by two faces
                assert np.nonzero(grad.values[i])

    def test_normalization(self):
        """Tests the normalization gradient values."""

        uxds = ux.open_dataset(self.quad_hex_grid_path, self.quad_hex_data_path)

        grad_l2_norm = uxds['t2m'].gradient(normalize=True)

        assert np.isclose(np.sum(grad_l2_norm.values**2), 1)


    def test_grad_multi_dim(self):

        uxgrid = ux.open_grid(self.quad_hex_grid_path)

        sample_data = np.random.randint(-10, 10, (5, 5, 4))

        uxda = ux.UxDataArray(uxgrid=uxgrid,
                              data=sample_data,
                              dims=["time", "lev", "n_face"])

        grad = uxda.gradient(normalize=True)

        assert grad.shape[:-1] == uxda.shape[:-1]




class TestDifference(TestCase):

    quad_hex_grid_path = current_path / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"
    quad_hex_data_path = current_path / "meshfiles" / "ugrid" / "quad-hexagon" / "data.nc"

    geoflow_grid_path = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
    geoflow_data_path = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"

    mpas_atmo_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'
    mpas_ocean_path = current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc'

    CSne30_grid_path = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    CSne30_data_path = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"


    def test_face_centered_difference(self):

        uxds = ux.open_dataset(self.CSne30_grid_path, self.CSne30_data_path)

        uxda_diff = uxds['psi'].difference(destination='edge')

        assert uxda_diff._edge_centered()

        uxds = ux.open_dataset(self.mpas_atmo_path, self.mpas_atmo_path)

        uxda_diff = uxds['areaCell'].difference(destination='edge')

        assert uxda_diff._edge_centered()



    def test_node_centered_difference(self):
        uxds = ux.open_dataset(self.geoflow_grid_path, self.geoflow_data_path)

        uxda_diff = uxds['v1'][0][0].difference(destination='edge')

        assert uxda_diff._edge_centered()


    def test_hexagon(self):
        uxds = ux.open_dataset(self.quad_hex_grid_path, self.quad_hex_data_path)

        uxda_diff = uxds['t2m'].difference(destination='edge')


        # expected number of edges is n_face + 1, since we have 4 polygons
        assert len(np.nonzero(uxda_diff.values)[0]) == uxds.uxgrid.n_face + 1
