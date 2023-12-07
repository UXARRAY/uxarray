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


class TestMPAS(TestCase):
    """Test suite for Read MPAS functionality."""

    # sample mpas dataset
    mpas_grid_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'
    mpas_xr_ds = xr.open_dataset(mpas_grid_path)

    mpas_ocean_mesh = current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc'

    # fill value (remove once there is a unified approach in uxarray)
    fv = INT_FILL_VALUE

    def test_read_mpas(self):
        """Tests execution of _read_mpas()"""
        mpas_primal_ugrid, _ = _read_mpas(self.mpas_xr_ds, use_dual=False)
        mpas_dual_ugrid, _ = _read_mpas(self.mpas_xr_ds, use_dual=True)

    def test_mpas_to_grid(self):
        """Tests creation of Grid object from converted MPAS dataset."""
        mpas_uxgrid_primal = ux.open_grid(self.mpas_grid_path, use_dual=False)
        mpas_uxgrid_dual = ux.open_grid(self.mpas_grid_path, use_dual=True)
        mpas_uxgrid_dual.__repr__()
        pass

    def test_primal_to_ugrid_conversion(self):
        """Verifies that the Primal-Mesh was converted properly."""

        for path in [self.mpas_grid_path, self.mpas_ocean_mesh]:
            # dual-mesh encoded in the UGRID conventions
            uxgrid = ux.open_grid(path, use_dual=False)
            ds = uxgrid._ds

            # check for correct dimensions
            expected_ugrid_dims = ['n_node', "n_face", "n_max_face_nodes"]
            for dim in expected_ugrid_dims:
                assert dim in ds.sizes

            # check for correct length of coordinates
            assert len(ds['node_lon']) == len(ds['node_lat'])
            assert len(ds['face_lon']) == len(ds['face_lat'])

            # check for correct shape of face nodes
            n_face = ds.sizes['n_face']
            n_max_face_nodes = ds.sizes['n_max_face_nodes']
            assert ds['face_node_connectivity'].shape == (n_face,
                                                          n_max_face_nodes)

            pass

    def test_dual_to_ugrid_conversion(self):
        """Verifies that the Dual-Mesh was converted properly."""

        for path in [self.mpas_grid_path, self.mpas_ocean_mesh]:

            # dual-mesh encoded in the UGRID conventions
            uxgrid = ux.open_grid(path, use_dual=True)
            ds = uxgrid._ds

            # check for correct dimensions
            expected_ugrid_dims = ['n_node', "n_face", "n_max_face_nodes"]
            for dim in expected_ugrid_dims:
                assert dim in ds.sizes

            # check for correct length of coordinates
            assert len(ds['node_lon']) == len(ds['node_lat'])
            assert len(ds['face_lon']) == len(ds['face_lat'])

            # check for correct shape of face nodes
            nMesh2_face = ds.sizes['n_face']
            assert ds['face_node_connectivity'].shape == (nMesh2_face, 3)

    def test_add_fill_values(self):
        """Test _add_fill_values() implementation, output should be both be
        zero-indexed and padded values should be replaced with fill values."""

        # two cells with 2, 3 and 2 padded faces respectively
        verticesOnCell = np.array([[1, 2, 1, 1], [3, 4, 5, 3], [6, 7, 0, 0]],
                                  dtype=INT_DTYPE)

        # cell has 2, 3 and 2 nodes respectively
        nEdgesOnCell = np.array([2, 3, 2])

        # expected output of _add_fill_values()
        gold_output = np.array([[0, 1, self.fv, self.fv], [2, 3, 4, self.fv],
                                [5, 6, self.fv, self.fv]],
                               dtype=INT_DTYPE)

        # test data output
        verticesOnCell = _replace_padding(verticesOnCell, nEdgesOnCell)
        verticesOnCell = _replace_zeros(verticesOnCell)
        verticesOnCell = _to_zero_index(verticesOnCell)

        assert np.array_equal(verticesOnCell, gold_output)

    def test_set_attrs(self):
        """Tests the execution of ``_set_global_attrs``, checking for
        attributes being correctly stored in ``Grid._ds``"""

        # full set of expected mpas attributes
        expected_attrs = [
            'sphere_radius', 'mesh_spec', 'on_a_sphere', 'mesh_id',
            'is_periodic', 'x_period', 'y_period'
        ]

        # included attrs: 'sphere_radius', 'mesh_spec' 'on_a_sphere'
        ds, _ = _read_mpas(self.mpas_xr_ds)

        # set dummy attrs to test execution
        ds.attrs['mesh_id'] = "12345678"
        ds.attrs['is_periodic'] = "YES"
        ds.attrs['x_period'] = 1.0
        ds.attrs['y_period'] = 1.0

        # create a grid
        uxgrid = ux.Grid(ds)

        # check if all expected attributes are set
        for mpas_attr in expected_attrs:
            assert mpas_attr in uxgrid._ds.attrs

    def test_face_mask(self):
        primal_uxgrid = ux.open_grid(self.mpas_ocean_mesh, use_dual=False)
        dual_uxgrid = ux.open_grid(self.mpas_ocean_mesh, use_dual=True)

        pass
