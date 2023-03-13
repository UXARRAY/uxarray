from uxarray._mpas import _read_mpas, _add_fill_values
import uxarray as ux
import xarray as xr
from unittest import TestCase
import numpy as np
import os
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestMPAS(TestCase):
    """Test suite for Read MPAS functionality."""

    mpas_grid_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'

    # fill value
    int_dtype = np.uint32
    fv = np.iinfo(int_dtype).max

    def test_read_mpas(self):
        """Tests execution of _read_mpas()"""
        mpas_xr_ds = xr.open_dataset(self.mpas_grid_path)
        mpas_ugrid = _read_mpas(mpas_xr_ds)

    def test_mpas_grid(self):
        """Tests creation of Grid object from converted MPAS dataset."""
        mpas_xr_ds = xr.open_dataset(self.mpas_grid_path)
        mpas_uxgrid = ux.Grid(mpas_xr_ds)

    def test_mpas_ugrid_conversion(self):
        """Verifies that the conversion to UGRID was successful."""
        mpas_xr_ds = xr.open_dataset(self.mpas_grid_path)
        ds = _read_mpas(mpas_xr_ds)

        # check for correct dimensions
        expected_ugrid_dims = [
            'nMesh2_node', "nMesh2_face", "nMaxMesh2_face_nodes"
        ]
        for dim in expected_ugrid_dims:
            assert dim in ds.sizes

        # check for correct length of coordinates
        assert len(ds['Mesh2_node_x']) == len(ds['Mesh2_node_y'])
        assert len(ds['Mesh2_face_x']) == len(ds['Mesh2_face_y'])

        # check for correct shape of face nodes
        nMesh2_face = ds.sizes['nMesh2_face']
        nMaxMesh2_face_nodes = ds.sizes['nMaxMesh2_face_nodes']
        assert ds['Mesh2_face_nodes'].shape == (nMesh2_face,
                                                nMaxMesh2_face_nodes)

        # check for zero-indexing
        assert ds['Mesh2_face_nodes'].min() == 0

    def test_add_fill_values(self):
        """Test _add_fill_values() implementation, output should be both zero-
        indexed and padded values should be replaced with fill values."""

        # two cells with 2 and 3 padded faces respectively
        test_verticesOnCell = np.array([[1, 2, 1, 1], [3, 4, 5, 3]])
        # cell has 2 and 3 nodes respectively
        test_nEdgesOnCell = np.array([2, 3])

        # expected output of _add_fill_values()
        gold_output = np.array([[0, 1, self.fv, self.fv],
                                [2, 3, 4, self.fv]]).astype(self.int_dtype)

        # test data output
        verticesOnCell_fill = _add_fill_values(test_verticesOnCell,
                                               test_nEdgesOnCell)

        assert np.array_equal(verticesOnCell_fill, gold_output)
