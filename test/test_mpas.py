from uxarray._mpas import _read_mpas
import uxarray as ux
import xarray as xr
from unittest import TestCase
import numpy as np

import os
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestMPAS(TestCase):

    mpas_small_grid_path = current_path / 'meshfiles' / "mpas" / "x1" / 'x1.40962.static.nc'
    mpas_large_grid_path = current_path / 'meshfiles' / "mpas" / "x1" / 'x1.655362.grid_subset.nc'

    def test_read_mpas(self):
        """Tests execution of _read_mpas()"""
        mpas_small_ds = xr.open_dataset(self.mpas_small_grid_path)
        mpas_small_ugrid = _read_mpas(mpas_small_ds)

        mpas_large_ds = xr.open_dataset(self.mpas_large_grid_path)
        mpas_large_ugrid = _read_mpas(mpas_large_ds)
        return

    def test_mpas_grid(self):
        """Tests creation of Grid object from converted mpas dataset."""
        mpas_small_ds = xr.open_dataset(self.mpas_small_grid_path)
        mpas_small_grid = ux.Grid(mpas_small_ds)

        mpas_large_ds = xr.open_dataset(self.mpas_large_grid_path)
        mpas_large_grid = ux.Grid(mpas_large_ds)

    def test_mpas_ugrid_conversion(self):
        """Verifies that the conversion to UGRID was successful."""
        mpas_small_ds = xr.open_dataset(self.mpas_small_grid_path)
        mpas_large_ds = xr.open_dataset(self.mpas_large_grid_path)

        mpas_small_ugrid = _read_mpas(mpas_small_ds)
        mpas_large_ugrid = _read_mpas(mpas_large_ds)

        for ds in [mpas_small_ugrid, mpas_large_ugrid]:
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
