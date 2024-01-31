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


esmf_ne30_grid_path = current_path / "meshfiles" / "esmf" / "ne30" / "ne30pg3_ESMFmesh.nc"




def test_read_esmf():
    """Tests the reading of an ESMF grid file and its encoding into the UGRID
    conventions."""

    uxgrid = ux.open_grid(esmf_ne30_grid_path)

    dims = ['n_node', 'n_face']

    coords = ['node_lon', 'node_lat', 'face_lon', 'face_lat']

    conns = ['face_node_connectivity', 'n_nodes_per_face']

    for dim in dims:
        assert dim in uxgrid._ds

    for coord in coords:
        assert coord in uxgrid._ds

    for conn in conns:
        assert conn in uxgrid._ds
