import uxarray as ux
import numpy as np
import dask.array as da

import pytest
import os
from pathlib import Path



current_path = Path(os.path.dirname(os.path.realpath(__file__)))

mpas_grid = current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc'

csne30_grid = current_path / 'meshfiles' / "ugrid" / "outCSne30" / 'outCSne30.ug'
csne30_data = current_path / 'meshfiles' / "ugrid" / "outCSne30" / 'outCSne30_var2.nc'


def test_grid_chunking():
    """Tests the chunking of an entire grid."""
    uxgrid = ux.open_grid(mpas_grid)

    for var in uxgrid._ds:
        # variables should all be np.ndarray
        assert isinstance(uxgrid._ds[var].data, np.ndarray)

    # chunk every data variable
    uxgrid.chunk(n_node=1, n_face=2, n_edge=4)

    for var in uxgrid._ds:
        # variables should all be da.Array
        assert isinstance(uxgrid._ds[var].data, da.Array)

def test_individual_var_chunking():
    """Tests the chunking of a single grid variable."""
    uxgrid = ux.open_grid(mpas_grid)

    # face_node_conn should originally be a numpy array
    assert isinstance(uxgrid.face_node_connectivity.data, np.ndarray)

    # chunk face_node_connectivity
    uxgrid.face_node_connectivity = uxgrid.face_node_connectivity.chunk(chunks={"n_face": 16})

    # face_node_conn should now be a dask array
    assert isinstance(uxgrid.face_node_connectivity.data, da.Array)


def test_uxds_chunking():
    uxds = ux.open_dataset(csne30_grid, csne30_data, chunks={"n_face": 4})

    pass
