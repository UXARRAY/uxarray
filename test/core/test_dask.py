import os
from pathlib import Path

import dask.array as da
import numpy as np

import uxarray as ux

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *

csne30_grid = OUTCSNE30_GRID
csne30_data = OUTCSNE30_VAR2

def test_grid_chunking():
    """Tests the chunking of an entire grid."""
    uxgrid = ux.open_grid(MPAS_OCEAN_MESH)

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
    uxgrid = ux.open_grid(MPAS_OCEAN_MESH)

    # face_node_conn should originally be a numpy array
    assert isinstance(uxgrid.face_node_connectivity.data, np.ndarray)

    # chunk face_node_connectivity
    uxgrid.face_node_connectivity = uxgrid.face_node_connectivity.chunk(chunks={"n_face": 16})

    # face_node_conn should now be a dask array
    assert isinstance(uxgrid.face_node_connectivity.data, da.Array)

def test_uxds_chunking():
    """Tests the chunking of a dataset."""
    uxds = ux.open_dataset(csne30_grid, csne30_data, chunks=-1)

    for var in uxds.variables:
        assert isinstance(uxds[var].data, da.Array)

def test_open_grid():
    uxgrid = ux.open_grid(MPAS_OCEAN_MESH, chunks=-1)
    assert isinstance(uxgrid.node_lon.data, da.Array)
    assert isinstance(uxgrid.node_lat.data, da.Array)
    assert isinstance(uxgrid.face_node_connectivity.data, da.Array)

def test_open_dataset():
    uxds = ux.open_dataset(QUAD_HEXAGON_GRID, QUAD_HEXAGON_MULTI_DIM_DATA, chunks=-1, chunk_grid=False)

    assert isinstance(uxds['multi_dim_data'].data, da.Array)
    assert isinstance(uxds.uxgrid.face_node_connectivity.data, np.ndarray)

    uxds = ux.open_dataset(QUAD_HEXAGON_GRID, QUAD_HEXAGON_MULTI_DIM_DATA, chunks=-1, chunk_grid=True)

    assert isinstance(uxds['multi_dim_data'].data, da.Array)
    assert isinstance(uxds.uxgrid.face_node_connectivity.data, da.Array)

    uxds = ux.open_dataset(QUAD_HEXAGON_GRID, QUAD_HEXAGON_MULTI_DIM_DATA, chunks={"time": 2, "lev": 2}, chunk_grid=True)

    # Chunk sizes should be 2, 2, 4
    chunks = uxds['multi_dim_data'].chunks
    assert chunks[0][0] == 2
    assert chunks[1][0] == 2
    assert chunks[2][0] == 4

    # Grid should not be chunked here
    assert isinstance(uxds.uxgrid.face_node_connectivity.data, da.Array)
