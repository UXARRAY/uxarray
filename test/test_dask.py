import os
from pathlib import Path

import dask.array as da
import numpy as np

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

quad_hex_grid_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'grid.nc'
quad_hex_data_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'multi_dim_data.nc'

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
    """Tests the chunking of a dataset."""
    uxds = ux.open_dataset(csne30_grid, csne30_data, chunks=-1)

    for var in uxds.variables:
        assert isinstance(uxds[var].data, da.Array)


def test_open_grid():
    uxgrid = ux.open_grid(mpas_grid, chunks=-1)
    assert isinstance(uxgrid.node_lon.data, da.Array)
    assert isinstance(uxgrid.node_lat.data, da.Array)
    assert isinstance(uxgrid.face_node_connectivity.data, da.Array)


def test_open_dataset():
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path, chunks=-1, chunk_grid=False)

    assert isinstance(uxds['multi_dim_data'].data, da.Array)
    assert isinstance(uxds.uxgrid.face_node_connectivity.data, np.ndarray)

    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path, chunks=-1, chunk_grid=True)

    assert isinstance(uxds['multi_dim_data'].data, da.Array)
    assert isinstance(uxds.uxgrid.face_node_connectivity.data, da.Array)

    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path, chunks={"time": 2, "lev": 2}, chunk_grid=True)

    # Chunk sizes should be 2, 2, 4
    chunks = uxds['multi_dim_data'].chunks
    assert chunks[0][0] == 2
    assert chunks[1][0] == 2
    assert chunks[2][0] == 4

    # Grid should not be chunked here
    assert isinstance(uxds.uxgrid.face_node_connectivity.data, da.Array)
