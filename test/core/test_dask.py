
import dask.array as da
import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux

def test_grid_chunking(gridpath):
    """Tests the chunking of an entire grid."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))

    for var in uxgrid._ds:
        # variables should all be np.ndarray
        assert isinstance(uxgrid._ds[var].data, np.ndarray)

    # chunk every data variable
    uxgrid.chunk(n_node=1, n_face=2, n_edge=4, inplace=True)

    for var in uxgrid._ds:
        # variables should all be da.Array
        assert isinstance(uxgrid._ds[var].data, da.Array)

def test_grid_chunk_default_warns_and_chunks_in_place(gridpath):
    """Without inplace, Grid.chunk() keeps chunking in place but warns about the coming default."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))

    with pytest.warns(FutureWarning, match="inplace=False"):
        result = uxgrid.chunk(n_face=500)

    assert result is None
    assert uxgrid.chunks["n_face"] == (500, 500, 500, 291)

def test_grid_chunk_not_in_place(gridpath):
    """Grid.chunk(inplace=False) returns a new chunked grid and leaves the original untouched."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))

    chunked = uxgrid.chunk(n_node=1000, n_edge=2000, n_face=500, inplace=False)

    assert isinstance(chunked, ux.Grid) and chunked is not uxgrid
    assert dict(uxgrid.chunks) == {}
    for var in uxgrid._ds:
        assert isinstance(uxgrid._ds[var].data, np.ndarray)

    assert chunked.chunks["n_node"] == (1000, 1000, 1000, 947)
    assert chunked.chunks["n_edge"] == (2000, 2000, 1754)
    assert chunked.chunks["n_face"] == (500, 500, 500, 291)
    nt.assert_array_equal(chunked.face_node_connectivity.values, uxgrid.face_node_connectivity.values)

    # a variable built lazily on the copy must not appear on the original
    _ = chunked.bounds
    assert "bounds" in chunked._ds
    assert "bounds" not in uxgrid._ds

def test_individual_var_chunking(gridpath):
    """Tests the chunking of a single grid variable."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))

    # face_node_conn should originally be a numpy array
    assert isinstance(uxgrid.face_node_connectivity.data, np.ndarray)

    # chunk face_node_connectivity
    uxgrid.face_node_connectivity = uxgrid.face_node_connectivity.chunk(chunks={"n_face": 16})

    # face_node_conn should now be a dask array
    assert isinstance(uxgrid.face_node_connectivity.data, da.Array)

def test_uxds_chunking(gridpath, datasetpath):
    """Tests the chunking of a dataset."""
    uxds = ux.open_dataset(
        gridpath("ugrid", "outCSne30", "outCSne30.ug"),
        datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc"),
        chunks=-1
    )

    for var in uxds.variables:
        assert isinstance(uxds[var].data, da.Array)

def test_open_grid(gridpath):
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"), chunks=-1)
    assert isinstance(uxgrid.node_lon.data, da.Array)
    assert isinstance(uxgrid.node_lat.data, da.Array)
    assert isinstance(uxgrid.face_node_connectivity.data, da.Array)

def test_open_dataset(gridpath, datasetpath):
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "multi_dim_data.nc"), chunks=-1, chunk_grid=False)

    assert isinstance(uxds['multi_dim_data'].data, da.Array)
    assert isinstance(uxds.uxgrid.face_node_connectivity.data, np.ndarray)

    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "multi_dim_data.nc"), chunks=-1, chunk_grid=True)

    assert isinstance(uxds['multi_dim_data'].data, da.Array)
    assert isinstance(uxds.uxgrid.face_node_connectivity.data, da.Array)

    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "multi_dim_data.nc"), chunks={"time": 2, "lev": 2}, chunk_grid=True)

    # Chunk sizes should be 2, 2, 4
    chunks = uxds['multi_dim_data'].chunks
    assert chunks[0][0] == 2
    assert chunks[1][0] == 2
    assert chunks[2][0] == 4

    # Grid should not be chunked here
    assert isinstance(uxds.uxgrid.face_node_connectivity.data, da.Array)
