import uxarray as ux
import xarray as xr


def test_read_esmf(gridpath):
    """Tests the reading of an ESMF grid file and its encoding into the UGRID
    conventions."""
    uxgrid = ux.open_grid(gridpath("esmf", "ne30", "ne30pg3.grid.nc"))

    dims = ['n_node', 'n_face', 'n_max_face_nodes']
    coords = ['node_lon', 'node_lat', 'face_lon', 'face_lat']
    conns = ['face_node_connectivity', 'n_nodes_per_face']

    for dim in dims:
        assert dim in uxgrid._ds.dims

    for coord in coords:
        assert coord in uxgrid._ds

    for conn in conns:
        assert conn in uxgrid._ds

def test_read_esmf_dataset(gridpath, datasetpath):
    """Tests the constructing of a UxDataset from an ESMF Grid and Data
    File."""
    uxds = ux.open_dataset(gridpath("esmf", "ne30", "ne30pg3.grid.nc"),
                           datasetpath("esmf", "ne30", "ne30pg3.data.nc"))

    dims = ['n_node', 'n_face']

    for dim in dims:
        assert dim in uxds.dims

def test_encode_esmf_structure(gridpath):
    """Encoding to ESMF produces the variables the format requires.

    Round-trip fidelity is covered for every writable format by
    ``TestIOWriteRoundTrip`` in test_io_common.py; this only pins down the
    ESMF-specific layout.
    """
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    esmf_dataset = uxgrid.to_xarray("ESMF")

    assert isinstance(esmf_dataset, xr.Dataset)
    assert 'nodeCoords' in esmf_dataset
    assert 'elementConn' in esmf_dataset
    assert 'numElementConn' in esmf_dataset

    # elementConn is 1-based with -1 marking unused slots
    assert esmf_dataset['elementConn'].attrs['_FillValue'] == -1
    assert esmf_dataset['numElementConn'].values.sum() == (
        uxgrid.n_nodes_per_face.values.sum()
    )
