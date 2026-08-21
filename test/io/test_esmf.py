import numpy as np
import pytest
import uxarray as ux
import xarray as xr
from uxarray.constants import INT_FILL_VALUE


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


@pytest.mark.parametrize("mask_and_scale", [True, False])
def test_read_esmf_padding_independent_of_cf_decoding(mask_and_scale, tmp_path):
    """Padding is recognized whether or not xarray decoded the fill value.

    CF decoding replaces the ``-1`` padding with NaN and promotes elementConn to
    float; an undecoded read hands back the raw int32. Neither survives the cast
    to INT_DTYPE as INT_FILL_VALUE, so both must be identified before it.
    """
    uxgrid = ux.Grid.from_topology(
        node_lon=np.array([0.0, 10.0, 10.0, 0.0, 20.0]),
        node_lat=np.array([0.0, 0.0, 10.0, 10.0, 0.0]),
        face_node_connectivity=np.array([
            [0, 1, 2, 3],
            [1, 4, 2, INT_FILL_VALUE],
            [0, 3, 4, INT_FILL_VALUE],
        ]),
        fill_value=INT_FILL_VALUE,
    )

    path = tmp_path / "esmf_ragged.nc"
    uxgrid.to_xarray("ESMF").to_netcdf(path)

    with xr.open_dataset(path, mask_and_scale=mask_and_scale) as ds:
        reloaded = ux.open_grid(ds)

    np.testing.assert_array_equal(
        reloaded.face_node_connectivity.values,
        uxgrid.face_node_connectivity.values,
    )
    np.testing.assert_array_equal(reloaded.n_nodes_per_face.values, [4, 3, 3])
