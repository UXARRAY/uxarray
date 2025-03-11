import numpy as np
import os
import pytest
import uxarray as ux
import xarray as xr
from pathlib import Path
import numpy.testing as nt
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.io._mpas import _replace_padding, _replace_zeros, _to_zero_index, _read_mpas

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

# Sample MPAS dataset paths
mpas_grid_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'
mpas_xr_ds = xr.open_dataset(mpas_grid_path)
mpas_ocean_mesh = current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc'

# Fill value
fv = INT_FILL_VALUE

def test_read_mpas():
    """Tests execution of _read_mpas()"""
    mpas_primal_ugrid, _ = _read_mpas(mpas_xr_ds, use_dual=False)
    mpas_dual_ugrid, _ = _read_mpas(mpas_xr_ds, use_dual=True)

def test_mpas_to_grid():
    """Tests creation of Grid object from converted MPAS dataset."""
    mpas_uxgrid_primal = ux.open_grid(mpas_grid_path, use_dual=False)
    mpas_uxgrid_dual = ux.open_grid(mpas_grid_path, use_dual=True)
    mpas_uxgrid_dual.__repr__()

def test_primal_to_ugrid_conversion():
    """Verifies that the Primal-Mesh was converted properly."""
    for path in [mpas_grid_path, mpas_ocean_mesh]:
        uxgrid = ux.open_grid(path, use_dual=False)
        ds = uxgrid._ds

        # Check for correct dimensions
        expected_ugrid_dims = ['n_node', "n_face", "n_max_face_nodes"]
        for dim in expected_ugrid_dims:
            assert dim in ds.sizes

        # Check for correct length of coordinates
        assert len(ds['node_lon']) == len(ds['node_lat'])
        assert len(ds['face_lon']) == len(ds['face_lat'])

        # Check for correct shape of face nodes
        n_face = ds.sizes['n_face']
        n_max_face_nodes = ds.sizes['n_max_face_nodes']
        assert ds['face_node_connectivity'].shape == (n_face, n_max_face_nodes)

def test_dual_to_ugrid_conversion():
    """Verifies that the Dual-Mesh was converted properly."""
    for path in [mpas_grid_path, mpas_ocean_mesh]:
        uxgrid = ux.open_grid(path, use_dual=True)
        ds = uxgrid._ds

        # Check for correct dimensions
        expected_ugrid_dims = ['n_node', "n_face", "n_max_face_nodes"]
        for dim in expected_ugrid_dims:
            assert dim in ds.sizes

        # Check for correct length of coordinates
        assert len(ds['node_lon']) == len(ds['node_lat'])
        assert len(ds['face_lon']) == len(ds['face_lat'])

        # Check for correct shape of face nodes
        nMesh2_face = ds.sizes['n_face']
        assert ds['face_node_connectivity'].shape == (nMesh2_face, 3)

def test_add_fill_values():
    """Test _add_fill_values() implementation."""
    verticesOnCell = np.array([[1, 2, 1, 1], [3, 4, 5, 3], [6, 7, 0, 0]], dtype=INT_DTYPE)
    nEdgesOnCell = np.array([2, 3, 2])
    gold_output = np.array([[0, 1, fv, fv], [2, 3, 4, fv], [5, 6, fv, fv]], dtype=INT_DTYPE)

    verticesOnCell = xr.DataArray(data=verticesOnCell, dims=['n_face', 'n_max_face_nodes'])
    nEdgesOnCell = xr.DataArray(data=nEdgesOnCell, dims=['n_face'])

    verticesOnCell = _replace_padding(verticesOnCell, nEdgesOnCell)
    verticesOnCell = _replace_zeros(verticesOnCell)
    verticesOnCell = _to_zero_index(verticesOnCell)

    assert np.array_equal(verticesOnCell, gold_output)

def test_set_attrs():
    """Tests the execution of _set_global_attrs."""
    expected_attrs = [
        'sphere_radius', 'mesh_spec', 'on_a_sphere', 'mesh_id',
        'is_periodic', 'x_period', 'y_period'
    ]

    ds, _ = _read_mpas(mpas_xr_ds)

    # Set dummy attrs to test execution
    ds.attrs['mesh_id'] = "12345678"
    ds.attrs['is_periodic'] = "YES"
    ds.attrs['x_period'] = 1.0
    ds.attrs['y_period'] = 1.0

    uxgrid = ux.Grid(ds)

    # Check if all expected attributes are set
    for mpas_attr in expected_attrs:
        assert mpas_attr in uxgrid._ds.attrs

def test_face_area():
    """Tests the parsing of face areas for MPAS grids."""
    uxgrid_primal = ux.open_grid(mpas_grid_path, use_dual=False)
    uxgrid_dual = ux.open_grid(mpas_grid_path, use_dual=True)

    assert "face_areas" in uxgrid_primal._ds
    assert "face_areas" in uxgrid_dual._ds


def test_distance_units():
    xrds = xr.open_dataset(mpas_ocean_mesh)
    uxgrid = ux.open_grid(mpas_ocean_mesh)

    assert "edge_node_distances" in uxgrid._ds
    assert "edge_face_distances" in uxgrid._ds

    nt.assert_array_almost_equal(uxgrid['edge_node_distances'].values, (xrds['dvEdge'].values / xrds.attrs['sphere_radius']))
    nt.assert_array_almost_equal(uxgrid['edge_face_distances'].values, (xrds['dcEdge'].values / xrds.attrs['sphere_radius']))
