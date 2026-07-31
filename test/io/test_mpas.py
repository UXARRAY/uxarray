import numpy as np
import pytest

import uxarray as ux
import xarray as xr
import numpy.testing as nt
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.io._mpas import _replace_padding, _replace_zeros, _to_zero_index, _read_mpas





@pytest.fixture
def mpas_xr_ds(gridpath):
    return xr.open_dataset(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))


# Fill value
fv = INT_FILL_VALUE



def test_read_primal(mpas_xr_ds):
    mpas_primal_ugrid, _ = _read_mpas(mpas_xr_ds, use_dual=False)

    assert "face_edge_connectivity" in mpas_primal_ugrid
    assert "face_node_connectivity" in mpas_primal_ugrid
    assert "node_edge_connectivity" in mpas_primal_ugrid
    assert "edge_face_connectivity" in mpas_primal_ugrid
    assert "node_face_connectivity" in mpas_primal_ugrid
    assert "edge_node_connectivity" in mpas_primal_ugrid
    assert "face_face_connectivity" in mpas_primal_ugrid

def test_read_dual(mpas_xr_ds):
    mpas_dual_ugrid, _ = _read_mpas(mpas_xr_ds, use_dual=True)

def test_mpas_to_grid(gridpath):
    """Tests creation of Grid object from converted MPAS dataset."""
    mpas_uxgrid_primal = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=False)
    mpas_uxgrid_dual = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=True)
    mpas_uxgrid_dual.__repr__()

def test_primal_to_ugrid_conversion(mpas_xr_ds, gridpath):
    """Verifies that the Primal-Mesh was converted properly."""
    for path in [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), gridpath("mpas", "QU", "oQU480.231010.nc")]:
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

def test_dual_to_ugrid_conversion(gridpath):
    """Verifies that the Dual-Mesh was converted properly."""
    for path in [gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), gridpath("mpas", "QU", "oQU480.231010.nc")]:
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

def test_set_attrs(mpas_xr_ds):
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

def test_face_area(gridpath):
    """Tests the parsing of face areas for MPAS grids."""
    uxgrid_primal = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=False)
    uxgrid_dual = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=True)

    assert "face_areas" in uxgrid_primal._ds
    assert "face_areas" in uxgrid_dual._ds


def test_distance_units(gridpath):
    xrds = xr.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"))
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))

    assert "edge_node_distances" in uxgrid._ds
    assert "edge_face_distances" in uxgrid._ds

    nt.assert_array_almost_equal(uxgrid['edge_node_distances'].values, (xrds['dvEdge'].values / xrds.attrs['sphere_radius']))
    nt.assert_array_almost_equal(uxgrid['edge_face_distances'].values, (xrds['dcEdge'].values / xrds.attrs['sphere_radius']))


def test_ocean_mesh_normalization(gridpath):
    """Test that MPAS ocean mesh with non-unit sphere radius is properly normalized."""
    # Ocean mesh has sphere_radius = 6371229.0 meters
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"), use_dual=False)

    # Check node coordinates are normalized to unit sphere
    node_x = uxgrid._ds['node_x'].values
    node_y = uxgrid._ds['node_y'].values
    node_z = uxgrid._ds['node_z'].values

    # Compute radius for each node
    radii = np.sqrt(node_x**2 + node_y**2 + node_z**2)

    # All radii should be 1.0 (unit sphere)
    nt.assert_array_almost_equal(radii, np.ones_like(radii), decimal=10)

    # Check face areas are normalized for unit sphere
    # Ocean mesh only covers ~70% of sphere (ocean area)
    face_areas = uxgrid.face_areas.values

    # Check that all face areas are positive
    assert np.all(face_areas > 0), "All face areas should be positive"

    # Total area should be less than full sphere (4*pi) but reasonable
    total_area = np.sum(face_areas)
    full_sphere_area = 4.0 * np.pi
    assert 0.5 < total_area < full_sphere_area, f"Total area {total_area} should be between 0.5 and {full_sphere_area}"

    # Check that individual face areas are reasonable for unit sphere
    max_face_area = np.max(face_areas)
    min_face_area = np.min(face_areas)
    assert max_face_area < 0.1 * full_sphere_area, "Maximum face area seems too large for unit sphere"
    assert min_face_area > 1e-10, "Minimum face area seems too small"

    # Check edge lengths are normalized for unit sphere
    if "edge_node_distances" in uxgrid._ds:
        edge_lengths = uxgrid._ds["edge_node_distances"].values

        # All edge lengths should be positive and <= pi
        assert np.all(edge_lengths > 0), "All edge lengths should be positive"
        assert np.max(edge_lengths) <= np.pi, "Edge lengths should not exceed pi on unit sphere"


def test_grid_normalization(gridpath):
    """Test that MPAS grid coordinates are properly normalized."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=False)

    # Check node coordinates are normalized
    node_lon = uxgrid._ds['node_lon'].values
    node_lat = uxgrid._ds['node_lat'].values
    node_x = uxgrid._ds['node_x'].values
    node_y = uxgrid._ds['node_y'].values
    node_z = uxgrid._ds['node_z'].values

    # Compute radius for each node
    radii = np.sqrt(node_x**2 + node_y**2 + node_z**2)

    # All radii should be 1.0 (unit sphere)
    nt.assert_array_almost_equal(radii, np.ones_like(radii), decimal=10)

    # Check face areas are normalized for unit sphere
    # Total surface area of unit sphere is 4*pi
    expected_total_area = 4.0 * np.pi

    # Get face areas
    face_areas = uxgrid.face_areas.values

    # Check that all face areas are positive
    assert np.all(face_areas > 0), "All face areas should be positive"

    # Check that total area equals surface area of unit sphere
    total_area = np.sum(face_areas)
    nt.assert_almost_equal(total_area, expected_total_area, decimal=7)

    # Check that face areas are reasonable (not too small or too large)
    # For a unit sphere, face areas should be much smaller than total area
    max_face_area = np.max(face_areas)
    min_face_area = np.min(face_areas)

    assert max_face_area < 0.1 * expected_total_area, "Maximum face area seems too large for unit sphere"
    assert min_face_area > 1e-10, "Minimum face area seems too small"

    # Check edge lengths are normalized for unit sphere
    # Edge lengths should be reasonable for a unit sphere (max possible is pi for antipodal points)
    if "edge_node_distances" in uxgrid._ds:
        edge_lengths = uxgrid._ds["edge_node_distances"].values

        # All edge lengths should be positive
        assert np.all(edge_lengths > 0), "All edge lengths should be positive"

        # Maximum edge length on unit sphere cannot exceed pi
        assert np.max(edge_lengths) <= np.pi, "Edge lengths should not exceed pi on unit sphere"

        # Minimum edge length should be reasonable
        assert np.min(edge_lengths) > 1e-10, "Minimum edge length seems too small"
