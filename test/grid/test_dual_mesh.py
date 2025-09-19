import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux


def test_dual_mesh_mpas(gridpath):
    """Test dual mesh creation for MPAS grid."""
    grid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=False)
    mpas_dual = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=True)

    dual = grid.get_dual()

    assert dual.n_face == mpas_dual.n_face
    assert dual.n_node == mpas_dual.n_node
    assert dual.n_max_face_nodes == mpas_dual.n_max_face_nodes

    nt.assert_equal(dual.face_node_connectivity.values, mpas_dual.face_node_connectivity.values)

def test_dual_duplicate(gridpath):
    """Test dual mesh with duplicate dataset."""
    dataset = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), gridpath("ugrid", "geoflow-small", "grid.nc"))
    with pytest.raises(RuntimeError):
        dataset.get_dual()

def test_dual_mesh_basic(gridpath):
    """Test basic dual mesh creation."""
    # Use a real grid file for dual mesh testing
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    # Create dual mesh
    dual = uxgrid.get_dual()

    # Basic validation
    assert dual.n_face > 0
    assert dual.n_node > 0

def test_dual_mesh_properties(gridpath):
    """Test dual mesh properties."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # Create dual mesh
    dual = uxgrid.get_dual()

    # Dual should have different structure
    assert dual.n_face != uxgrid.n_face
    assert dual.n_node != uxgrid.n_node

    # But should be valid
    assert dual.validate()

def test_dual_mesh_connectivity(gridpath):
    """Test dual mesh connectivity."""
    uxgrid = ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc"))

    # Create dual mesh
    dual = uxgrid.get_dual()

    # Should have valid connectivity
    assert hasattr(dual, 'face_node_connectivity')
    assert dual.face_node_connectivity.shape[0] == dual.n_face

    # Check connectivity values are valid
    face_node_conn = dual.face_node_connectivity.values
    valid_nodes = face_node_conn[face_node_conn != ux.INT_FILL_VALUE]
    assert np.all(valid_nodes >= 0)
    assert np.all(valid_nodes < dual.n_node)
