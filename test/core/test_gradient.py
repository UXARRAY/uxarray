from uxarray.io._mpas import _replace_padding, _replace_zeros, _to_zero_index
from uxarray.io._mpas import _read_mpas
import uxarray as ux
import numpy as np
import numpy.testing as nt
import os
from pathlib import Path
from uxarray.constants import INT_FILL_VALUE

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *

CSne30_grid_path = OUTCSNE30_GRID
CSne30_data_path = OUTCSNE30_VORTEX
geoflow_data_path = GEOFLOW_V1

def test_uniform_data():
    """Computes the gradient on meshes with uniform data, with the expected
    gradient being zero on all edges."""
    for grid_path in [
            MPAS_QU_MESH, MPAS_OCEAN_MESH, CSne30_grid_path, QUAD_HEXAGON_GRID,
    ]:
        uxgrid = ux.open_grid(grid_path)

        uxda_zeros = ux.UxDataArray(data=np.zeros(uxgrid.n_face),
                                     uxgrid=uxgrid,
                                     name="zeros",
                                     dims=['n_face'])

        zero_grad = uxda_zeros.gradient()
        # numpy.testing
        nt.assert_array_equal(zero_grad.values, np.zeros(uxgrid.n_edge))

        uxda_ones = ux.UxDataArray(data=np.ones(uxgrid.n_face),
                                    uxgrid=uxgrid,
                                    name="ones",
                                    dims=['n_face'])

        one_grad = uxda_ones.gradient()
        nt.assert_array_equal(one_grad.values, np.zeros(uxgrid.n_edge))

def test_quad_hex():
    """Computes the gradient on a mesh of 4 hexagons."""
    uxds = ux.open_dataset(QUAD_HEXAGON_GRID, QUAD_HEXAGON_DATA)
    grad = uxds['t2m'].gradient()

    for i, edge in enumerate(uxds.uxgrid.edge_face_connectivity.values):
        if INT_FILL_VALUE in edge:
            assert grad.values[i] == 0
        else:
            assert grad.values[i] != 0

    expected_values = np.array([28.963, 13.100, 14.296, 0, 0, 0, 0, 67.350, 0, 85.9397, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    nt.assert_almost_equal(grad.values, expected_values, 1e-2)

def test_normalization():
    """Tests the normalization gradient values."""
    uxds = ux.open_dataset(QUAD_HEXAGON_GRID, QUAD_HEXAGON_DATA)
    grad_l2_norm = uxds['t2m'].gradient(normalize=True)

    assert np.isclose(np.sum(grad_l2_norm.values**2), 1)

def test_grad_multi_dim():
    uxgrid = ux.open_grid(QUAD_HEXAGON_GRID)
    sample_data = np.random.randint(-10, 10, (5, 5, 4))
    uxda = ux.UxDataArray(uxgrid=uxgrid,
                          data=sample_data,
                          dims=["time", "lev", "n_face"])

    grad = uxda.gradient(normalize=True)
    assert grad.shape[:-1] == uxda.shape[:-1]

def test_face_centered_difference():
    uxds = ux.open_dataset(CSne30_grid_path, CSne30_data_path)
    uxda_diff = uxds['psi'].difference(destination='edge')

    assert uxda_diff._edge_centered()

    uxds = ux.open_dataset(MPAS_QU_MESH, MPAS_QU_MESH)
    uxda_diff = uxds['areaCell'].difference(destination='edge')

    assert uxda_diff._edge_centered()

def test_node_centered_difference():
    uxds = ux.open_dataset(GEOFLOW_GRID, geoflow_data_path)
    uxda_diff = uxds['v1'][0][0].difference(destination='edge')

    assert uxda_diff._edge_centered()

def test_hexagon():
    uxds = ux.open_dataset(QUAD_HEXAGON_GRID, QUAD_HEXAGON_DATA)
    uxda_diff = uxds['t2m'].difference(destination='edge')

    assert len(np.nonzero(uxda_diff.values)[0]) == uxds.uxgrid.n_face + 1
