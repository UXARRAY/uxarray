import uxarray as ux
import os

import numpy.testing as nt
import numpy as np

import pytest

from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


csne30_grid_path = current_path / 'meshfiles' / "ugrid" / "outCSne30" / "outCSne30.ug"
csne30_data_path = current_path / 'meshfiles' / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"

quad_hex_grid_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / "grid.nc"
quad_hex_data_path_face_centered = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / "data.nc"
quad_hex_data_path_edge_centered = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / "random-edge-data.nc"


def test_quad_hex_face_centered():
    """Compares the weighted average computation for the quad hexagon grid
    using a face centered data variable to the expected value computed by
    hand."""
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path_face_centered)

    # expected weighted average computed by hand
    expected_weighted_mean = 297.55

    # compute the weighted mean
    result = uxds['t2m'].weighted_mean()

    # ensure values are within 3 decimal points of each other
    nt.assert_almost_equal(result.values, expected_weighted_mean, decimal=3)

def test_quad_hex_edge_centered():
    """Compares the weighted average computation for the quad hexagon grid
    using an edge centered data variable to the expected value computed by
    hand."""
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path_edge_centered)

    # expected weighted average computed by hand
    # expected_weighted_mean = 297.55
    expected_weighted_mean = (uxds['random_data_edge'].values * uxds.uxgrid.edge_node_distances).sum() / uxds.uxgrid.edge_node_distances.sum()

    # compute the weighted mean
    result = uxds['random_data_edge'].weighted_mean()

    nt.assert_equal(result, expected_weighted_mean)


def test_csne30_equal_area():
    """Compute the weighted average with a grid that has equal-area faces and
    compare the result to the regular mean."""
    uxds = ux.open_dataset(csne30_grid_path, csne30_data_path)
    face_areas = uxds.uxgrid.face_areas

    # set the area of each face to be one
    uxds.uxgrid._ds['face_areas'].data = np.ones(uxds.uxgrid.n_face)


    weighted_mean = uxds['psi'].weighted_mean()
    unweighted_mean = uxds['psi'].mean()

    # with equal area, both should be equal
    nt.assert_equal(weighted_mean, unweighted_mean)
