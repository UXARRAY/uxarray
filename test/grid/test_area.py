import os
from pathlib import Path

import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux

import sys
sys.path.append(str(Path(__file__).parent.parent))
import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parent

gridfile_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"


def test_face_areas_calculate_total_face_area_triangle():
    """Create a uxarray grid from vertices and saves an exodus file."""
    verts = [
        [[0.02974582, -0.74469018, 0.66674712],
         [0.1534193, -0.88744577, 0.43462917],
         [0.18363692, -0.72230586, 0.66674712]]
    ]

    grid_verts = ux.open_grid(verts, latlon=False)

    # validate the grid
    assert grid_verts.validate()

    # calculate area without correction
    area_triangular = grid_verts.calculate_total_face_area(
        quadrature_rule="triangular", order=4)
    nt.assert_almost_equal(area_triangular, constants.TRI_AREA, decimal=1)

    # calculate area
    area_gaussian = grid_verts.calculate_total_face_area(
        quadrature_rule="gaussian", order=5, latitude_adjusted_area=True)
    nt.assert_almost_equal(area_gaussian, constants.CORRECTED_TRI_AREA, decimal=3)


def test_face_areas_calculate_total_face_area_file():
    """Create a uxarray grid from vertices and saves an exodus file."""
    area = ux.open_grid(gridfile_CSne30).calculate_total_face_area()
    nt.assert_almost_equal(area, constants.MESH30_AREA, decimal=3)


def test_face_areas_calculate_total_face_area_sphere():
    """Computes the total face area of an MPAS mesh that lies on a unit sphere, with an expected total face area of 4pi."""
    mpas_grid_path = current_path.parent / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'

    primal_grid = ux.open_grid(mpas_grid_path, use_dual=False)
    dual_grid = ux.open_grid(mpas_grid_path, use_dual=True)

    primal_face_area = primal_grid.calculate_total_face_area()
    dual_face_area = dual_grid.calculate_total_face_area()

    nt.assert_almost_equal(primal_face_area, constants.UNIT_SPHERE_AREA, decimal=3)
    nt.assert_almost_equal(dual_face_area, constants.UNIT_SPHERE_AREA, decimal=3)


def test_face_areas_compute_face_areas_geoflow_small():
    """Checks if the GeoFlow Small can generate a face areas output."""
    grid_geoflow = ux.open_grid(gridfile_geoflow)
    grid_geoflow.compute_face_areas()


def test_face_areas_verts_calc_area():
    """Create a uxarray grid from vertices and calculate area."""
    verts = [
        [[0.57735027, -0.57735027, 0.57735027],
         [0.57735027, 0.57735027, 0.57735027],
         [-0.57735027, 0.57735027, 0.57735027]]
    ]

    grid_verts = ux.open_grid(verts, latlon=False)

    # validate the grid
    assert grid_verts.validate()

    # calculate area
    area = grid_verts.calculate_total_face_area()
    nt.assert_almost_equal(area, constants.TRI_AREA, decimal=3)
