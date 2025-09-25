import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE


def test_face_areas_calculate_total_face_area_triangle(mesh_constants):
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
    nt.assert_almost_equal(area_triangular, mesh_constants['TRI_AREA'], decimal=1)

    # calculate area
    area_gaussian = grid_verts.calculate_total_face_area(
        quadrature_rule="gaussian", order=5, latitude_adjusted_area=True)
    nt.assert_almost_equal(area_gaussian, mesh_constants['CORRECTED_TRI_AREA'], decimal=3)


def test_face_areas_compute_face_areas_geoflow_small(gridpath):
    """Checks if the GeoFlow Small can generate a face areas output."""
    grid_geoflow = ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc"))
    grid_geoflow.compute_face_areas()


class TestFaceAreas:
    def test_face_areas_calculate_total_face_area_file(self, gridpath, mesh_constants):
        """Create a uxarray grid from vertices and saves an exodus file."""
        area = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug")).calculate_total_face_area()
        nt.assert_almost_equal(area, mesh_constants['MESH30_AREA'], decimal=3)

    def test_face_areas_calculate_total_face_area_sphere(self, gridpath, mesh_constants):
        """Computes the total face area of an MPAS mesh that lies on a unit sphere, with an expected total face area of 4pi."""
        mpas_grid_path = gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")

        primal_grid = ux.open_grid(mpas_grid_path, use_dual=False)
        dual_grid = ux.open_grid(mpas_grid_path, use_dual=True)

        primal_face_area = primal_grid.calculate_total_face_area()
        dual_face_area = dual_grid.calculate_total_face_area()

        nt.assert_almost_equal(primal_face_area, mesh_constants['UNIT_SPHERE_AREA'], decimal=3)
        nt.assert_almost_equal(dual_face_area, mesh_constants['UNIT_SPHERE_AREA'], decimal=3)

    def test_face_areas_verts_calc_area(self, gridpath, mesh_constants):
        faces_verts_ndarray = np.array([
            np.array([[150, 10, 0], [160, 20, 0], [150, 30, 0], [135, 30, 0],
                      [125, 20, 0], [135, 10, 0]]),
            np.array([[125, 20, 0], [135, 30, 0], [125, 60, 0], [110, 60, 0],
                      [100, 30, 0], [105, 20, 0]]),
            np.array([[95, 10, 0], [105, 20, 0], [100, 30, 0], [85, 30, 0],
                      [75, 20, 0], [85, 10, 0]]),
        ])
        verts_grid = ux.open_grid(faces_verts_ndarray, latlon=True)
        face_verts_areas = verts_grid.face_areas
        nt.assert_almost_equal(face_verts_areas.sum(), mesh_constants['FACE_VERTS_AREA'], decimal=3)


def test_latlon_bounds_populate_bounds_GCA_mix():
    """Test bounds population with mixed GCA faces."""
    face_1 = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    face_2 = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
    face_3 = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
    face_4 = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]

    faces = [face_1, face_2, face_3, face_4]

    expected_bounds = [[[0.17453293, 1.07370494], [0.17453293, 0.87266463]],
                       [[0.17453293, 1.10714872], [6.10865238, 0.87266463]],
                       [[1.04719755, 1.57079633], [3.66519143, 0.52359878]],
                       [[1.04719755, 1.57079633], [0., 6.28318531]]]

    grid = ux.Grid.from_face_vertices(faces, latlon=True)
    bounds_xarray = grid.bounds
    nt.assert_allclose(bounds_xarray.values, expected_bounds, atol=ERROR_TOLERANCE)


def test_latlon_bounds_populate_bounds_MPAS(gridpath):
    """Test bounds population with MPAS grid."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))
    bounds_xarray = uxgrid.bounds
