import uxarray as ux
import os
from unittest import TestCase
from pathlib import Path
import numpy as np

import numpy.testing as nt

import uxarray as ux
from uxarray.grid.coordinates import node_lonlat_rad_to_xyz
from uxarray.grid.integrate import _get_zonal_face_weight_rad

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestIntegrate(TestCase):
    gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"

    def test_single_dim(self):
        """Integral with 1D data mapped to each face."""
        uxgrid = ux.open_grid(self.gridfile_ne30)

        test_data = np.ones(uxgrid.n_face)

        dims = {"n_face": uxgrid.n_face}

        uxda = ux.UxDataArray(data=test_data,
                              dims=dims,
                              uxgrid=uxgrid,
                              name='var2')

        integral = uxda.integrate()

        # integration reduces the dimension by 1
        assert integral.ndim == len(dims) - 1

        nt.assert_almost_equal(integral, 4 * np.pi)

    def test_multi_dim(self):
        """Integral with 3D data mapped to each face."""
        uxgrid = ux.open_grid(self.gridfile_ne30)

        test_data = np.ones((5, 5, uxgrid.n_face))

        dims = {"a": 5, "b": 5, "n_face": uxgrid.n_face}

        uxda = ux.UxDataArray(data=test_data,
                              dims=dims,
                              uxgrid=uxgrid,
                              name='var2')

        integral = uxda.integrate()

        # integration reduces the dimension by 1
        assert integral.ndim == len(dims) - 1

        nt.assert_almost_equal(integral, np.ones((5, 5)) * 4 * np.pi)

class TestFaceWeights(TestCase):

    def test_get_zonal_face_weight_rad_GCA(self):
        """Test that the zonal face weights are correct."""
        vertices_lonlat = [[-0.4 * np.pi, 0.25 * np.pi], [-0.4 * np.pi, - 0.25 * np.pi],
                           [0.4*np.pi, -0.25 * np.pi], [0.4 * np.pi, 0.25 * np.pi]]
        vertices = [node_lonlat_rad_to_xyz(v) for v in vertices_lonlat]

        face_edge_nodes = np.array([[vertices[0], vertices[1]], [vertices[1], vertices[2]],
                           [vertices[2], vertices[3]], [vertices[3], vertices[0]]])

        # The latlon bounds for the latitude is not necessarily correct below since we don't use the latitudes bound anyway
        weight, overlap_flag = _get_zonal_face_weight_rad(face_edge_nodes, 0.20, np.array([[-0.25 * np.pi, 0.25 * np.pi],[1.6 * np.pi,0.4 * np.pi]]),is_directed=False)
        self.assertAlmostEqual(weight, 0.8 * np.pi, places=15)
        self.assertFalse(overlap_flag)

    def test_get_zonal_face_weight_rad_GCA_constLat(self):
        """Test that the zonal face weights are correct."""
        vertices_lonlat = [[-0.4 * np.pi, 0.25 * np.pi], [-0.4 * np.pi, - 0.25 * np.pi],
                           [0.4*np.pi, -0.25 * np.pi], [0.4 * np.pi, 0.25 * np.pi]]

        vertices = [node_lonlat_rad_to_xyz(v) for v in vertices_lonlat]

        face_edge_nodes = np.array([[vertices[0], vertices[1]], [vertices[1], vertices[2]],
                           [vertices[2], vertices[3]], [vertices[3], vertices[0]]])

        weight, overlap_flag = _get_zonal_face_weight_rad(  face_edge_nodes, np.sin(0.25 * np.pi)
                                                            , np.array([[-0.25 * np.pi, 0.25 * np.pi]
                                                            , [1.6 * np.pi,0.4 * np.pi]]),is_directed=False,
                                                            is_GCA_list=np.array([True, False, True, False]))
        self.assertAlmostEqual(weight, 0.8 * np.pi, places=15)
        self.assertTrue(overlap_flag)

    def test_get_zonal_face_weight_rad_equator(self):
        """Test that the zonal face weights are correct."""
        vertices_lonlat = [[-0.4 * np.pi, 0.25 * np.pi], [-0.4 * np.pi, 0.0],
                           [0.4*np.pi, 0.0], [0.4 * np.pi, 0.25 * np.pi]]

        vertices = [node_lonlat_rad_to_xyz(v) for v in vertices_lonlat]

        face_edge_nodes = np.array([[vertices[0], vertices[1]], [vertices[1], vertices[2]],
                           [vertices[2], vertices[3]], [vertices[3], vertices[0]]])

        weight, overlap_flag = _get_zonal_face_weight_rad(  face_edge_nodes, 0.0
                                                            , np.array([[-0.25 * np.pi, 0.25 * np.pi]
                                                            , [1.6 * np.pi,0.4 * np.pi]]),is_directed=False,
                                                            is_GCA_list=np.array([True, True, True, True]))
        self.assertAlmostEqual(weight, 0.8 * np.pi, places=15)
        self.assertTrue(overlap_flag)

        # Even if we change the is_GCA_list to False, the result should be the same
        weight, overlap_flag = _get_zonal_face_weight_rad(  face_edge_nodes, 0.0
                                                            , np.array([[-0.25 * np.pi, 0.25 * np.pi]
                                                            , [1.6 * np.pi,0.4 * np.pi]]),is_directed=False,
                                                            is_GCA_list=np.array([True, False, True, False]))
        self.assertAlmostEqual(weight, 0.8 * np.pi, places=15)
        self.assertTrue(overlap_flag)



