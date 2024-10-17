import uxarray as ux
import os
from unittest import TestCase
from pathlib import Path
import numpy as np
import pandas as pd

import numpy.testing as nt

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.integrate import _get_zonal_face_interval, _process_overlapped_intervals, _get_zonal_faces_weight_at_constLat,_get_faces_constLat_intersection_info


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
    gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"

    def test_get_faces_constLat_intersection_info_one_intersection(self):
        face_edges_cart = np.array([
            [[-5.4411371445381629e-01, -4.3910468172333759e-02, -8.3786164521844386e-01],
             [-5.4463903501502697e-01, -6.6699045092185599e-17, -8.3867056794542405e-01]],

            [[-5.4463903501502697e-01, -6.6699045092185599e-17, -8.3867056794542405e-01],
             [-4.9999999999999994e-01, -6.1232339957367648e-17, -8.6602540378443871e-01]],

            [[-4.9999999999999994e-01, -6.1232339957367648e-17, -8.6602540378443871e-01],
             [-4.9948581138450826e-01, -4.5339793804534498e-02, -8.6513480297773349e-01]],

            [[-4.9948581138450826e-01, -4.5339793804534498e-02, -8.6513480297773349e-01],
             [-5.4411371445381629e-01, -4.3910468172333759e-02, -8.3786164521844386e-01]]
        ])

        latitude_cart = -0.8660254037844386
        is_directed=False
        is_latlonface=False
        is_GCA_list=None
        unique_intersections, pt_lon_min, pt_lon_max = _get_faces_constLat_intersection_info(face_edges_cart, latitude_cart, is_GCA_list, is_latlonface, is_directed)
        # The expected unique_intersections length is 1
        self.assertEqual(len(unique_intersections), 1)

    def test_get_faces_constLat_intersection_info_encompass_pole(self):
        face_edges_cart = np.array([
            [[0.03982285692494229, 0.00351700770436231, 0.9992005658140627],
             [0.00896106681877875, 0.03896060263227105, 0.9992005658144913]],

            [[0.00896106681877875, 0.03896060263227105, 0.9992005658144913],
             [-0.03428461218295055, 0.02056197086916728, 0.9992005658132106]],

            [[-0.03428461218295055, 0.02056197086916728, 0.9992005658132106],
             [-0.03015012448894485, -0.02625260499902213, 0.9992005658145248]],

            [[-0.03015012448894485, -0.02625260499902213, 0.9992005658145248],
             [0.01565081128889155, -0.03678697293262131, 0.9992005658167203]],

            [[0.01565081128889155, -0.03678697293262131, 0.9992005658167203],
             [0.03982285692494229, 0.00351700770436231, 0.9992005658140627]]
        ])

        latitude_cart = 0.9993908270190958
        # Convert the latitude to degrees
        latitude_rad = np.arcsin(latitude_cart)
        latitude_deg = np.rad2deg(latitude_rad)
        print(latitude_deg)
        is_directed=False
        is_latlonface=False
        is_GCA_list=None
        unique_intersections, pt_lon_min, pt_lon_max = _get_faces_constLat_intersection_info(face_edges_cart, latitude_cart, is_GCA_list, is_latlonface, is_directed)
        # The expected unique_intersections length should be no greater than 2* n_edges
        self.assertLessEqual(len(unique_intersections), 2*len(face_edges_cart))

    def test_get_faces_constLat_intersection_info_on_pole(self):
        face_edges_cart = np.array([
            [[-5.2264427688714095e-02, -5.2264427688714102e-02, -9.9726468863423734e-01],
             [-5.2335956242942412e-02, -6.4093061293235361e-18, -9.9862953475457394e-01]],

            [[-5.2335956242942412e-02, -6.4093061293235361e-18, -9.9862953475457394e-01],
             [6.1232339957367660e-17, 0.0000000000000000e+00, -1.0000000000000000e+00]],

            [[6.1232339957367660e-17, 0.0000000000000000e+00, -1.0000000000000000e+00],
             [3.2046530646617680e-18, -5.2335956242942412e-02, -9.9862953475457394e-01]],

            [[3.2046530646617680e-18, -5.2335956242942412e-02, -9.9862953475457394e-01],
             [-5.2264427688714095e-02, -5.2264427688714102e-02, -9.9726468863423734e-01]]
        ])
        latitude_cart = -0.9998476951563913
        is_directed=False
        is_latlonface=False
        is_GCA_list=None
        unique_intersections, pt_lon_min, pt_lon_max = _get_faces_constLat_intersection_info(face_edges_cart, latitude_cart, is_GCA_list, is_latlonface, is_directed)
        # The expected unique_intersections length is 2
        self.assertEqual(len(unique_intersections), 2)


    def test_get_faces_constLat_intersection_info_near_pole(self):
        face_edges_cart = np.array([
            [[-5.1693346290592648e-02, 1.5622531297347531e-01, -9.8636780641686628e-01],
             [-5.1195320928843470e-02, 2.0763904784932552e-01, -9.7686491641537532e-01]],
            [[-5.1195320928843470e-02, 2.0763904784932552e-01, -9.7686491641537532e-01],
             [1.2730919333264125e-17, 2.0791169081775882e-01, -9.7814760073380580e-01]],
            [[1.2730919333264125e-17, 2.0791169081775882e-01, -9.7814760073380580e-01],
             [9.5788483443923397e-18, 1.5643446504023048e-01, -9.8768834059513777e-01]],
            [[9.5788483443923397e-18, 1.5643446504023048e-01, -9.8768834059513777e-01],
             [-5.1693346290592648e-02, 1.5622531297347531e-01, -9.8636780641686628e-01]]
        ])

        latitude_cart = -0.9876883405951378
        latitude_rad = np.arcsin(latitude_cart)
        latitude_deg = np.rad2deg(latitude_rad)
        is_directed=False
        is_latlonface=False
        is_GCA_list=None
        unique_intersections, pt_lon_min, pt_lon_max = _get_faces_constLat_intersection_info(face_edges_cart, latitude_cart, is_GCA_list, is_latlonface, is_directed)
        # The expected unique_intersections length is 2
        self.assertEqual(len(unique_intersections), 1)


    def test_get_faces_constLat_intersection_info_2(self):
        """This might test the case where the calculated intersection points
        might suffer from floating point errors If not handled properly, the
        function might return more than 2 unique intersections."""

        face_edges_cart = np.array([[[0.6546536707079771, -0.37796447300922714, -0.6546536707079772],
                                     [0.6652465971273088, -0.33896007142593115, -0.6652465971273087]],

                                    [[0.6652465971273088, -0.33896007142593115, -0.6652465971273087],
                                     [0.6949903639307233, -0.3541152775760984, -0.6257721344312508]],

                                    [[0.6949903639307233, -0.3541152775760984, -0.6257721344312508],
                                     [0.6829382762718700, -0.39429459764546304, -0.6149203859609873]],

                                    [[0.6829382762718700, -0.39429459764546304, -0.6149203859609873],
                                     [0.6546536707079771, -0.37796447300922714, -0.6546536707079772]]])

        latitude_cart = -0.6560590289905073
        is_directed=False
        is_latlonface=False
        is_GCA_list=None
        unique_intersections, pt_lon_min, pt_lon_max = _get_faces_constLat_intersection_info(face_edges_cart, latitude_cart, is_GCA_list, is_latlonface, is_directed)
        # The expected unique_intersections length is 2
        self.assertEqual(len(unique_intersections), 2)


    def test_get_faces_constLat_intersection_info_2(self):
        """This might test the case where the calculated intersection points
        might suffer from floating point errors If not handled properly, the
        function might return more than 2 unique intersections."""

        face_edges_cart = np.array([[[0.6546536707079771, -0.37796447300922714, -0.6546536707079772],
                                     [0.6652465971273088, -0.33896007142593115, -0.6652465971273087]],

                                    [[0.6652465971273088, -0.33896007142593115, -0.6652465971273087],
                                     [0.6949903639307233, -0.3541152775760984, -0.6257721344312508]],

                                    [[0.6949903639307233, -0.3541152775760984, -0.6257721344312508],
                                     [0.6829382762718700, -0.39429459764546304, -0.6149203859609873]],

                                    [[0.6829382762718700, -0.39429459764546304, -0.6149203859609873],
                                     [0.6546536707079771, -0.37796447300922714, -0.6546536707079772]]])

        latitude_cart = -0.6560590289905073
        is_directed=False
        is_latlonface=False
        is_GCA_list=None
        unique_intersections, pt_lon_min, pt_lon_max = _get_faces_constLat_intersection_info(face_edges_cart, latitude_cart, is_GCA_list, is_latlonface, is_directed)
        # The expected unique_intersections length is 2
        self.assertEqual(len(unique_intersections), 2)

    def test_get_zonal_face_interval(self):
        """Test the _get_zonal_face_interval function for correct interval
        computation.

        This test verifies that the _get_zonal_face_interval function
        accurately computes the zonal face intervals given a set of face
        edge nodes and a constant latitude value (constZ). The expected
        intervals are compared against the calculated intervals to
        ensure correctness.
        """
        vertices_lonlat = [[1.6 * np.pi, 0.25 * np.pi],
                           [1.6 * np.pi, -0.25 * np.pi],
                           [0.4 * np.pi, -0.25 * np.pi],
                           [0.4 * np.pi, 0.25 * np.pi]]
        vertices = [_lonlat_rad_to_xyz(*v) for v in vertices_lonlat]

        face_edge_nodes = np.array([[vertices[0], vertices[1]],
                                    [vertices[1], vertices[2]],
                                    [vertices[2], vertices[3]],
                                    [vertices[3], vertices[0]]])

        constZ = np.sin(0.20)
        # The latlon bounds for the latitude is not necessarily correct below since we don't use the latitudes bound anyway
        interval_df = _get_zonal_face_interval(face_edge_nodes, constZ,
                                               np.array([[-0.25 * np.pi, 0.25 * np.pi], [1.6 * np.pi,
                                                                                         0.4 * np.pi]]),
                                               is_directed=False)
        expected_interval_df = pd.DataFrame({
            'start': [1.6 * np.pi, 0.0],
            'end': [2.0 * np.pi, 00.4 * np.pi]
        })
        # Sort both DataFrames by 'start' column before comparison
        expected_interval_df_sorted = expected_interval_df.sort_values(by='start').reset_index(drop=True)

        # Converting the sorted DataFrames to NumPy arrays
        actual_values_sorted = interval_df[['start', 'end']].to_numpy()
        expected_values_sorted = expected_interval_df_sorted[['start', 'end']].to_numpy()

        # Asserting almost equal arrays
        nt.assert_array_almost_equal(actual_values_sorted, expected_values_sorted, decimal=13)


    def test_get_zonal_face_interval_empty_interval(self):
        """Test the _get_zonal_face_interval function for cases where the
        interval is empty.

        This test verifies that the _get_zonal_face_interval function
        correctly returns an empty interval when the latitude only
        touches the face but does not intersect it.
        """
        face_edges_cart = np.array([
            [[-5.4411371445381629e-01, -4.3910468172333759e-02, -8.3786164521844386e-01],
             [-5.4463903501502697e-01, -6.6699045092185599e-17, -8.3867056794542405e-01]],

            [[-5.4463903501502697e-01, -6.6699045092185599e-17, -8.3867056794542405e-01],
             [-4.9999999999999994e-01, -6.1232339957367648e-17, -8.6602540378443871e-01]],

            [[-4.9999999999999994e-01, -6.1232339957367648e-17, -8.6602540378443871e-01],
             [-4.9948581138450826e-01, -4.5339793804534498e-02, -8.6513480297773349e-01]],

            [[-4.9948581138450826e-01, -4.5339793804534498e-02, -8.6513480297773349e-01],
             [-5.4411371445381629e-01, -4.3910468172333759e-02, -8.3786164521844386e-01]]
        ])

        latitude_cart = -0.8660254037844386
        face_latlon_bounds = np.array([
            [-1.04719755, -0.99335412],
            [3.14159265, 3.2321175]
        ])

        res = _get_zonal_face_interval(face_edges_cart, latitude_cart, face_latlon_bounds, is_directed=False)
        expected_res = pd.DataFrame({"start": [0.0], "end": [0.0]})
        pd.testing.assert_frame_equal(res, expected_res)

    def test_get_zonal_face_interval_encompass_pole(self):
        """Test the _get_zonal_face_interval function for cases where the face
        encompasses the pole inside."""
        face_edges_cart = np.array([
            [[0.03982285692494229, 0.00351700770436231, 0.9992005658140627],
             [0.00896106681877875, 0.03896060263227105, 0.9992005658144913]],

            [[0.00896106681877875, 0.03896060263227105, 0.9992005658144913],
             [-0.03428461218295055, 0.02056197086916728, 0.9992005658132106]],

            [[-0.03428461218295055, 0.02056197086916728, 0.9992005658132106],
             [-0.03015012448894485, -0.02625260499902213, 0.9992005658145248]],

            [[-0.03015012448894485, -0.02625260499902213, 0.9992005658145248],
             [0.01565081128889155, -0.03678697293262131, 0.9992005658167203]],

            [[0.01565081128889155, -0.03678697293262131, 0.9992005658167203],
             [0.03982285692494229, 0.00351700770436231, 0.9992005658140627]]
        ])

        latitude_cart = 0.9993908270190958

        face_latlon_bounds = np.array([
            [np.arcsin(0.9992005658145248), 0.5*np.pi],
            [0, 2*np.pi]
        ])
        # Expected result DataFrame
        expected_df = pd.DataFrame({
            'start': [0.000000, 1.101091, 2.357728, 3.614365, 4.871002, 6.127640],
            'end': [0.331721, 1.588358, 2.844995, 4.101632, 5.358270, 6.283185]
        })

        # Call the function to get the result
        res = _get_zonal_face_interval(face_edges_cart, latitude_cart, face_latlon_bounds, is_directed=False)

        # Assert the result matches the expected DataFrame
        pd.testing.assert_frame_equal(res,  expected_df)



    def test_get_zonal_face_interval_FILL_VALUE(self):
        """Test the _get_zonal_face_interval function for cases where there are
        dummy nodes."""
        dummy_node = [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE]
        vertices_lonlat = [[1.6 * np.pi, 0.25 * np.pi],
                           [1.6 * np.pi, -0.25 * np.pi],
                           [0.4 * np.pi, -0.25 * np.pi],
                           [0.4 * np.pi, 0.25 * np.pi]]
        vertices = [_lonlat_rad_to_xyz(*v) for v in vertices_lonlat]

        face_edge_nodes = np.array([[vertices[0], vertices[1]],
                                    [vertices[1], vertices[2]],
                                    [vertices[2], vertices[3]],
                                    [vertices[3], vertices[0]],
                                    [dummy_node,dummy_node]])

        constZ = np.sin(0.20)
        # The latlon bounds for the latitude is not necessarily correct below since we don't use the latitudes bound anyway
        interval_df = _get_zonal_face_interval(face_edge_nodes, constZ,
                                               np.array([[-0.25 * np.pi, 0.25 * np.pi], [1.6 * np.pi,
                                                                                         0.4 * np.pi]]),
                                               is_directed=False)
        expected_interval_df = pd.DataFrame({
            'start': [1.6 * np.pi, 0.0],
            'end': [2.0 * np.pi, 00.4 * np.pi]
        })
        # Sort both DataFrames by 'start' column before comparison
        expected_interval_df_sorted = expected_interval_df.sort_values(by='start').reset_index(drop=True)

        # Converting the sorted DataFrames to NumPy arrays
        actual_values_sorted = interval_df[['start', 'end']].to_numpy()
        expected_values_sorted = expected_interval_df_sorted[['start', 'end']].to_numpy()

        # Asserting almost equal arrays
        nt.assert_array_almost_equal(actual_values_sorted, expected_values_sorted, decimal=13)


    def test_get_zonal_face_interval_GCA_constLat(self):
        vertices_lonlat = [[-0.4 * np.pi, 0.25 * np.pi],
                           [-0.4 * np.pi, -0.25 * np.pi],
                           [0.4 * np.pi, -0.25 * np.pi],
                           [0.4 * np.pi, 0.25 * np.pi]]

        vertices = [_lonlat_rad_to_xyz(*v) for v in vertices_lonlat]

        face_edge_nodes = np.array([[vertices[0], vertices[1]],
                                    [vertices[1], vertices[2]],
                                    [vertices[2], vertices[3]],
                                    [vertices[3], vertices[0]]])

        constZ = np.sin(0.20 * np.pi)
        interval_df = _get_zonal_face_interval(face_edge_nodes, constZ,
                                               np.array([[-0.25 * np.pi, 0.25 * np.pi], [1.6 * np.pi,
                                                                                         0.4 * np.pi]]),
                                               is_directed=False, is_GCA_list=np.array([True, False, True, False]))
        expected_interval_df = pd.DataFrame({
            'start': [1.6 * np.pi, 0.0],
            'end': [2.0 * np.pi, 00.4 * np.pi]
        })
        # Sort both DataFrames by 'start' column before comparison
        expected_interval_df_sorted = expected_interval_df.sort_values(by='start').reset_index(drop=True)

        # Converting the sorted DataFrames to NumPy arrays
        actual_values_sorted = interval_df[['start', 'end']].to_numpy()
        expected_values_sorted = expected_interval_df_sorted[['start', 'end']].to_numpy()

        # Asserting almost equal arrays
        nt.assert_array_almost_equal(actual_values_sorted, expected_values_sorted, decimal=13)


    def test_get_zonal_face_interval_equator(self):
        """Test that the face interval is correctly computed when the latitude
        is at the equator."""
        vertices_lonlat = [[-0.4 * np.pi, 0.25 * np.pi], [-0.4 * np.pi, 0.0],
                           [0.4 * np.pi, 0.0], [0.4 * np.pi, 0.25 * np.pi]]

        vertices = [_lonlat_rad_to_xyz(*v) for v in vertices_lonlat]

        face_edge_nodes = np.array([[vertices[0], vertices[1]],
                                    [vertices[1], vertices[2]],
                                    [vertices[2], vertices[3]],
                                    [vertices[3], vertices[0]]])

        interval_df = _get_zonal_face_interval(face_edge_nodes, 0.0,
                                               np.array([[-0.25 * np.pi, 0.25 * np.pi], [1.6 * np.pi,
                                                                                         0.4 * np.pi]]),
                                               is_directed=False, is_GCA_list=np.array([True, True, True, True]))
        expected_interval_df = pd.DataFrame({
            'start': [1.6 * np.pi, 0.0],
            'end': [2.0 * np.pi, 00.4 * np.pi]
        })
        # Sort both DataFrames by 'start' column before comparison
        expected_interval_df_sorted = expected_interval_df.sort_values(by='start').reset_index(drop=True)

        # Converting the sorted DataFrames to NumPy arrays
        actual_values_sorted = interval_df[['start', 'end']].to_numpy()
        expected_values_sorted = expected_interval_df_sorted[['start', 'end']].to_numpy()

        # Asserting almost equal arrays
        nt.assert_array_almost_equal(actual_values_sorted, expected_values_sorted, decimal=13)

        # Even if we change the is_GCA_list to False, the result should be the same
        interval_df = _get_zonal_face_interval(face_edge_nodes, 0.0,
                                               np.array([[-0.25 * np.pi, 0.25 * np.pi], [1.6 * np.pi,
                                                                                         0.4 * np.pi]]),
                                               is_directed=False, is_GCA_list=np.array([True, False, True, False]))
        expected_interval_df = pd.DataFrame({
            'start': [1.6 * np.pi, 0.0],
            'end': [2.0 * np.pi, 00.4 * np.pi]
        })

        # Sort both DataFrames by 'start' column before comparison
        expected_interval_df_sorted = expected_interval_df.sort_values(by='start').reset_index(drop=True)

        # Converting the sorted DataFrames to NumPy arrays
        actual_values_sorted = interval_df[['start', 'end']].to_numpy()
        expected_values_sorted = expected_interval_df_sorted[['start', 'end']].to_numpy()

        # Asserting almost equal arrays
        nt.assert_array_almost_equal(actual_values_sorted, expected_values_sorted, decimal=13)


    def test_process_overlapped_intervals_overlap_and_gap(self):
        # Test intervals data that has overlapping intervals and gap
        intervals_data = [
            {
                'start': 0.0,
                'end': 100.0,
                'face_index': 0
            },
            {
                'start': 50.0,
                'end': 150.0,
                'face_index': 1
            },
            {
                'start': 140.0,
                'end': 150.0,
                'face_index': 2
            },
            {
                'start': 150.0,
                'end': 250.0,
                'face_index': 3
            },
            {
                'start': 260.0,
                'end': 350.0,
                'face_index': 4
            },
        ]

        df = pd.DataFrame(intervals_data)
        df['interval'] = df.apply(lambda row: pd.Interval(
            left=row['start'], right=row['end'], closed='both'),
                                  axis=1)
        df['interval'] = df['interval'].astype('interval[float64]')

        # Expected result
        expected_overlap_contributions = np.array({
            0: 75.0,
            1: 70.0,
            2: 5.0,
            3: 100.0,
            4: 90.0
        })
        overlap_contributions, total_length = _process_overlapped_intervals(df)
        self.assertEqual(total_length, 340.0)
        nt.assert_array_equal(overlap_contributions,
                              expected_overlap_contributions)


    def test_process_overlapped_intervals_antimerdian(self):
        intervals_data = [
            {
                'start': 350.0,
                'end': 360.0,
                'face_index': 0
            },
            {
                'start': 0.0,
                'end': 100.0,
                'face_index': 0
            },
            {
                'start': 100.0,
                'end': 150.0,
                'face_index': 1
            },
            {
                'start': 100.0,
                'end': 300.0,
                'face_index': 2
            },
            {
                'start': 310.0,
                'end': 360.0,
                'face_index': 3
            },
        ]

        df = pd.DataFrame(intervals_data)
        df['interval'] = df.apply(lambda row: pd.Interval(
            left=row['start'], right=row['end'], closed='both'),
                                  axis=1)
        df['interval'] = df['interval'].astype('interval[float64]')

        # Expected result
        expected_overlap_contributions = np.array({
            0: 105.0,
            1: 25.0,
            2: 175.0,
            3: 45.0
        })
        overlap_contributions, total_length = _process_overlapped_intervals(df)
        self.assertEqual(total_length, 350.0)
        nt.assert_array_equal(overlap_contributions,
                              expected_overlap_contributions)


    def test_get_zonal_faces_weight_at_constLat_equator(self):
        face_0 = [[1.7 * np.pi, 0.25 * np.pi], [1.7 * np.pi, 0.0],
                  [0.3 * np.pi, 0.0], [0.3 * np.pi, 0.25 * np.pi]]
        face_1 = [[0.3 * np.pi, 0.0], [0.3 * np.pi, -0.25 * np.pi],
                  [0.6 * np.pi, -0.25 * np.pi], [0.6 * np.pi, 0.0]]
        face_2 = [[0.3 * np.pi, 0.25 * np.pi], [0.3 * np.pi, 0.0], [np.pi, 0.0],
                  [np.pi, 0.25 * np.pi]]
        face_3 = [[0.7 * np.pi, 0.0], [0.7 * np.pi, -0.25 * np.pi],
                  [np.pi, -0.25 * np.pi], [np.pi, 0.0]]

        # Convert the face vertices to xyz coordinates
        face_0 = [_lonlat_rad_to_xyz(*v) for v in face_0]
        face_1 = [_lonlat_rad_to_xyz(*v) for v in face_1]
        face_2 = [_lonlat_rad_to_xyz(*v) for v in face_2]
        face_3 = [_lonlat_rad_to_xyz(*v) for v in face_3]

        face_0_edge_nodes = np.array([[face_0[0], face_0[1]],
                                      [face_0[1], face_0[2]],
                                      [face_0[2], face_0[3]],
                                      [face_0[3], face_0[0]]])
        face_1_edge_nodes = np.array([[face_1[0], face_1[1]],
                                      [face_1[1], face_1[2]],
                                      [face_1[2], face_1[3]],
                                      [face_1[3], face_1[0]]])
        face_2_edge_nodes = np.array([[face_2[0], face_2[1]],
                                      [face_2[1], face_2[2]],
                                      [face_2[2], face_2[3]],
                                      [face_2[3], face_2[0]]])
        face_3_edge_nodes = np.array([[face_3[0], face_3[1]],
                                      [face_3[1], face_3[2]],
                                      [face_3[2], face_3[3]],
                                      [face_3[3], face_3[0]]])

        face_0_latlon_bound = np.array([[0.0, 0.25 * np.pi],
                                        [1.7 * np.pi, 0.3 * np.pi]])
        face_1_latlon_bound = np.array([[-0.25 * np.pi, 0.0],
                                        [0.3 * np.pi, 0.6 * np.pi]])
        face_2_latlon_bound = np.array([[0.0, 0.25 * np.pi],
                                        [0.3 * np.pi, np.pi]])
        face_3_latlon_bound = np.array([[-0.25 * np.pi, 0.0],
                                        [0.7 * np.pi, np.pi]])

        latlon_bounds = np.array([
            face_0_latlon_bound, face_1_latlon_bound, face_2_latlon_bound,
            face_3_latlon_bound
        ])

        expected_weight_df = pd.DataFrame({
            'face_index': [0, 1, 2, 3],
            'weight': [0.46153, 0.11538, 0.30769, 0.11538]
        })

        # Assert the results is the same to the 3 decimal places
        weight_df = _get_zonal_faces_weight_at_constLat(np.array([
            face_0_edge_nodes, face_1_edge_nodes, face_2_edge_nodes,
            face_3_edge_nodes
        ]), 0.0, latlon_bounds, is_directed=False)

        nt.assert_array_almost_equal(weight_df, expected_weight_df, decimal=3)


    def test_get_zonal_faces_weight_at_constLat_regular(self):
        face_0 = [[1.7 * np.pi, 0.25 * np.pi], [1.7 * np.pi, 0.0],
                  [0.3 * np.pi, 0.0], [0.3 * np.pi, 0.25 * np.pi]]
        face_1 = [[0.4 * np.pi, 0.3 * np.pi], [0.4 * np.pi, 0.0],
                  [0.5 * np.pi, 0.0], [0.5 * np.pi, 0.3 * np.pi]]
        face_2 = [[0.5 * np.pi, 0.25 * np.pi], [0.5 * np.pi, 0.0], [np.pi, 0.0],
                  [np.pi, 0.25 * np.pi]]
        face_3 = [[1.2 * np.pi, 0.25 * np.pi], [1.2 * np.pi, 0.0],
                  [1.6 * np.pi, -0.01 * np.pi], [1.6 * np.pi, 0.25 * np.pi]]

        # Convert the face vertices to xyz coordinates
        face_0 = [_lonlat_rad_to_xyz(*v) for v in face_0]
        face_1 = [_lonlat_rad_to_xyz(*v) for v in face_1]
        face_2 = [_lonlat_rad_to_xyz(*v) for v in face_2]
        face_3 = [_lonlat_rad_to_xyz(*v) for v in face_3]

        face_0_edge_nodes = np.array([[face_0[0], face_0[1]],
                                      [face_0[1], face_0[2]],
                                      [face_0[2], face_0[3]],
                                      [face_0[3], face_0[0]]])
        face_1_edge_nodes = np.array([[face_1[0], face_1[1]],
                                      [face_1[1], face_1[2]],
                                      [face_1[2], face_1[3]],
                                      [face_1[3], face_1[0]]])
        face_2_edge_nodes = np.array([[face_2[0], face_2[1]],
                                      [face_2[1], face_2[2]],
                                      [face_2[2], face_2[3]],
                                      [face_2[3], face_2[0]]])
        face_3_edge_nodes = np.array([[face_3[0], face_3[1]],
                                      [face_3[1], face_3[2]],
                                      [face_3[2], face_3[3]],
                                      [face_3[3], face_3[0]]])

        face_0_latlon_bound = np.array([[0.0, 0.25 * np.pi],
                                        [1.7 * np.pi, 0.3 * np.pi]])
        face_1_latlon_bound = np.array([[0, 0.3 * np.pi],
                                        [0.4 * np.pi, 0.5 * np.pi]])
        face_2_latlon_bound = np.array([[0.0, 0.25 * np.pi],
                                        [0.5 * np.pi, np.pi]])
        face_3_latlon_bound = np.array([[-0.01 * np.pi, 0.25 * np.pi],
                                        [1.2 * np.pi, 1.6 * np.pi]])

        latlon_bounds = np.array([
            face_0_latlon_bound, face_1_latlon_bound, face_2_latlon_bound,
            face_3_latlon_bound
        ])

        expected_weight_df = pd.DataFrame({
            'face_index': [0, 1, 2, 3],
            'weight': [0.375, 0.0625, 0.3125, 0.25]
        })

        # Assert the results is the same to the 3 decimal places
        weight_df = _get_zonal_faces_weight_at_constLat(np.array([
            face_0_edge_nodes, face_1_edge_nodes, face_2_edge_nodes,
            face_3_edge_nodes
        ]), np.sin(0.1 * np.pi), latlon_bounds, is_directed=False)

        nt.assert_array_almost_equal(weight_df, expected_weight_df, decimal=3)

    def test_get_zonal_faces_weight_at_constLat_on_pole_one_face(self):
        #The face is touching the pole， so the weight should be 1.0 since there's only 1 face
        face_edges_cart = np.array([[
            [[-5.22644277e-02, -5.22644277e-02, -9.97264689e-01],
             [-5.23359562e-02, -6.40930613e-18, -9.98629535e-01]],

            [[-5.23359562e-02, -6.40930613e-18, -9.98629535e-01],
             [6.12323400e-17, 0.00000000e+00, -1.00000000e+00]],

            [[6.12323400e-17, 0.00000000e+00, -1.00000000e+00],
             [3.20465306e-18, -5.23359562e-02, -9.98629535e-01]],

            [[3.20465306e-18, -5.23359562e-02, -9.98629535e-01],
             [-5.22644277e-02, -5.22644277e-02, -9.97264689e-01]]
        ]])

        # Corrected face_bounds
        face_bounds = np.array([
            [-1.57079633, -1.4968158],
            [3.14159265, 0.]
        ])
        constLat_cart = -1

        weight_df = _get_zonal_faces_weight_at_constLat(face_edges_cart, constLat_cart, face_bounds, is_directed=False)
        # Define the expected DataFrame
        expected_weight_df = pd.DataFrame({"face_index": [0], "weight": [1.0]})

        # Assert that the resulting should have weight is 1.0
        pd.testing.assert_frame_equal(weight_df, expected_weight_df)


    def test_get_zonal_faces_weight_at_constLat_on_pole_faces(self):
        #there will be 4 faces touching the pole, so the weight should be 0.25 for each face
        face_edges_cart = np.array([
            [
                [[5.22644277e-02, -5.22644277e-02, 9.97264689e-01], [5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
                [[5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
                [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [3.20465306e-18, -5.23359562e-02, 9.98629535e-01]],
                [[3.20465306e-18, -5.23359562e-02, 9.98629535e-01], [5.22644277e-02, -5.22644277e-02, 9.97264689e-01]]
            ],
            [
                [[5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [5.22644277e-02, 5.22644277e-02, 9.97264689e-01]],
                [[5.22644277e-02, 5.22644277e-02, 9.97264689e-01], [3.20465306e-18, 5.23359562e-02, 9.98629535e-01]],
                [[3.20465306e-18, 5.23359562e-02, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
                [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [5.23359562e-02, 0.00000000e+00, 9.98629535e-01]]
            ],
            [
                [[3.20465306e-18, -5.23359562e-02, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
                [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [-5.23359562e-02, -6.40930613e-18, 9.98629535e-01]],
                [[-5.23359562e-02, -6.40930613e-18, 9.98629535e-01],
                 [-5.22644277e-02, -5.22644277e-02, 9.97264689e-01]],
                [[-5.22644277e-02, -5.22644277e-02, 9.97264689e-01], [3.20465306e-18, -5.23359562e-02, 9.98629535e-01]]
            ],
            [
                [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [3.20465306e-18, 5.23359562e-02, 9.98629535e-01]],
                [[3.20465306e-18, 5.23359562e-02, 9.98629535e-01], [-5.22644277e-02, 5.22644277e-02, 9.97264689e-01]],
                [[-5.22644277e-02, 5.22644277e-02, 9.97264689e-01], [-5.23359562e-02, -6.40930613e-18, 9.98629535e-01]],
                [[-5.23359562e-02, -6.40930613e-18, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]]
            ]
        ])

        face_bounds = np.array([
            [[1.4968158, 1.57079633], [4.71238898, 0.0]],
            [[1.4968158, 1.57079633], [0.0, 1.57079633]],
            [[1.4968158, 1.57079633], [3.14159265, 0.0]],
            [[1.4968158, 1.57079633], [0.0, 3.14159265]]
        ])

        constLat_cart = 1.0

        weight_df = _get_zonal_faces_weight_at_constLat(face_edges_cart, constLat_cart, face_bounds, is_directed=False)
        # Define the expected DataFrame
        expected_weight_df = pd.DataFrame({
            'face_index': [0, 1, 2, 3],
            'weight': [0.25, 0.25, 0.25, 0.25]
        })

        # Assert that the DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(weight_df, expected_weight_df)


    def test_get_zonal_face_interval_pole(self):
        #The face is touching the pole
        face_edges_cart = np.array([
            [[-5.22644277e-02, -5.22644277e-02, -9.97264689e-01],
             [-5.23359562e-02, -6.40930613e-18, -9.98629535e-01]],

            [[-5.23359562e-02, -6.40930613e-18, -9.98629535e-01],
             [6.12323400e-17, 0.00000000e+00, -1.00000000e+00]],

            [[6.12323400e-17, 0.00000000e+00, -1.00000000e+00],
             [3.20465306e-18, -5.23359562e-02, -9.98629535e-01]],

            [[3.20465306e-18, -5.23359562e-02, -9.98629535e-01],
             [-5.22644277e-02, -5.22644277e-02, -9.97264689e-01]]
        ])

        # Corrected face_bounds
        face_bounds = np.array([
            [-1.57079633, -1.4968158],
            [3.14159265, 0.]
        ])
        constLat_cart = -0.9986295347545738

        weight_df = _get_zonal_face_interval(face_edges_cart, constLat_cart, face_bounds, is_directed=False)
        # No Nan values should be present in the weight_df
        self.assertFalse(weight_df.isnull().values.any())


    def test_get_zonal_faces_weight_at_constLat_latlonface(self):
        face_0 = [[np.deg2rad(350), np.deg2rad(40)], [np.deg2rad(350), np.deg2rad(20)],
                  [np.deg2rad(10), np.deg2rad(20)], [np.deg2rad(10), np.deg2rad(40)]]
        face_1 = [[np.deg2rad(5), np.deg2rad(20)], [np.deg2rad(5), np.deg2rad(10)],
                  [np.deg2rad(25), np.deg2rad(10)], [np.deg2rad(25), np.deg2rad(20)]]
        face_2 = [[np.deg2rad(30), np.deg2rad(40)], [np.deg2rad(30), np.deg2rad(20)],
                  [np.deg2rad(40), np.deg2rad(20)], [np.deg2rad(40), np.deg2rad(40)]]

        # Convert the face vertices to xyz coordinates
        face_0 = [_lonlat_rad_to_xyz(*v) for v in face_0]
        face_1 = [_lonlat_rad_to_xyz(*v) for v in face_1]
        face_2 = [_lonlat_rad_to_xyz(*v) for v in face_2]


        face_0_edge_nodes = np.array([[face_0[0], face_0[1]],
                                      [face_0[1], face_0[2]],
                                      [face_0[2], face_0[3]],
                                      [face_0[3], face_0[0]]])
        face_1_edge_nodes = np.array([[face_1[0], face_1[1]],
                                      [face_1[1], face_1[2]],
                                      [face_1[2], face_1[3]],
                                      [face_1[3], face_1[0]]])
        face_2_edge_nodes = np.array([[face_2[0], face_2[1]],
                                      [face_2[1], face_2[2]],
                                      [face_2[2], face_2[3]],
                                      [face_2[3], face_2[0]]])

        face_0_latlon_bound = np.array([[np.deg2rad(20), np.deg2rad(40)],
                                        [np.deg2rad(350), np.deg2rad(10)]])
        face_1_latlon_bound = np.array([[np.deg2rad(10), np.deg2rad(20)],
                                        [np.deg2rad(5), np.deg2rad(25)]])
        face_2_latlon_bound = np.array([[np.deg2rad(20), np.deg2rad(40)],
                                        [np.deg2rad(30), np.deg2rad(40)]])


        latlon_bounds = np.array([
            face_0_latlon_bound, face_1_latlon_bound, face_2_latlon_bound
        ])

        sum = 17.5 + 17.5 + 10
        expected_weight_df = pd.DataFrame({
            'face_index': [0, 1, 2],
            'weight': [17.5 / sum, 17.5/sum, 10/sum]
        })

        # Assert the results is the same to the 3 decimal places
        weight_df = _get_zonal_faces_weight_at_constLat(np.array([
            face_0_edge_nodes, face_1_edge_nodes, face_2_edge_nodes
        ]), np.sin(np.deg2rad(20)), latlon_bounds, is_directed=False, is_latlonface=True)


        nt.assert_array_almost_equal(weight_df, expected_weight_df, decimal=3)



        # A error will be raise if we don't set is_latlonface=True since the face_2 will be concave if
        # It's edges are all GCA
        with self.assertRaises(ValueError):
            _get_zonal_faces_weight_at_constLat(np.array([
                face_0_edge_nodes, face_1_edge_nodes, face_2_edge_nodes
            ]), np.deg2rad(20), latlon_bounds, is_directed=False)
