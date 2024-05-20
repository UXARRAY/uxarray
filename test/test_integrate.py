import uxarray as ux
import os
from unittest import TestCase
from pathlib import Path
import numpy as np
import pandas as pd

import numpy.testing as nt

import uxarray as ux
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.integrate import _get_zonal_face_interval, _process_overlapped_intervals, _get_zonal_faces_weight_at_constLat

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

    def test_get_zonal_face_interval(self):
        """Test that the zonal face weights are correct."""
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

    def test_get_zonal_face_interval_GCA_constLat(self):
        """Test that the zonal face weights are correct."""
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
        """Test that the zonal face weights are correct."""
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

    def test_process_overlapped_intervals(self):
        # Example data that has overlapping intervals and gap
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
        ]),
                                                        0.0,
                                                        latlon_bounds,
                                                        is_directed=False)

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
        ]),
                                                        np.sin(0.1 * np.pi),
                                                        latlon_bounds,
                                                        is_directed=False)

        nt.assert_array_almost_equal(weight_df, expected_weight_df, decimal=3)

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
        ]),
                                                        np.sin(np.deg2rad(20)),
                                                        latlon_bounds,
                                                        is_directed=False, is_latlonface=True)


        nt.assert_array_almost_equal(weight_df, expected_weight_df, decimal=3)



        # A error will be raise if we don't set is_latlonface=True since the face_2 will be concave if
        # It's edges are all GCA
        with self.assertRaises(ValueError):
            _get_zonal_faces_weight_at_constLat(np.array([
            face_0_edge_nodes, face_1_edge_nodes, face_2_edge_nodes
        ]),
                                                        np.deg2rad(20),
                                                        latlon_bounds,
                                                        is_directed=False)
