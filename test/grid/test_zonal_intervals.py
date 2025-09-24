import numpy as np
import numpy.testing as nt
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.integrate import _get_zonal_face_interval, _process_overlapped_intervals


def test_get_zonal_face_interval():
    """Test the _get_zonal_face_interval function for correct interval computation."""
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
    interval_df = _get_zonal_face_interval(face_edge_nodes, constZ,
                                           np.array([[-0.25 * np.pi, 0.25 * np.pi], [1.6 * np.pi,
                                                                                     0.4 * np.pi]]))
    expected_interval_df = pd.DataFrame({
        'start': [1.6 * np.pi, 0.0],
        'end': [2.0 * np.pi, 0.4 * np.pi]
    })
    expected_interval_df_sorted = expected_interval_df.sort_values(by='start').reset_index(drop=True)

    actual_values_sorted = interval_df[['start', 'end']].to_numpy()
    expected_values_sorted = expected_interval_df_sorted[['start', 'end']].to_numpy()

    nt.assert_array_almost_equal(actual_values_sorted, expected_values_sorted, decimal=13)


def test_get_zonal_face_interval_empty_interval():
    """Test the _get_zonal_face_interval function for cases where the interval is empty."""
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

    res = _get_zonal_face_interval(face_edges_cart, latitude_cart, face_latlon_bounds)
    expected_res = pl.DataFrame({"start": [0.0], "end": [0.0]})
    assert_frame_equal(res, expected_res)


def test_get_zonal_face_interval_encompass_pole():
    """Test the _get_zonal_face_interval function for cases where the face encompasses the pole inside."""
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
        [np.arcsin(0.9992005658145248), 0.5 * np.pi],
        [0, 2 * np.pi]
    ])
    expected_df = pl.DataFrame({
        'start': [0.000000, 1.101091, 2.357728, 3.614365, 4.871002, 6.127640],
        'end': [0.331721, 1.588358, 2.844995, 4.101632, 5.358270, 6.283185]
    })

    res = _get_zonal_face_interval(face_edges_cart, latitude_cart, face_latlon_bounds)

    assert_frame_equal(res, expected_df)


def test_get_zonal_face_interval_FILL_VALUE():
    """Test the _get_zonal_face_interval function for cases where there are dummy nodes."""
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
                                [dummy_node, dummy_node]])

    constZ = np.sin(0.20)
    interval_df = _get_zonal_face_interval(face_edge_nodes, constZ,
                                           np.array([[-0.25 * np.pi, 0.25 * np.pi], [1.6 * np.pi,
                                                                                     0.4 * np.pi]]))
    expected_interval_df = pd.DataFrame({
        'start': [1.6 * np.pi, 0.0],
        'end': [2.0 * np.pi, 0.4 * np.pi]
    })
    expected_interval_df_sorted = expected_interval_df.sort_values(by='start').reset_index(drop=True)

    actual_values_sorted = interval_df[['start', 'end']].to_numpy()
    expected_values_sorted = expected_interval_df_sorted[['start', 'end']].to_numpy()

    nt.assert_array_almost_equal(actual_values_sorted, expected_values_sorted, decimal=13)


def test_get_zonal_face_interval_GCA_constLat():
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
                                           is_GCA_list=np.array([True, False, True, False]))
    expected_interval_df = pd.DataFrame({
        'start': [1.6 * np.pi, 0.0],
        'end': [2.0 * np.pi, 0.4 * np.pi]
    })
    expected_interval_df_sorted = expected_interval_df.sort_values(by='start').reset_index(drop=True)

    actual_values_sorted = interval_df[['start', 'end']].to_numpy()
    expected_values_sorted = expected_interval_df_sorted[['start', 'end']].to_numpy()

    nt.assert_array_almost_equal(actual_values_sorted, expected_values_sorted, decimal=13)


def test_get_zonal_face_interval_equator():
    """Test that the face interval is correctly computed when the latitude
            is at the equator."""
    vertices_lonlat = [[-0.4 * np.pi, 0.25 * np.pi], [-0.4 * np.pi, 0.0],
                       [0.4 * np.pi, 0.0], [0.4 * np.pi, 0.25 * np.pi]]
    vertices = [_lonlat_rad_to_xyz(*v) for v in vertices_lonlat]

    face_edge_nodes = np.array([[vertices[0], vertices[1]],
                                [vertices[1], vertices[2]],
                                [vertices[2], vertices[3]],
                                [vertices[3], vertices[0]]])

    constZ = 0.0
    interval_df = _get_zonal_face_interval(face_edge_nodes, constZ,
                                           np.array([[0.0, 0.25 * np.pi], [1.6 * np.pi,
                                                                           0.4 * np.pi]]))
    expected_interval_df = pd.DataFrame({
        'start': [1.6 * np.pi, 0.0],
        'end': [2.0 * np.pi, 0.4 * np.pi]
    })
    expected_interval_df_sorted = expected_interval_df.sort_values(by='start').reset_index(drop=True)

    actual_values_sorted = interval_df[['start', 'end']].to_numpy()
    expected_values_sorted = expected_interval_df_sorted[['start', 'end']].to_numpy()

    nt.assert_array_almost_equal(actual_values_sorted, expected_values_sorted, decimal=13)


def test_process_overlapped_intervals_overlap_and_gap():
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

    # Create Polars DataFrame with explicit types
    df = pl.DataFrame(
        {
            'start': pl.Series([x['start'] for x in intervals_data], dtype=pl.Float64),
            'end': pl.Series([x['end'] for x in intervals_data], dtype=pl.Float64),
            'face_index': pl.Series([x['face_index'] for x in intervals_data], dtype=pl.Int64)
        }
    )

    # Expected results
    expected_overlap_contributions = {
        0: 75.0,
        1: 70.0,
        2: 5.0,
        3: 100.0,
        4: 90.0
    }

    # Process intervals
    overlap_contributions, total_length = _process_overlapped_intervals(df)

    # Assertions
    assert abs(total_length - 340.0) < 1e-10  # Using small epsilon for float comparison

    # Check each contribution matches expected value
    for face_idx, expected_value in expected_overlap_contributions.items():
        assert abs(overlap_contributions[face_idx] - expected_value) < 1e-10, \
            f"Mismatch for face_index {face_idx}: expected {expected_value}, got {overlap_contributions[face_idx]}"

    # Check that we have all expected face indices
    assert set(overlap_contributions.keys()) == set(expected_overlap_contributions.keys()), \
        "Mismatch in face indices"

    # Check total contributions sum matches total length
    assert abs(sum(overlap_contributions.values()) - total_length) < 1e-10, \
        "Sum of contributions doesn't match total length"


def test_process_overlapped_intervals_antimeridian():
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

    # Create Polars DataFrame with explicit types
    df = pl.DataFrame(
        {
            'start': pl.Series([x['start'] for x in intervals_data], dtype=pl.Float64),
            'end': pl.Series([x['end'] for x in intervals_data], dtype=pl.Float64),
            'face_index': pl.Series([x['face_index'] for x in intervals_data], dtype=pl.Int64)
        }
    )

    # Expected results for antimeridian case
    expected_overlap_contributions = {
        0: 105.0,
        1: 25.0,
        2: 175.0,
        3: 45.0
    }

    # Process intervals
    overlap_contributions, total_length = _process_overlapped_intervals(df)

    # Assert total length
    assert abs(total_length - 350.0) < 1e-10, \
        f"Expected total length 350.0, got {total_length}"

    # Check each contribution matches expected value
    for face_idx, expected_value in expected_overlap_contributions.items():
        assert abs(overlap_contributions[face_idx] - expected_value) < 1e-10, \
            f"Mismatch for face_index {face_idx}: expected {expected_value}, got {overlap_contributions[face_idx]}"

    # Verify all expected face indices are present
    assert set(overlap_contributions.keys()) == set(expected_overlap_contributions.keys()), \
        "Mismatch in face indices"


def test_get_zonal_face_interval_pole():
    # The face is touching the pole
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

    weight_df = _get_zonal_face_interval(face_edges_cart, constLat_cart, face_bounds)
    df_null_counts = weight_df.null_count()

    total_nulls = df_null_counts.to_numpy().sum()

    assert total_nulls == 0, f"Found {total_nulls} null values in the DataFrame"
