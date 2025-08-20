import os
from pathlib import Path

import numpy as np
import numpy.testing as nt
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
import uxarray as ux
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.integrate import (
    _get_zonal_face_interval,
    _process_overlapped_intervals,
    _get_faces_constLat_intersection_info,
    _zonal_face_weights,
    _zonal_face_weights_robust
)
from uxarray.grid.utils import _get_cartesian_face_edge_nodes_array

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parent

gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"


def test_get_faces_constLat_intersection_info_one_intersection():
    """Test for a face that has one intersection with a constant latitude."""
    face_edges_cart = np.array([
        [[0.8660254037844387, 0.5, 0.0], [0.8660254037844387, -0.5, 0.0]],
        [[0.8660254037844387, -0.5, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.8660254037844387, 0.5, 0.0]]
    ])

    constZ = 0.5
    intersection_info = _get_faces_constLat_intersection_info(face_edges_cart, constZ)
    expected_intersection_info = np.array([
        [True, 0.0, 0.0],
        [False, 0.0, 0.0],
        [False, 0.0, 0.0],
        [True, 0.0, 0.0]
    ])

    nt.assert_allclose(intersection_info, expected_intersection_info, atol=ERROR_TOLERANCE)


def test_get_faces_constLat_intersection_info_encompass_pole():
    """Test for a face that encompasses the pole."""
    face_edges_cart = np.array([
        [[0.0, 0.0, 1.0], [0.8660254037844387, 0.5, 0.0]],
        [[0.8660254037844387, 0.5, 0.0], [0.8660254037844387, -0.5, 0.0]],
        [[0.8660254037844387, -0.5, 0.0], [0.0, 0.0, 1.0]]
    ])

    constZ = 0.5
    intersection_info = _get_faces_constLat_intersection_info(face_edges_cart, constZ)
    expected_intersection_info = np.array([
        [True, 0.0, 0.0],
        [False, 0.0, 0.0],
        [True, 0.0, 0.0]
    ])

    nt.assert_allclose(intersection_info, expected_intersection_info, atol=ERROR_TOLERANCE)


def test_get_zonal_face_interval():
    """Test for getting the zonal face interval."""
    face_edges_cart = np.array([
        [[0.8660254037844387, 0.5, 0.0], [0.8660254037844387, -0.5, 0.0]],
        [[0.8660254037844387, -0.5, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.8660254037844387, 0.5, 0.0]]
    ])

    constZ = 0.5
    interval = _get_zonal_face_interval(face_edges_cart, constZ)
    expected_interval = np.array([0.0, 0.0])

    nt.assert_allclose(interval, expected_interval, atol=ERROR_TOLERANCE)


def test_get_zonal_face_interval_empty_interval():
    """Test for getting an empty zonal face interval."""
    face_edges_cart = np.array([
        [[0.8660254037844387, 0.5, 0.0], [0.8660254037844387, -0.5, 0.0]],
        [[0.8660254037844387, -0.5, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.8660254037844387, 0.5, 0.0]]
    ])

    constZ = 0.9
    interval = _get_zonal_face_interval(face_edges_cart, constZ)
    expected_interval = np.array([INT_FILL_VALUE, INT_FILL_VALUE])

    nt.assert_allclose(interval, expected_interval, atol=ERROR_TOLERANCE)


def test_process_overlapped_intervals_overlap_and_gap():
    """Test processing overlapped intervals with overlaps and gaps."""
    intervals = np.array([
        [0.0, 1.0],
        [0.5, 1.5],
        [2.0, 3.0],
        [2.5, 3.5]
    ])

    processed_intervals = _process_overlapped_intervals(intervals)
    expected_intervals = np.array([
        [0.0, 1.5],
        [2.0, 3.5]
    ])

    nt.assert_allclose(processed_intervals, expected_intervals, atol=ERROR_TOLERANCE)


def test_process_overlapped_intervals_antimeridian():
    """Test processing overlapped intervals across the antimeridian."""
    intervals = np.array([
        [5.5, 6.5],
        [0.0, 1.0],
        [6.0, 0.5]
    ])

    processed_intervals = _process_overlapped_intervals(intervals, is_latlonface=False)
    expected_intervals = np.array([
        [5.5, 1.0]
    ])

    nt.assert_allclose(processed_intervals, expected_intervals, atol=ERROR_TOLERANCE)
