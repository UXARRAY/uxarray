"""Baseline regression tests ported from the AccuSphGeom C++ test suite.

Test data and expected results are taken directly from:
    https://github.com/hongyuchen1030/AccuSphGeom

Specific C++ tests mirrored here:
    tests/test_gca_gca_intersection_baseline.cpp     — 31 near-tangent GCA pairs
    tests/test_gca_constlat_intersection_baseline.cpp — 200 arc/latitude cases
    tests/test_pip_robust.cpp                         — simple spherical triangle
    tests/test_pip_complicated.cpp                    — 12-vertex concave polygon

The C++ library uses ultra-tight tolerances (3–100 ULP) backed by Shewchuk
adaptive precision and a geogram fallback.  This Python port implements only
the EFT tier, so the tolerances here reflect what double-precision EFT can
achieve:

    GCA-GCA intersection:     3e-8  (C++ reference: 1e-8)
    GCA-const-lat intersection: 1e-13 (C++ reference: 3–100 ULP ≈ 7e-16–2e-14)
    Point-in-polygon:         exact location codes (same as C++)
"""

import math
import os

import numpy as np
import pytest

from uxarray.grid.intersections import gca_const_lat_intersection, gca_gca_intersection
from uxarray.grid.point_in_face import (
    _LOC_INSIDE,
    _LOC_ON_EDGE,
    _LOC_ON_VERTEX,
    _LOC_OUTSIDE,
    _point_in_polygon_sphere,
)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "accusphgeom")
_GCA_GCA_CSV = os.path.join(
    _DATA_DIR, "gca_gca_pairs_seed20251104_N100_with_baseline.csv"
)
_GCA_CONSTLAT_CSV = os.path.join(
    _DATA_DIR, "gca_constlat_cases_with_baseline.csv"
)


def _sigexp(sig, exp):
    return math.ldexp(int(sig), int(exp))


def _parse_vec3(fields, start):
    return np.array(
        [
            _sigexp(fields[start], fields[start + 1]),
            _sigexp(fields[start + 2], fields[start + 3]),
            _sigexp(fields[start + 4], fields[start + 5]),
        ]
    )


def _parse_scalar(fields, start):
    return _sigexp(fields[start], fields[start + 1])


# ── GCA-GCA intersection ──────────────────────────────────────────────────────


def _load_gca_gca():
    rows = []
    with open(_GCA_GCA_CSV) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            assert len(fields) == 32
            pair_id = int(fields[0])
            a0 = _parse_vec3(fields, 2)
            a1 = _parse_vec3(fields, 8)
            b0 = _parse_vec3(fields, 14)
            b1 = _parse_vec3(fields, 20)
            baseline = _parse_vec3(fields, 26)
            rows.append((pair_id, a0, a1, b0, b1, baseline))
    return rows


@pytest.fixture(scope="module")
def gca_gca_rows():
    return _load_gca_gca()


def test_gca_gca_row_count(gca_gca_rows):
    assert len(gca_gca_rows) == 31


@pytest.mark.parametrize("idx", range(31))
def test_gca_gca_intersection_baseline(gca_gca_rows, idx):
    pair_id, a0, a1, b0, b1, baseline = gca_gca_rows[idx]
    result = gca_gca_intersection(np.stack([a0, a1]), np.stack([b0, b1]))
    assert result.shape[0] >= 1, f"pair_id={pair_id}: expected intersection, got none"
    err = float(np.linalg.norm(result[0] - baseline))
    assert err < 1e-15, f"pair_id={pair_id}: err={err:.3e} ≥ 1e-15"


# ── GCA-const-lat intersection ────────────────────────────────────────────────


def _load_gca_constlat():
    rows = []
    with open(_GCA_CONSTLAT_CSV) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            assert len(fields) == 19
            case_id = int(fields[0])
            a0 = _parse_vec3(fields, 1)
            a1 = _parse_vec3(fields, 7)
            z0 = _parse_scalar(fields, 13)
            bx = _parse_scalar(fields, 15)
            by = _parse_scalar(fields, 17)
            rows.append((case_id, a0, a1, z0, bx, by))
    return rows


@pytest.fixture(scope="module")
def gca_constlat_rows():
    return _load_gca_constlat()


def test_gca_constlat_row_count(gca_constlat_rows):
    assert len(gca_constlat_rows) == 200


@pytest.mark.parametrize("idx", range(200))
def test_gca_constlat_intersection_baseline(gca_constlat_rows, idx):
    case_id, a0, a1, z0, bx, by = gca_constlat_rows[idx]
    result = gca_const_lat_intersection(np.stack([a0, a1]), z0)
    assert not np.all(np.isnan(result[0])), f"case_id={case_id}: no intersection returned"
    dx = result[0, 0] - bx
    dy = result[0, 1] - by
    err = math.sqrt(dx * dx + dy * dy)
    assert err < 5e-15, f"case_id={case_id}: err_xy={err:.3e} ≥ 5e-15"


# ── Point-in-polygon: simple spherical triangle ───────────────────────────────
# From test_pip_robust.cpp: triangle A=(1,0,0) B=(0,1,0) C=(0,0,1)

_SIMPLE_POLY = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def test_pip_simple_on_vertex():
    q = np.array([1.0, 0.0, 0.0])
    assert _point_in_polygon_sphere(q, _SIMPLE_POLY) == _LOC_ON_VERTEX


def test_pip_simple_on_edge():
    # Normalize([1,1,0]) — midpoint of edge AB
    q = np.array([0.70710678118654752, 0.70710678118654752, 0.0])
    assert _point_in_polygon_sphere(q, _SIMPLE_POLY) == _LOC_ON_EDGE


def test_pip_simple_inside():
    q = np.array([1.0, 1.0, 1.0])
    q = q / np.linalg.norm(q)
    assert _point_in_polygon_sphere(q, _SIMPLE_POLY) == _LOC_INSIDE


# ── Point-in-polygon: complicated 12-vertex polygon ──────────────────────────
# From test_pip_complicated.cpp (Tier 4 / no-global-id overload)

_COMPLICATED_POLY = np.array(
    [
        [0.77114888623389370, -0.15726142646764130, 0.61692644537707060],
        [0.45249789144681710, -0.75061357063415830, 0.48148200985709080],
        [0.68946150885186746, -0.59933974587969335, 0.40673664307580021],
        [0.53398361424012150, -0.82144802877974800, 0.20021147753544170],
        [0.72547341102583852, -0.63064441484306173, 0.27563735581699919],
        [0.90662646752004000, -0.37288916572560260, 0.19743889808393390],
        [0.74736479846796566, -0.64967430761889954, 0.13917310096006544],
        [0.75468084319451650, -0.65603404827296060, -0.00872653549837396],
        [0.49138625363591330, -0.85368085756667700, -0.17253562867386300],
        [0.86555356123625300, -0.23932615843504300, -0.43993183849315200],
        [0.73819995144420940, -0.26096774566031860, -0.62205841157622660],
        [0.60166139617200880, -0.05234812405382043, -0.79703402578835670],
    ],
    dtype=np.float64,
)

_PIP_CASES = [
    ([0.75367527697268680, -0.65515992289232780, -0.05233595624294383], _LOC_INSIDE, "Q1 inside"),
    ([0.92054211727315200, -0.38498585550407840, 0.06624274592780397], _LOC_INSIDE, "Q2 inside"),
    ([0.53882393432914170, -0.82565565483991800, 0.16721694718218960], _LOC_OUTSIDE, "Q3 outside"),
    ([0.63494819288856630, -0.65761549896072850, 0.40544130015845230], _LOC_OUTSIDE, "Q4 outside"),
    # Q5 is exactly vertex P8 (0-indexed)
    ([0.49138625363591330, -0.85368085756667700, -0.17253562867386300], _LOC_ON_VERTEX, "Q5 on vertex"),
]


@pytest.mark.parametrize("q_xyz,expected,name", _PIP_CASES)
def test_pip_complicated(q_xyz, expected, name):
    q = np.array(q_xyz, dtype=np.float64)
    result = _point_in_polygon_sphere(q, _COMPLICATED_POLY)
    assert result == expected, f"{name}: expected {expected}, got {result}"
