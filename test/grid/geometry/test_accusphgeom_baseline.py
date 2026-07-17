"""Baseline regression tests ported from the AccuSphGeom C++ test suite.

Test data and expected results are taken directly from:
    https://github.com/hongyuchen1030/AccuSphGeom

Specific C++ tests mirrored here:
    tests/test_gca_gca_intersection_baseline.cpp     — 31 near-tangent GCA pairs
    tests/test_gca_constlat_intersection_baseline.cpp — 200 arc/latitude cases

The C++ library uses ultra-tight tolerances (3–100 ULP) backed by Shewchuk
adaptive precision and a geogram fallback.  This Python port implements only
the EFT tier, so the tolerances here reflect what double-precision EFT can
achieve:

    GCA-GCA intersection:     3e-8  (C++ reference: 1e-8)
    GCA-const-lat intersection: 1e-13 (C++ reference: 3–100 ULP ≈ 7e-16–2e-14)
"""

import math
import os

import numpy as np
import pytest

from uxarray.grid.intersections import gca_const_lat_intersection, gca_gca_intersection

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
