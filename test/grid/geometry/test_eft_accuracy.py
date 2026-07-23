"""Accuracy regression tests for the compensated EFT primitives in
``uxarray.utils.computing`` — specifically the compensated sum-of-squares used by
the GCA / constant-latitude intersection kernel.

An earlier revision computed the naive ``Σ h² + Σ l²`` instead of the
correct compensated squared norm ``Σ (h + l)² = Σ h² + 2·Σ h·l`` (matching
AccuSphGeom's ``numeric::sum_of_squares_c``. The correct compensated result
reaches ~1e-30 relative accuracy on well-conditioned inputs; the naive form
only reaches ~1e-15.
"""

from fractions import Fraction

import numpy as np
import pytest

from uxarray.utils.computing import (
    _HAS_FMA,
    _sum_of_squares_c,
    _two_prod_veltkamp,
    accucross,
    two_prod,
)

# Correct compensated result: ~1e-30 rel err.  Naive Σh²+Σl²: ~1e-15.
_SUM_SQ_REL_TOL = 1e-24


def _unit(v):
    return v / np.linalg.norm(v)


def _make_normals(n, seed):
    """Compensated cross products (hi, lo) of well-separated arc pairs.

    ``|dot(a, b)| < 0.999`` keeps ``|n|`` away from the noise floor so the
    compensated squared norm is meaningful. (Extreme near-parallel arcs lose
    accuracy for *any* algorithm and are out of scope for this primitive test.)
    """
    rng = np.random.default_rng(seed)
    hi = np.empty((n, 3))
    lo = np.empty((n, 3))
    i = 0
    while i < n:
        a = _unit(rng.standard_normal(3))
        b = _unit(rng.standard_normal(3))
        if abs(np.dot(a, b)) > 0.999:
            continue
        xh, yh, zh, xl, yl, zl = accucross(a[0], a[1], a[2], b[0], b[1], b[2])
        hi[i] = (xh, yh, zh)
        lo[i] = (xl, yl, zl)
        i += 1
    return hi, lo


def _rel_err_vs_exact(res, hi_tup, lo_tup):
    """|(res_hi + res_lo) − Σ (h + l)²| / Σ (h + l)², computed exactly."""
    true = sum((Fraction(h) + Fraction(l)) ** 2 for h, l in zip(hi_tup, lo_tup))
    if true == 0:
        return 0.0
    return abs(float((Fraction(res[0]) + Fraction(res[1]) - true) / true))


@pytest.fixture(scope="module")
def normals():
    return _make_normals(2000, seed=20260716)


@pytest.mark.parametrize("ncomp", [2, 3])
def test_sum_of_squares_compensated_accuracy(normals, ncomp):
    """The generic keeps the 2·Σh·l cross term for each tuple length used by the
    const-lat kernel (N=2 -> denominator, N=3 -> |n|²). A naive Σh²+Σl² would
    regress to ~1e-15 and fail this bound."""
    hi, lo = normals
    worst = 0.0
    for i in range(hi.shape[0]):
        h = tuple(float(x) for x in hi[i, :ncomp])
        lo_t = tuple(float(x) for x in lo[i, :ncomp])
        worst = max(worst, _rel_err_vs_exact(_sum_of_squares_c(h, lo_t), h, lo_t))
    assert worst < _SUM_SQ_REL_TOL, (
        f"N={ncomp}: _sum_of_squares_c max rel err {worst:.2e} ≥ "
        f"{_SUM_SQ_REL_TOL:.0e} (naive Σh²+Σl² regressed the cross term?)"
    )


def test_sum_of_squares_keeps_cross_term(normals):
    """Directly assert the result tracks the compensated ``Σh² + 2·Σh·l`` and
    NOT the naive ``Σh² + Σl²``, on cases where the two formulas are exactly
    distinguishable."""
    hi, lo = normals
    checked = 0
    for i in range(hi.shape[0]):
        h = (float(hi[i, 0]), float(hi[i, 1]))
        lo_t = (float(lo[i, 0]), float(lo[i, 1]))
        big_h = [Fraction(x) for x in h]
        big_l = [Fraction(x) for x in lo_t]
        correct = sum(big_h[k] * big_h[k] for k in range(2)) + 2 * sum(
            big_h[k] * big_l[k] for k in range(2)
        )
        naive = sum(big_h[k] * big_h[k] + big_l[k] * big_l[k] for k in range(2))
        # The two formulas differ by ~2·Σh·l ≈ 1e-16 relative — that IS the bug
        # we test. Skip only cases where the gap falls below the compensated
        # accuracy floor (so the two are genuinely indistinguishable there).
        if correct == 0 or abs(float((naive - correct) / correct)) < 1e-20:
            continue
        res = _sum_of_squares_c(h, lo_t)
        val = Fraction(res[0]) + Fraction(res[1])
        assert abs(val - correct) < abs(val - naive)
        checked += 1
    assert checked > 100, "test did not exercise enough distinguishable cases"


def test_sum_of_squares_generic_any_n():
    """The single generic works for any tuple length — e.g. N=4, which no
    hand-written ``_sum_sq_cN`` exists for — with the same compensated accuracy.
    This is the point of one N-generic primitive instead of per-N helpers.
    """
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(2000):
        h = tuple(float(x) for x in rng.standard_normal(4))
        # residual-scale low parts (~1 ulp of each high part)
        lo_t = tuple(x * 1e-16 * float(rng.standard_normal()) for x in h)
        worst = max(worst, _rel_err_vs_exact(_sum_of_squares_c(h, lo_t), h, lo_t))
    assert worst < _SUM_SQ_REL_TOL, (
        f"N=4: _sum_of_squares_c max rel err {worst:.2e} ≥ {_SUM_SQ_REL_TOL:.0e}"
    )


def _random_operands(rng, n):
    for _ in range(n):
        yield (
            float(rng.standard_normal() * rng.integers(1, 1 << 20)),
            float(rng.standard_normal() * rng.integers(1, 1 << 20)),
        )


def test_two_prod_error_term_is_the_exact_residual():
    """``two_prod`` must return the EXACT rounding error of ``a * b``.

    This is the property ``computing._validate_fma`` exists to guarantee at
    import time, asserted here directly against exact rational arithmetic so it
    cannot silently lapse. A non-fused FMA lowering (or a non-compliant FMA)
    yields ``e = 0.0``, which fails this outright.

    Note this is deliberately checked as ``p + e == a*b`` **exactly, over the
    rationals** — NOT as the float expression ``p + e``, which rounds straight
    back to ``p`` (since ``|e| <= ulp(p)/2``) and would make the assertion
    vacuously true for any ``e`` whatsoever.
    """
    rng = np.random.default_rng(20260716)
    for a, b in _random_operands(rng, 20000):
        p, e = two_prod(a, b)
        assert Fraction(p) + Fraction(e) == Fraction(a) * Fraction(b), (
            f"two_prod({a!r}, {b!r}) = ({p!r}, {e!r}) is not an exact "
            f"error-free transform"
        )


@pytest.mark.skipif(not _HAS_FMA, reason="no FMA path on this toolchain")
def test_two_prod_fma_matches_veltkamp_bit_for_bit():
    """The FMA and Veltkamp paths must agree bit-for-bit, so ``_HAS_FMA`` can
    never change results. The exact residual is unique and representable, so
    two correct implementations have no freedom to differ."""
    rng = np.random.default_rng(20260717)
    for a, b in _random_operands(rng, 20000):
        p, e = two_prod(a, b)
        pv, ev = _two_prod_veltkamp(a, b)
        assert np.float64(p).view(np.int64) == np.float64(pv).view(np.int64)
        assert np.float64(e).view(np.int64) == np.float64(ev).view(np.int64)
