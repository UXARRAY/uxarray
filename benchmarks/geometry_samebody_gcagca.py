"""Same-body FP64-vs-AccuX diagnostic for the GCA x GCA path.

Companion to ``geometry_samebody.py``, which only covers the GCA/constant-latitude
stack.  T

gca_gca is the kernel invoked per edge from the njit point-in-polygon crossing
loop (``uxarray/grid/geometry.py`` ``_check_intersection``), so its allocation
profile is on the hot path of ``get_faces_containing_point`` and pole detection.

Factoring (identical to ``geometry_samebody.py``)::

    real AccuX time  -  same-body FP64 time  == EFT math only
    same-body FP64   -  direct FP64 kernel   == L2/L3 plumbing only

"""

import math
import time

import numpy as np
from numba import njit

from uxarray.grid.arcs import on_minor_arc
from uxarray.grid.intersections import _accux_gca, gca_gca_intersection


@njit(cache=True, inline="always")
def _fp64_gca(w0, w1, v0, v1):
    """
    L1 (FP64 body) -- plain double-precision cross-product triple, the direct
    analogue of _accux_gca (intersections.py) with accucross/accucross_pair
    replaced by naive FP64 cross products.  Same allocation shape (two np.empty(3))
    so the twin's allocation profile matches the real kernel exactly.
    """

    n1x = w0[1] * w1[2] - w0[2] * w1[1]
    n1y = w0[2] * w1[0] - w0[0] * w1[2]
    n1z = w0[0] * w1[1] - w0[1] * w1[0]
    n2x = v0[1] * v1[2] - v0[2] * v1[1]
    n2y = v0[2] * v1[0] - v0[0] * v1[2]
    n2z = v0[0] * v1[1] - v0[1] * v1[0]

    vx = n1y * n2z - n1z * n2y
    vy = n1z * n2x - n1x * n2z
    vz = n1x * n2y - n1y * n2x

    vn = math.sqrt(vx * vx + vy * vy + vz * vz)
    inv = 1.0 / vn if vn != 0.0 else np.inf
    pos = np.empty(3)
    pos[0] = vx * inv
    pos[1] = vy * inv
    pos[2] = vz * inv
    neg = np.empty(3)
    neg[0] = -pos[0]
    neg[1] = -pos[1]
    neg[2] = -pos[2]
    return pos, neg


@njit(cache=True)
def _fp64_try_gca_gca_intersection(w0, w1, v0, v1):
    """
    L2 (FP64 body) -- byte-for-byte identical logic to _try_gca_gca_intersection
    (intersections.py), only the L1 call differs.
    """
    pos, neg = _fp64_gca(w0, w1, v0, v1)

    pos_fin = (
        1
        if math.isfinite(pos[0]) and math.isfinite(pos[1]) and math.isfinite(pos[2])
        else 0
    )
    neg_fin = (
        1
        if math.isfinite(neg[0]) and math.isfinite(neg[1]) and math.isfinite(neg[2])
        else 0
    )
    pos_on_a = 1 if (pos_fin and on_minor_arc(pos, w0, w1)) else 0
    pos_on_b = 1 if (pos_fin and on_minor_arc(pos, v0, v1)) else 0
    neg_on_a = 1 if (neg_fin and on_minor_arc(neg, w0, w1)) else 0
    neg_on_b = 1 if (neg_fin and on_minor_arc(neg, v0, v1)) else 0

    pos_valid = pos_fin * pos_on_a * pos_on_b
    neg_valid = neg_fin * neg_on_a * neg_on_b

    pos_mask = pos_valid * (1 - neg_valid)
    neg_mask = neg_valid * (1 - pos_valid)

    point = np.empty(3)
    point[0] = pos_mask * pos[0] + neg_mask * neg[0]
    point[1] = pos_mask * pos[1] + neg_mask * neg[1]
    point[2] = pos_mask * pos[2] + neg_mask * neg[2]

    both = pos_valid * neg_valid
    none = (1 - pos_valid) * (1 - neg_valid)
    status = both + none * 2
    return point, status, pos, neg


@njit(cache=True)
def _fp64_gca_gca_intersection(gca_a_xyz, gca_b_xyz):
    """
    L3 (FP64 body) -- identical dispatcher to gca_gca_intersection
    (intersections.py), same np.empty((2, 3)) + res[:count] slice profile.
    """
    if gca_a_xyz.shape[1] != 3 or gca_b_xyz.shape[1] != 3:
        raise ValueError("The two GCAs must be in the cartesian [x, y, z] format")

    w0 = gca_a_xyz[0]
    w1 = gca_a_xyz[1]
    v0 = gca_b_xyz[0]
    v1 = gca_b_xyz[1]

    point, status, pos, neg = _fp64_try_gca_gca_intersection(w0, w1, v0, v1)

    res = np.empty((2, 3))
    count = 0
    if status == 0:
        res[0, 0] = point[0]
        res[0, 1] = point[1]
        res[0, 2] = point[2]
        count = 1
    elif status == 1:
        res[0, 0] = pos[0]
        res[0, 1] = pos[1]
        res[0, 2] = pos[2]
        res[1, 0] = neg[0]
        res[1, 1] = neg[1]
        res[1, 2] = neg[2]
        count = 2
    else:
        if on_minor_arc(v0, w0, w1):
            res[count, 0] = v0[0]
            res[count, 1] = v0[1]
            res[count, 2] = v0[2]
            count += 1
        if on_minor_arc(v1, w0, w1):
            res[count, 0] = v1[0]
            res[count, 1] = v1[1]
            res[count, 2] = v1[2]
            count += 1
    return res[:count]


# ---------------------------------------------------------------------------
# Case generation -- a LARGE set of DISTINCT, SHORT arcs (matching real grid
# edges, ~0.2-8 deg / median ~1.5 deg) with a controlled intersect / no-intersect
# mix.  Short arcs put a x b in the near-parallel cancellation regime the EFT
# targets; the mix exercises both the status 0/1 (allocating) and status 2
# (empty-return) dispatcher branches.
# ---------------------------------------------------------------------------


def _unit(v):
    return v / np.linalg.norm(v)


def _short_arc(rng, mid=None):
    """A short great-circle arc (~0.2-8 deg, median ~1.5 deg) around ``mid`` (a
    random point if not given), matching real unstructured-grid edge lengths."""
    if mid is None:
        mid = _unit(rng.standard_normal(3))
    tan = _unit(np.cross(mid, _unit(rng.standard_normal(3))))
    half = 0.5 * math.radians(10.0 ** rng.uniform(math.log10(0.2), math.log10(8.0)))
    ca, sa = math.cos(half), math.sin(half)
    return _unit(mid * ca - tan * sa), _unit(mid * ca + tan * sa)


def _make_gca_cases(n, seed, frac_intersect=0.6):
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < n:
        if rng.random() < frac_intersect:
            # two short arcs sharing a midpoint p -> p is the midpoint of both
            # minor arcs, so they intersect there (status 0/1).
            p = _unit(rng.standard_normal(3))
            a0, a1 = _short_arc(rng, mid=p)
            b0, b1 = _short_arc(rng, mid=p)
        else:
            # two independent short arcs -> their great circles cross off at
            # least one minor arc, exercising the status 2 / empty-return branch.
            a0, a1 = _short_arc(rng)
            b0, b1 = _short_arc(rng)
        cases.append((np.stack([a0, a1]), np.stack([b0, b1])))
    return cases


def _pack_gca(cases):
    wa = np.ascontiguousarray([c[0][0] for c in cases])
    wb = np.ascontiguousarray([c[0][1] for c in cases])
    va = np.ascontiguousarray([c[1][0] for c in cases])
    vb = np.ascontiguousarray([c[1][1] for c in cases])
    ga = np.ascontiguousarray([c[0] for c in cases])  # (n, 2, 3)
    gb = np.ascontiguousarray([c[1] for c in cases])
    return wa, wb, va, vb, ga, gb


# ---------------------------------------------------------------------------
# Batched in-kernel drivers -- the timing loop lives inside njit (mirrors
# geometry_samebody.py).  Accumulate a scalar to defeat dead-code elimination.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _batch_accux_gca_kernel(wa, wb, va, vb):
    acc = 0.0
    for i in range(wa.shape[0]):
        pos, neg = _accux_gca(wa[i], wb[i], va[i], vb[i])
        acc += pos[0] + pos[1] + pos[2]
    return acc


@njit(cache=True)
def _batch_fp64_gca_kernel(wa, wb, va, vb):
    acc = 0.0
    for i in range(wa.shape[0]):
        pos, neg = _fp64_gca(wa[i], wb[i], va[i], vb[i])
        acc += pos[0] + pos[1] + pos[2]
    return acc


@njit(cache=True)
def _batch_accux_gca_dispatch(ga, gb):
    acc = 0.0
    for i in range(ga.shape[0]):
        res = gca_gca_intersection(ga[i], gb[i])
        if res.shape[0] > 0:
            acc += res[0, 0]
    return acc


@njit(cache=True)
def _batch_fp64_gca_dispatch(ga, gb):
    acc = 0.0
    for i in range(ga.shape[0]):
        res = _fp64_gca_gca_intersection(ga[i], gb[i])
        if res.shape[0] > 0:
            acc += res[0, 0]
    return acc


def _time_batch(fn, args, repeat=7):
    """Best-of-`repeat` wall-time for one batched call (compile excluded)."""
    fn(*args)  # warm / compile
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main(n_cases=100_000, seed=20251104):
    """
    Standalone diagnostic driver (prints ns/edge-pair). Mirrors geometry_samebody.main.
    """
    cases = _make_gca_cases(n_cases, seed=seed)
    wa, wb, va, vb, ga, gb = _pack_gca(cases)
    n = wa.shape[0]

    # correctness: FP64 twin vs real AccuX dispatcher (row count + max diff)
    row_mismatch = 0
    max_out_diff = 0.0
    n_check = min(n, 5000)
    for i in range(n_check):
        r_fp = _fp64_gca_gca_intersection(ga[i], gb[i])
        r_ax = gca_gca_intersection(ga[i], gb[i])
        if r_fp.shape[0] != r_ax.shape[0]:
            row_mismatch += 1
        elif r_fp.shape[0] > 0:
            max_out_diff = max(max_out_diff, float(np.max(np.abs(r_fp - r_ax))))

    t_fp64_k = _time_batch(_batch_fp64_gca_kernel, (wa, wb, va, vb))
    t_accux_k = _time_batch(_batch_accux_gca_kernel, (wa, wb, va, vb))
    t_fp64_d = _time_batch(_batch_fp64_gca_dispatch, (ga, gb))
    t_accux_d = _time_batch(_batch_accux_gca_dispatch, (ga, gb))

    def ns(t):
        return t / n * 1e9

    print("=" * 70)
    print("Same-body FP64-vs-AccuX diagnostic -- GCA x GCA")
    print("=" * 70)
    print(f"distinct cases : {n}   (best of 7, in-kernel batch)")
    print(
        f"correctness    : row mismatches {row_mismatch}/{n_check}"
        f"   max |diff| {max_out_diff:.3e}"
    )
    print()
    print("TIMING (ns per edge-pair)")
    print(f"  L1  FP64  kernel        : {ns(t_fp64_k):8.2f} ns")
    print(f"  L1  AccuX kernel        : {ns(t_accux_k):8.2f} ns")
    print(f"  L1+L2+L3 FP64  dispatch : {ns(t_fp64_d):8.2f} ns")
    print(f"  L1+L2+L3 AccuX dispatch : {ns(t_accux_d):8.2f} ns")
    print()
    print("DECOMPOSITION")
    print(f"  plumbing (FP64  body)   : {ns(t_fp64_d - t_fp64_k):8.2f} ns/edge")
    print(f"  plumbing (AccuX body)   : {ns(t_accux_d - t_accux_k):8.2f} ns/edge")
    print(
        f"  EFT math (kernel)       : {ns(t_accux_k - t_fp64_k):8.2f} ns/edge"
        f"  ({t_accux_k / t_fp64_k:.2f}x)"
    )
    print(
        f"  EFT math (dispatch)     : {ns(t_accux_d - t_fp64_d):8.2f} ns/edge"
        f"  ({t_accux_d / t_fp64_d:.2f}x)"
    )
    print("=" * 70)


class SameBodyGcaGca:
    """
    asv timing class (Numba warmed in setup, distinct cases)
    same-body FP64 vs real AccuX gca_gca at kernel (L1) and dispatch (L3).
    """

    def setup(self):
        cases = _make_gca_cases(100_000, seed=20251104)
        self.wa, self.wb, self.va, self.vb, self.ga, self.gb = _pack_gca(cases)
        _batch_fp64_gca_kernel(self.wa, self.wb, self.va, self.vb)
        _batch_accux_gca_kernel(self.wa, self.wb, self.va, self.vb)
        _batch_fp64_gca_dispatch(self.ga, self.gb)
        _batch_accux_gca_dispatch(self.ga, self.gb)

    def time_fp64_kernel(self):
        _batch_fp64_gca_kernel(self.wa, self.wb, self.va, self.vb)

    def time_accux_kernel(self):
        _batch_accux_gca_kernel(self.wa, self.wb, self.va, self.vb)

    def time_fp64_dispatch(self):
        _batch_fp64_gca_dispatch(self.ga, self.gb)

    def time_accux_dispatch(self):
        _batch_accux_gca_dispatch(self.ga, self.gb)


if __name__ == "__main__":
    main()
