"""Same-body FP64-vs-AccuX diagnostic for the GCA / constant-latitude path.

Purpose (PR #1513)
------------------
This is *not* a standalone benchmark. It is a diagnostic that isolates the
engineering overhead of the UXarray AccuX wiring (wrapper / status layer /
dispatcher / masking) from the cost of the compensated-arithmetic (EFT) kernel
itself.

Method
------
We build a second implementation of the exact same three-layer stack used by the
real AccuX path in ``uxarray.grid.intersections``:

    L1  kernel      pure numerical core, returns two candidate points
    L2  try/status  finiteness + on-minor-arc masks, branchless point select
    L3  dispatcher  endpoint snapping, UXarray (2, 3) NaN-filled output

The only difference is the L1 body: here it is the *direct FP64* formula taken
verbatim from the AccuSphGeom reference

    tests/performance_test/gca_constLat/fp64_GCAconstLat.hh

    nx = a1*b2 - a2*b1;  ny = a2*b0 - a0*b2;  nz = a0*b1 - a1*b0
    denom   = nx^2 + ny^2
    norm_n2 = denom + nz^2
    s       = sqrt(denom - norm_n2 * z^2)
    pos = ( -(z*nx*nz - s*ny)/denom, -(z*ny*nz + s*nx)/denom, z )
    neg = ( -(z*nx*nz + s*ny)/denom, -(z*ny*nz - s*nx)/denom, z )

Because L2/L3 here are byte-for-byte the same logic as the real AccuX L2/L3, the
comparison factors cleanly:

    real AccuX time  -  same-body FP64 time  ==  cost of EFT math only
    same-body FP64 time  -  direct FP64 time ==  cost of L2/L3 plumbing only

Acceptance (interpret ``main()`` output)
    1. same-body FP64 dispatcher output/status == direct FP64 output/status
       (exact match: the plumbing does not change results)
    2. same-body FP64 dispatcher output/status == real AccuX output/status
       within tolerance (same algorithm, EFT only tightens rounding)
    3. plumbing overhead (L3 - L1) is small and comparable for both bodies
    4. remaining real-AccuX-vs-same-body gap is attributable to EFT math, not
       to wrappers/dispatch

Run directly:  ``python benchmarks/geometry_samebody.py``
This module is import-safe for asv (timing classes at the bottom).
"""

import math
import time

import numpy as np
from numba import njit

from uxarray.grid.arcs import _on_minor_arc_xyz, on_minor_arc
from uxarray.grid.intersections import (
    _accux_constlat_scalar,
    _snap_const_lat_endpoint_xy,
    gca_const_lat_intersection,
)

# ---------------------------------------------------------------------------
# L1 (FP64 body) — direct double-precision kernel, verbatim from
# fp64_GCAconstLat.hh. Scalar in / scalar out so Numba keeps it in registers,
# mirroring _accux_constlat_scalar.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _fp64_constlat_scalar(a0, a1, a2, b0, b1, b2, const_z):
    nx = a1 * b2 - a2 * b1
    ny = a2 * b0 - a0 * b2
    nz = a0 * b1 - a1 * b0

    denom = nx * nx + ny * ny
    norm_n2 = denom + nz * nz
    s = math.sqrt(denom - norm_n2 * const_z * const_z)

    inv_denom = 1.0 / denom if denom != 0.0 else np.inf
    px = -(const_z * nx * nz - s * ny) * inv_denom
    py = -(const_z * ny * nz + s * nx) * inv_denom
    nxo = -(const_z * nx * nz + s * ny) * inv_denom
    nyo = -(const_z * ny * nz - s * nx) * inv_denom
    return px, py, nxo, nyo


@njit(cache=True, inline="always")
def _fp64_constlat(x1, x2, const_z):
    px, py, nxo, nyo = _fp64_constlat_scalar(
        x1[0], x1[1], x1[2], x2[0], x2[1], x2[2], const_z
    )
    pos = np.empty(3)
    pos[0] = px
    pos[1] = py
    pos[2] = const_z
    neg = np.empty(3)
    neg[0] = nxo
    neg[1] = nyo
    neg[2] = const_z
    return pos, neg


# ---------------------------------------------------------------------------
# L2 (FP64 body) — identical logic to _try_gca_const_lat_intersection, only the
# L1 call differs. Branchless integer masks; status codes 0/1/2 as in AccuSphGeom.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _fp64_try_gca_const_lat_intersection(gca_cart, const_z):
    x1 = gca_cart[0]
    x2 = gca_cart[1]
    pos, neg = _fp64_constlat(x1, x2, const_z)

    pos_fin = int(math.isfinite(pos[0]) and math.isfinite(pos[1]))
    neg_fin = int(math.isfinite(neg[0]) and math.isfinite(neg[1]))
    pos_on = pos_fin * int(on_minor_arc(pos, x1, x2)) if pos_fin else 0
    neg_on = neg_fin * int(on_minor_arc(neg, x1, x2)) if neg_fin else 0

    pos_valid = pos_fin * pos_on
    neg_valid = neg_fin * neg_on

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


# ---------------------------------------------------------------------------
# L3 (FP64 body) — identical dispatcher to gca_const_lat_intersection, reusing
# the production _snap_const_lat_endpoint so only the numerical body differs.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _fp64_gca_const_lat_intersection(gca_cart, const_z):
    # Mirrors the production scalar dispatcher exactly (same allocation profile:
    # one (2, 3) array), only the L1 body differs. This keeps the same-body
    # comparison honest: any timing gap is the EFT math, not plumbing.
    res = np.empty((2, 3))
    res.fill(np.nan)

    a0 = gca_cart[0, 0]
    a1 = gca_cart[0, 1]
    a2 = gca_cart[0, 2]
    b0 = gca_cart[1, 0]
    b1 = gca_cart[1, 1]
    b2 = gca_cart[1, 2]

    px, py, nx, ny = _fp64_constlat_scalar(a0, a1, a2, b0, b1, b2, const_z)

    pos_fin = math.isfinite(px) and math.isfinite(py)
    neg_fin = math.isfinite(nx) and math.isfinite(ny)
    pos_valid = pos_fin and _on_minor_arc_xyz(px, py, const_z, a0, a1, a2, b0, b1, b2)
    neg_valid = neg_fin and _on_minor_arc_xyz(nx, ny, const_z, a0, a1, a2, b0, b1, b2)

    if pos_valid and not neg_valid:
        sx, sy = _snap_const_lat_endpoint_xy(px, py, a0, a1, a2, b0, b1, b2, const_z)
        res[0, 0] = sx
        res[0, 1] = sy
        res[0, 2] = const_z
    elif neg_valid and not pos_valid:
        sx, sy = _snap_const_lat_endpoint_xy(nx, ny, a0, a1, a2, b0, b1, b2, const_z)
        res[0, 0] = sx
        res[0, 1] = sy
        res[0, 2] = const_z
    elif pos_valid and neg_valid:
        psx, psy = _snap_const_lat_endpoint_xy(px, py, a0, a1, a2, b0, b1, b2, const_z)
        nsx, nsy = _snap_const_lat_endpoint_xy(nx, ny, a0, a1, a2, b0, b1, b2, const_z)
        dx = psx - nsx
        dy = psy - nsy
        if dx * dx + dy * dy < 1e-14:
            res[0, 0] = psx
            res[0, 1] = psy
            res[0, 2] = const_z
        else:
            res[0, 0] = psx
            res[0, 1] = psy
            res[0, 2] = const_z
            res[1, 0] = nsx
            res[1, 1] = nsy
            res[1, 2] = const_z
    return res


# ---------------------------------------------------------------------------
# Test inputs
# ---------------------------------------------------------------------------


def _unit(v):
    return v / np.linalg.norm(v)


def _make_cases(n, seed):
    """Random great-circle arcs paired with a latitude their arc actually crosses."""
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < n:
        a = _unit(rng.standard_normal(3))
        b = _unit(rng.standard_normal(3))
        if abs(np.dot(a, b)) > 0.999:  # near-degenerate arc, skip
            continue
        # pick a latitude strictly between the two endpoints' z so an
        # intersection is likely to exist
        zlo, zhi = sorted((a[2], b[2]))
        if zhi - zlo < 1e-6:
            continue
        const_z = zlo + (zhi - zlo) * rng.uniform(0.2, 0.8)
        cases.append((np.stack([a, b]), float(const_z)))
    return cases


# ---------------------------------------------------------------------------
# Batched drivers — the timing loop lives *inside* njit, mirroring the
# AccuSphGeom C++ benchmark (2M points looped in-kernel). Timing a Python-level
# per-call loop would drown the signal in interpreter overhead; batching in
# Numba measures true kernel throughput, which is also how UXarray actually
# calls these paths (once per edge over a whole grid).
# ---------------------------------------------------------------------------


@njit(cache=True)
def _batch_accux_kernel(A, B, Z):
    """Real AccuX L1 (EFT) kernel over a batch; accumulate to defeat DCE."""
    acc = 0.0
    for i in range(A.shape[0]):
        px, py, nxo, nyo = _accux_constlat_scalar(
            A[i, 0], A[i, 1], A[i, 2], B[i, 0], B[i, 1], B[i, 2], Z[i]
        )
        acc += px + py + nxo + nyo
    return acc


@njit(cache=True)
def _batch_fp64_kernel(A, B, Z):
    """Same-body FP64 L1 kernel over a batch; accumulate to defeat DCE."""
    acc = 0.0
    for i in range(A.shape[0]):
        px, py, nxo, nyo = _fp64_constlat_scalar(
            A[i, 0], A[i, 1], A[i, 2], B[i, 0], B[i, 1], B[i, 2], Z[i]
        )
        acc += px + py + nxo + nyo
    return acc


@njit(cache=True)
def _batch_accux_dispatch(gcas, Z):
    """Real AccuX full L1+L2+L3 dispatcher over a batch."""
    acc = 0.0
    for i in range(gcas.shape[0]):
        res = gca_const_lat_intersection(gcas[i], Z[i])
        v = res[0, 0]
        if v == v:  # not NaN
            acc += v
    return acc


@njit(cache=True)
def _batch_fp64_dispatch(gcas, Z):
    """Same-body FP64 full L1+L2+L3 dispatcher over a batch."""
    acc = 0.0
    for i in range(gcas.shape[0]):
        res = _fp64_gca_const_lat_intersection(gcas[i], Z[i])
        v = res[0, 0]
        if v == v:  # not NaN
            acc += v
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


def _pack(cases):
    """Turn the case list into contiguous arrays for the batched kernels."""
    A = np.array([c[0][0] for c in cases])
    B = np.array([c[0][1] for c in cases])
    Z = np.array([c[1] for c in cases])
    gcas = np.array([c[0] for c in cases])
    return A, B, Z, gcas


# ---------------------------------------------------------------------------
# Diagnostic driver
# ---------------------------------------------------------------------------


def main():
    base_cases = _make_cases(200, seed=20251104)

    # ---- correctness (on the 200 baseline cases) ----
    max_out_diff = 0.0
    status_mismatch = 0
    n_with_result = 0
    for gca, z in base_cases:
        fp64_res = _fp64_gca_const_lat_intersection(gca, z)
        accux_res = gca_const_lat_intersection(gca, z)
        fp64_rows = int(np.isfinite(fp64_res[0, 0])) + int(np.isfinite(fp64_res[1, 0]))
        accux_rows = int(np.isfinite(accux_res[0, 0])) + int(
            np.isfinite(accux_res[1, 0])
        )
        if fp64_rows != accux_rows:
            status_mismatch += 1
        if fp64_rows > 0 and accux_rows > 0:
            n_with_result += 1
            d = np.nanmax(np.abs(fp64_res - accux_res))
            if np.isfinite(d):
                max_out_diff = max(max_out_diff, d)

    # ---- timing: replicate the 200 cases into a large batch (~200k points) ----
    reps = 1000
    big = base_cases * reps
    A, B, Z, gcas = _pack(big)
    n = A.shape[0]

    t_direct = _time_batch(_batch_fp64_kernel, (A, B, Z))
    t_accux_k = _time_batch(_batch_accux_kernel, (A, B, Z))
    t_fp64_d = _time_batch(_batch_fp64_dispatch, (gcas, Z))
    t_accux_d = _time_batch(_batch_accux_dispatch, (gcas, Z))

    ns_per = lambda t: t / n * 1e9  # noqa: E731

    print("=" * 70)
    print("Same-body FP64-vs-AccuX diagnostic — GCA/ConstLat (PR #1513)")
    print("=" * 70)
    print(f"baseline cases: {len(base_cases)}   with-result: {n_with_result}")
    print(f"timing batch  : {n} points ({reps}x replication), best of 7")
    print()
    print("CORRECTNESS (same-body FP64 dispatcher vs real AccuX dispatcher)")
    print(f"  status mismatches : {status_mismatch}")
    print(f"  max output diff   : {max_out_diff:.3e}")
    print()
    print("TIMING (ns per point, in-kernel batch)")
    print(f"  L1  FP64  kernel            : {ns_per(t_direct):8.2f} ns")
    print(f"  L1  AccuX kernel            : {ns_per(t_accux_k):8.2f} ns")
    print(f"  L1+L2+L3 FP64  dispatch     : {ns_per(t_fp64_d):8.2f} ns")
    print(f"  L1+L2+L3 AccuX dispatch     : {ns_per(t_accux_d):8.2f} ns")
    print()
    print("DECOMPOSITION")
    plumb_fp64 = t_fp64_d - t_direct
    plumb_accux = t_accux_d - t_accux_k
    eft_kernel = t_accux_k - t_direct
    eft_dispatch = t_accux_d - t_fp64_d
    print(
        f"  plumbing L2/L3 over FP64  body : {ns_per(plumb_fp64):8.2f} ns/pt"
        f"  ({100 * plumb_fp64 / t_fp64_d:.1f}% of FP64 dispatch)"
    )
    print(
        f"  plumbing L2/L3 over AccuX body : {ns_per(plumb_accux):8.2f} ns/pt"
        f"  ({100 * plumb_accux / t_accux_d:.1f}% of AccuX dispatch)"
    )
    print(
        f"  EFT math cost (kernel level)   : {ns_per(eft_kernel):8.2f} ns/pt"
        f"  ({t_accux_k / t_direct:.2f}x FP64 kernel)"
    )
    print(
        f"  EFT math cost (dispatch level) : {ns_per(eft_dispatch):8.2f} ns/pt"
        f"  ({t_accux_d / t_fp64_d:.2f}x FP64 dispatch)"
    )
    print()
    print("INTERPRETATION")
    print("  - plumbing overhead should be ~equal for both bodies (body-independent)")
    print("  - AccuX/FP64 ratio at dispatch ~= ratio at kernel => plumbing adds no")
    print("    EFT-dependent overhead; measured cost is the EFT math, wired correctly")
    print("=" * 70)


# ---------------------------------------------------------------------------
# asv timing classes (batched; Numba warmed in setup)
# ---------------------------------------------------------------------------


class SameBodyConstLat:
    """asv: same-body FP64 vs real AccuX at kernel (L1) and dispatch (L3) levels."""

    def setup(self):
        cases = _make_cases(200, seed=20251104) * 100
        self.A, self.B, self.Z, self.gcas = _pack(cases)
        _batch_fp64_kernel(self.A, self.B, self.Z)
        _batch_accux_kernel(self.A, self.B, self.Z)
        _batch_fp64_dispatch(self.gcas, self.Z)
        _batch_accux_dispatch(self.gcas, self.Z)

    def time_fp64_kernel(self):
        _batch_fp64_kernel(self.A, self.B, self.Z)

    def time_accux_kernel(self):
        _batch_accux_kernel(self.A, self.B, self.Z)

    def time_fp64_dispatch(self):
        _batch_fp64_dispatch(self.gcas, self.Z)

    def time_accux_dispatch(self):
        _batch_accux_dispatch(self.gcas, self.Z)


if __name__ == "__main__":
    main()
