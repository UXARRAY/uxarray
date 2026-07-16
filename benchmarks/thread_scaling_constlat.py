"""End-to-end thread-scaling benchmark for the GCA / constant-latitude path.

Unlike the bare-kernel micro-benchmarks, this drives the *real* UXarray
dispatcher (`gca_const_lat_intersection`) over the edges of a real mesh, and a
same-body FP64 dispatcher built from the identical L1/L2/L3 structure, sweeping
the Numba thread count. It answers: as threads increase, does the compensated
(AccuX/EFT) path scale the same way as plain FP64 when hit through the full
UXarray path (dispatch + masks + snapping + output packaging), not just the
raw kernel?

Run:  python benchmarks/thread_scaling_constlat.py [grid] [n_lat] [repeats]
Prints CSV to stdout: threads,fp64_ns,accux_ns,accux_over_fp64
(ns are per edge-x-latitude evaluation.)
"""

import math
import sys
import time

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

import uxarray as ux
from uxarray.grid.intersections import gca_const_lat_intersection

# same-body FP64 dispatcher: identical structure, FP64 body instead of EFT.
from benchmarks.geometry_samebody import _fp64_gca_const_lat_intersection


@njit(cache=False, parallel=True)
def _batch_accux(edges, zs, acc):
    """Real AccuX dispatcher over every (edge, latitude) pair, in parallel."""
    n_edge = edges.shape[0]
    for i in prange(n_edge):
        s = 0.0
        for k in range(zs.shape[0]):
            res = gca_const_lat_intersection(edges[i], zs[k])
            v = res[0, 0]
            if v == v:  # not NaN
                s += v
        acc[i] = s


@njit(cache=False, parallel=True)
def _batch_fp64(edges, zs, acc):
    """Same-body FP64 dispatcher over every (edge, latitude) pair, in parallel."""
    n_edge = edges.shape[0]
    for i in prange(n_edge):
        s = 0.0
        for k in range(zs.shape[0]):
            res = _fp64_gca_const_lat_intersection(edges[i], zs[k])
            v = res[0, 0]
            if v == v:
                s += v
        acc[i] = s


def _build_edges(grid):
    """Extract each edge as a (2, 3) Cartesian arc from a real grid."""
    en = grid.edge_node_connectivity.values
    x = grid.node_x.values
    y = grid.node_y.values
    z = grid.node_z.values
    n_edge = en.shape[0]
    edges = np.empty((n_edge, 2, 3), dtype=np.float64)
    edges[:, 0, 0] = x[en[:, 0]]
    edges[:, 0, 1] = y[en[:, 0]]
    edges[:, 0, 2] = z[en[:, 0]]
    edges[:, 1, 0] = x[en[:, 1]]
    edges[:, 1, 1] = y[en[:, 1]]
    edges[:, 1, 2] = z[en[:, 1]]
    return edges


def main():
    grid_name = sys.argv[1] if len(sys.argv) > 1 else "outCSne30"
    n_lat = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    grid = ux.tutorial.open_grid(grid_name)
    edges = _build_edges(grid)
    # latitudes spanning the sphere (as z = sin(lat)); avoid exact poles
    zs = np.linspace(-0.95, 0.95, n_lat)
    n_eval = edges.shape[0] * n_lat
    acc = np.empty(edges.shape[0], dtype=np.float64)

    # Cap the sweep at the number of fast performance cores. On heterogeneous
    # CPUs (e.g. Apple M-series: performance + efficiency cores) adding the slow
    # efficiency cores makes the parallel loop wait on the slowest chunk, so
    # times go *up* past the P-core count — a scheduling artifact, not a
    # property of the kernels. Override with the PERF_CORES env var if needed.
    import os

    perf_cores = int(os.environ.get("PERF_CORES", "8"))
    max_threads = min(get_num_threads(), perf_cores)
    counts = []
    t = 1
    while t < max_threads:
        counts.append(t)
        t *= 2
    counts.append(max_threads)
    counts = sorted(set(counts))

    # warm-up compile
    set_num_threads(max_threads)
    _batch_accux(edges[:8], zs, acc[:8])
    _batch_fp64(edges[:8], zs, acc[:8])

    sys.stderr.write(
        f"grid={grid_name} n_edge={edges.shape[0]} n_lat={n_lat} "
        f"evals/pass={n_eval} max_threads={max_threads} counts={counts}\n"
    )

    def best(fn, nt):
        set_num_threads(nt)
        b = math.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(edges, zs, acc)
            b = min(b, (time.perf_counter() - t0) / n_eval * 1e9)
        return b

    print("threads,fp64_ns,accux_ns,accux_over_fp64")
    for nt in counts:
        f = best(_batch_fp64, nt)
        a = best(_batch_accux, nt)
        print(f"{nt},{f:.4f},{a:.4f},{a / f:.4f}")


if __name__ == "__main__":
    main()
