"""Does the package's compiled work actually overlap across threads?

Every other benchmark in this suite drives uxarray from a single Python
thread, so none of them can observe whether a kernel releases the GIL. That
makes ``nogil=True`` invisible to CI right up until the moment it matters:
under ``dask="parallelized"`` on the threaded scheduler, a kernel that holds
the GIL serializes across workers silently -- no error, no warning, and no
movement in any timing here.

These benchmarks close that hole by measuring the one thing that does move:
how wall time responds to running the same kernel from N threads at once.

    ratio = wall(N threads) / wall(1 thread)

A kernel that releases the GIL overlaps, so the ratio stays near 1.0. A
kernel that holds it runs the calls back to back, so the ratio climbs toward
N. The control below is the same kernel recompiled without the flag, which
makes the contrast self-evidencing: if the harness itself were broken, both
numbers would move together.

This measures the flag rather than a user-facing operation, which is a fair
criticism of it -- but until ``core/`` grows the chunked call paths (the
refactor plan's Tier 1), no user-facing operation reaches a kernel from two
threads, and there is nothing else to measure. Retire these in favour of a
real chunked workload once one exists.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .helpers._peakmem import numba_threads

#: Threads to contend with. Two is enough to separate "overlaps" from
#: "serializes" and fits the 2-core runners CI uses.
N_THREADS = 2

#: Big enough that a call dwarfs thread hand-off, small enough to stay quick.
N_NODE = 200_000
N_EDGE = 800_000

_REPEATS = 5


class GILScaling:
    """Thread scaling of a representative Python-entry-point kernel.

    ``_construct_edge_node_distances`` is the clean probe: it is one of the
    kernels Python invokes once per whole array, it is plain ``@njit`` rather
    than ``parallel=True`` (so numba's own threadpool cannot muddy the
    reading), and its body is pure arithmetic over an edge-sized array.
    """

    def setup(self):
        from numba import njit

        from uxarray.grid.neighbors import _construct_edge_node_distances

        rng = np.random.default_rng(0)
        self.args = (
            rng.uniform(-180.0, 180.0, N_NODE),
            rng.uniform(-89.0, 89.0, N_NODE),
            rng.integers(0, N_NODE, size=(N_EDGE, 2)).astype(np.int64),
        )

        self.kernel = _construct_edge_node_distances

        # Same source, same options, minus the flag. Anything that moves both
        # this and the kernel above is the machine, not the GIL.
        # ``nopython`` is implied by njit and warns if passed back in.
        options = {
            k: v
            for k, v in _construct_edge_node_distances.targetoptions.items()
            if k not in ("nogil", "nopython")
        }
        self.control = njit(**options)(_construct_edge_node_distances.py_func)

        for fn in (self.kernel, self.control):
            fn(*self.args)

    def _ratio(self, fn):
        with numba_threads(1):
            one = self._wall(fn, 1)
            many = self._wall(fn, N_THREADS)
        return many / one

    def _wall(self, fn, n_threads):
        best = float("inf")
        for _ in range(_REPEATS):
            with ThreadPoolExecutor(n_threads) as pool:
                start = time.perf_counter()
                list(pool.map(lambda _: fn(*self.args), range(n_threads)))
                best = min(best, time.perf_counter() - start)
        return best

    def track_gil_scaling(self):
        """wall(2 threads) / wall(1 thread) for the shipped kernel.

        Near 1.0 while the kernel releases the GIL. Climbs toward 2.0 if the
        flag is ever dropped -- which is the regression this exists to catch.
        """
        return self._ratio(self.kernel)

    def track_gil_scaling_control(self):
        """The same kernel recompiled without ``nogil``, as a reference.

        Expected near 2.0. If this ever reads near 1.0 the measurement has
        stopped working and ``track_gil_scaling`` means nothing.
        """
        return self._ratio(self.control)
