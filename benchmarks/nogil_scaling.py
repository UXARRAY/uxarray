"""Thread scaling of a kernel that dask calls once per chunk.

Every other benchmark in this suite drives uxarray from a single Python
thread, so none of them can see whether a kernel releases the GIL. That makes
the flag invisible right up until it matters: under ``dask="parallelized"``
on the threaded scheduler, a kernel holding the GIL serializes across workers
with no error and no movement in any timing here.

This measures the one thing that does move -- wall time as the same kernel is
called from more threads at once.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

#: Two is enough to separate "overlaps" from "serializes", and fits the
#: 2-core runners CI uses.
N_THREADS = 2

#: Big enough that a call dwarfs thread hand-off, small enough to stay quick.
N_NODE = 200_000
N_EDGE = 800_000

_REPEATS = 5


class GILScaling:
    """Thread scaling of a representative Python-entry-point kernel.

    ``_construct_edge_node_distances`` is the clean probe: Python invokes it
    once per whole array, it is plain ``@njit`` rather than ``parallel=True``
    (so numba's own threadpool cannot muddy the reading), and its body is pure
    arithmetic over an edge-sized array.
    """

    def setup(self):
        from uxarray.grid.neighbors import _construct_edge_node_distances

        rng = np.random.default_rng(0)
        self.args = (
            rng.uniform(-180.0, 180.0, N_NODE),
            rng.uniform(-89.0, 89.0, N_NODE),
            rng.integers(0, N_NODE, size=(N_EDGE, 2)).astype(np.int64),
        )
        self.kernel = _construct_edge_node_distances
        self.kernel(*self.args)

    def _wall(self, n_threads):
        best = float("inf")
        for _ in range(_REPEATS):
            with ThreadPoolExecutor(n_threads) as pool:
                start = time.perf_counter()
                list(pool.map(lambda _: self.kernel(*self.args), range(n_threads)))
                best = min(best, time.perf_counter() - start)
        return best

    def track_gil_scaling(self):
        """wall(2 threads) / wall(1 thread).

        Near 1.0 while the kernel releases the GIL. Climbs toward 2.0 if the
        flag is ever dropped -- the regression this exists to catch.
        """
        return self._wall(N_THREADS) / self._wall(1)
