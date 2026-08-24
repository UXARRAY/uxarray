"""Warming benchmark state in the interpreter every benchmark is forked from.

Under ``launch_method: forkserver`` ASV imports the suite once and forks each
benchmark from that interpreter, so whatever a module prepares at import is
inherited copy-on-write.

Only tasks that leaves no numba thread pool behind may be warmed this way.
Running a ``parallel=True`` kernel or  calling ``.compile()`` on one
launches the pool, and numba's OpenMP layer is not fork-safe. So this checks when a
warmed kernel goes parallel, it becomes an import error rather than a benchmark
that hangs on a cluster.
"""

import numba

__all__ = ["warm_in_parent"]


def warm_in_parent(warm, what):
    """Runs ``warm``, then fails if it left a numba thread pool behind.

    ``what`` names the thing being warmed, for the error message.
    """
    warm()
    try:
        numba.threading_layer()
    except ValueError:
        return  # nothing launched a pool, which is what makes this inheritable
    raise RuntimeError(
        f"warming {what} started numba's thread pool, which a forked benchmark "
        "cannot safely inherit -- warm it from setup() instead, now that these "
        "kernels run in parallel"
    )
