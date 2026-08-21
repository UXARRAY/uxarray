"""Warming benchmark state in the interpreter every benchmark is forked from.

Under ``launch_method: forkserver`` asv imports the suite once and forks each
benchmark from that interpreter, so whatever a module prepares at import is
inherited copy-on-write instead of being paid for again by every benchmark
process. On this suite that is 0.5s of JIT and cache loading for the
connectivity kernels and 4.1s of case generation for the gca-gca drivers, each
of which was being repeated per benchmark.

Only work that leaves no numba thread pool behind may be warmed this way.
Running a ``parallel=True`` kernel -- or merely calling ``.compile()`` on one --
launches the pool, and numba's OpenMP layer is not fork-safe, with no at-fork
handler to rebuild it in the child. So this checks rather than trusts: the day a
warmed kernel goes parallel becomes a loud import error rather than a benchmark
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
