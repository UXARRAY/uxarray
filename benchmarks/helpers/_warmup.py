"""Warming benchmark state in the interpreter every benchmark is forked from.

Under ``launch_method: forkserver`` ASV imports the suite once and forks each
benchmark from that interpreter, so whatever a module prepares at import is
inherited copy-on-write.
"""

import os
import sys

import numba

__all__ = ["warm_in_parent", "will_run_benchmarks"]

# ``benchmark.py <mode>``: asv runs discovery, setup_cache and check as their own processes
_NON_RUNNING_MODES = frozenset({"discover", "setup_cache", "check"})

# Numba installs fork handlers for these two; ``omp`` is on its own.
_FORK_SAFE = frozenset({"tbb", "workqueue"})

_reported = False


def will_run_benchmarks():
    """Whether this interpreter is going to run a benchmark."""
    return not (
        os.path.basename(sys.argv[0]) == "benchmark.py"
        and len(sys.argv) > 1
        and sys.argv[1] in _NON_RUNNING_MODES
    )


def warm_in_parent(warm, what):
    """Runs ``warm``, then checks any pool it leaves behind survives a fork.

    ``what`` names the thing being warmed, for the report. A no-op in an
    interpreter that will not run a benchmark, per :func:`will_run_benchmarks`.
    """
    if not will_run_benchmarks():
        return

    warm()

    try:
        layer = numba.threading_layer()
    except ValueError:
        return  # nothing initialized a pool, so there is nothing to inherit

    if layer in _FORK_SAFE:
        return

    global _reported
    if _reported:
        return
    _reported = True
    print(
        f"asv: warming {what} left numba's {layer!r} thread pool behind, and "
        f"{layer!r} does not survive fork(). Forked benchmarks that run a "
        "parallel kernel will be killed by the OpenMP runtime. Install tbb in "
        "the benchmark environment, or set NUMBA_THREADING_LAYER=forksafe.",
        file=sys.stderr,
    )
