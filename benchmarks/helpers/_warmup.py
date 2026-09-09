"""Warming benchmark state in the interpreter every benchmark is forked from.

Under ``launch_method: forkserver`` ASV imports the suite once and forks each
benchmark from that interpreter, so whatever a module prepares at import is
inherited copy-on-write.

Warming can leave a numba thread pool behind. *Compiling* a ``parallel=True``
kernel initializes the threading layer -- before the kernel is ever run -- and
``cache=True`` only avoids that while the on-disk cache is warm, which it is
not on a fresh runner. Whether an inherited pool is safe then depends entirely
on which layer numba picked:

* ``tbb`` and ``workqueue`` install fork handlers and come through it intact.
* ``omp`` does not. Where that is libgomp -- Linux, unless TBB is installed --
  every forked child that touches a parallel kernel is killed with
  ``Terminating: fork() called from a process already using GNU OpenMP``.

numba prefers TBB, so a machine that has it cannot reproduce the failure at
all; the benchmark environment pins it instead (see ``tbb`` and
``NUMBA_THREADING_LAYER`` in ``asv.conf.json``). This reports when that has not
taken effect rather than raising. Raising fails the module's import in the
parent *and* again in every child that re-imports it, so it converts "these
benchmarks are slower than they could be" into "this commit produced no
results" -- and the pool exists either way, and asv forks either way.
"""

import os
import sys

import numba

__all__ = ["warm_in_parent", "will_run_benchmarks"]

# ``benchmark.py <mode>``: asv runs discovery, setup_cache and check as their own
# processes, and none of them goes on to run a benchmark, so warming in them is
# time bought for nobody. ``run_server`` -- the forkserver parent, the one whose
# state every benchmark inherits -- is deliberately absent.
_NON_RUNNING_MODES = frozenset({"discover", "setup_cache", "check"})

# Numba installs fork handlers for these two; ``omp`` is on its own.
_FORK_SAFE = frozenset({"tbb", "workqueue"})

_reported = False


def will_run_benchmarks():
    """Whether this interpreter is going to run a benchmark.

    Conservative by construction: an argv this does not recognize is assumed to
    be a benchmark run, so a change in how asv invokes itself costs the warm
    twice over rather than losing it where it counts.
    """
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
