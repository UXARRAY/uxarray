"""Resolves how many threads a benchmark run may use.

numba treats ``NUMBA_NUM_THREADS`` as a ceiling fixed at import, so the count
must be in the environment before ``asv run`` (asv passes ``os.environ`` on).
It is deliberately absent from ``env_nobuild``, which would override the shell::

    export NUMBA_NUM_THREADS=$(python -m benchmarks.helpers._threads)
    asv run ...

Defaults to physical cores: SMT siblings measured ~1.28x slower per thread on
these memory-bound kernels. ``UXARRAY_BENCH_THREADS`` overrides it with an
integer, ``physical`` or ``logical``.
"""

import os
import subprocess
import sys

__all__ = ["logical_cores", "physical_cores", "resolve"]

_ENV_VAR = "UXARRAY_BENCH_THREADS"


def logical_cores():
    """Schedulable CPUs, honouring the affinity mask (batch scheduler, ``taskset``)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def physical_cores():
    """Cores rather than hardware threads, or the logical count if unknown.

    Shells out rather than adding ``psutil`` to the asv environment matrix.
    """
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout
            return max(1, int(out.strip()))
        if sys.platform.startswith("linux"):
            # Distinct (core, socket) pairs; core ids alone collapse sockets.
            out = subprocess.run(
                ["lscpu", "-p=core,socket"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout
            pairs = {
                line for line in (l.strip() for l in out.splitlines())
                if line and not line.startswith("#")
            }
            if pairs:
                return max(1, len(pairs))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return logical_cores()


def resolve(spec=None):
    """The thread count to run with.

    ``spec`` defaults to ``$UXARRAY_BENCH_THREADS``, then ``physical``. Invalid
    values warn and fall back to physical rather than blocking the run.
    """
    if spec is None:
        spec = os.environ.get(_ENV_VAR, "").strip()
    spec = (spec or "physical").lower()

    if spec == "logical":
        return logical_cores()
    if spec != "physical":
        try:
            return max(1, int(spec))
        except ValueError:
            print(
                f"{_ENV_VAR}={spec!r} is not an integer, 'physical' or 'logical'; "
                "using the physical core count",
                file=sys.stderr,
            )
    # Never hand back more than this process may schedule on.
    return min(physical_cores(), logical_cores())


if __name__ == "__main__":
    print(resolve())
