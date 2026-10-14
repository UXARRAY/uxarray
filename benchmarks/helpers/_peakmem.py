
import contextlib
import os
import subprocess
import sys
import tempfile

import numba

__all__ = ["peak_allocated", "numba_threads", "subprocess_peak_rss"]


def peak_allocated(build):
    """Bytes held at the high-water allocation point of ``build``.

    ``tracemalloc.start`` begins with an empty trace table, so whatever the
    process is already holding -- including everything allocated in ``setup`` --
    is excluded, and only what ``build`` allocates counts. A ``reset_peak()``
    here would be a no-op for that reason.
    """
    # Imported here rather than at module scope: asv preimports every benchmark
    # module under its default ``forkserver`` launch method
    import tracemalloc

    if tracemalloc.is_tracing():
        raise RuntimeError("tracemalloc is already tracing")

    # nframe=1: the reported peak is identical at any traceback depth, while the
    # cost is not -- nframe=25 runs 26x slower.
    tracemalloc.start(1)
    try:
        build()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


@contextlib.contextmanager
def numba_threads(n):
    """Runs the block with numba's thread pool held at ``n``.

    Tracing serializes on tracemalloc's global allocator lock, so a
    ``parallel=True`` kernel under :func:`peak_allocated` degrades badly as
    threads contend for it.
    """
    restore = numba.get_num_threads()
    numba.set_num_threads(n)
    try:
        yield
    finally:
        numba.set_num_threads(restore)


# ``ru_maxrss`` is bytes on macOS and kilobytes elsewhere, mirroring
# ``asv_runner/benchmarks/_maxrss.py:117,132``.
_MAXRSS_TO_BYTES = 1 if sys.platform == "darwin" else 1024


def subprocess_peak_rss(statement):
    """Peak resident bytes of a fresh interpreter that has run ``statement``.

    For the cases an in-process metric cannot reach: measuring an import, whose
    cost is already paid before a ``peakmem_*`` body runs, and measuring
    ``numba.typed`` containers, which tracemalloc does not see at all.

    The child reports its own ``ru_maxrss`` rather than the parent reading
    ``RUSAGE_CHILDREN``, which is a maximum over every child that has exited and
    so would not isolate this one.
    """
    with tempfile.TemporaryDirectory() as scratch:
        report_path = os.path.join(scratch, "peak_rss")
        reporter = (
            "import resource\n"
            f"exec({statement!r})\n"
            "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
            # Reported through a file rather than stdout, which is not ours
            # alone: uxarray prints coordinate warnings there for some grids,
            # and the number would arrive with prose in front of it.
            f"open({report_path!r}, 'w').write(str(peak))\n"
        )
        subprocess.run(
            [sys.executable, "-c", reporter],
            capture_output=True,
            text=True,
            check=True,
        )
        with open(report_path) as report:
            return int(report.read()) * _MAXRSS_TO_BYTES
