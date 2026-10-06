"""Guarding numba's thread pool against being entered from several threads.

A ``parallel=True`` kernel called from a thread pool -- dask's threaded
scheduler, typically -- nests numba's pool under it. What that does depends on
numba's threading layer: ``tbb`` composes, ``omp`` starts a full team per
calling thread, and ``workqueue`` kills the process on concurrent entry.
"""

import contextlib
import functools
import threading

import numba
from numba import njit

_WORKQUEUE_LOCK = threading.Lock()


@functools.cache
def _threading_layer():
    # get_num_threads starts the layer, as a first parallel call would
    numba.get_num_threads()
    return numba.threading_layer()


@contextlib.contextmanager
def numba_pool():
    """Makes the enclosed parallel kernel call safe under the current layer:
    one call at a time under ``workqueue``, one thread per call off the main
    thread under ``omp``."""
    layer = _threading_layer()
    if layer == "workqueue":
        with _WORKQUEUE_LOCK:
            yield
    elif layer == "omp" and threading.current_thread() is not threading.main_thread():
        previous = numba.get_num_threads()
        numba.set_num_threads(1)
        try:
            yield
        finally:
            numba.set_num_threads(previous)
    else:
        yield


def parallel_njit(func):
    """``@njit(cache=True, parallel=True, nogil=True)``, called under
    :func:`numba_pool`. Only for kernels Python calls directly."""
    kernel = njit(cache=True, parallel=True, nogil=True)(func)

    @functools.wraps(func)
    def call(*args, **kwargs):
        with numba_pool():
            return kernel(*args, **kwargs)

    return call
