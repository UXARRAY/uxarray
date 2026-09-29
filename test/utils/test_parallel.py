import os
import subprocess
import sys

_CODE = """
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from uxarray.grid.coordinates import _construct_face_centroids
from uxarray.grid.neighbors import Neighborhood

rng = np.random.default_rng(0)
n_node, n_face = 100_000, 400_000
nodes = [rng.random(n_node) for _ in range(3)]
face_nodes = rng.integers(0, n_node, (n_face, 4))
n_nodes_per_face = np.full(n_face, 4)
centroids = (*nodes, face_nodes, n_nodes_per_face)

data = rng.random((8, n_node))
counts = np.full(n_node, 16)
starts = np.arange(n_node) * 16
flat = rng.integers(0, n_node, n_node * 16)
mean = (data, flat, starts, counts, 0.0)

for kernel, args in [
    (_construct_face_centroids, centroids),
    (Neighborhood._mean_kernel, mean),
]:
    expected = kernel(*args)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: kernel(*args), range(32)))
    for result in results:
        np.testing.assert_allclose(result, expected)
"""


def test_parallel_kernels_called_from_a_thread_pool():
    """Numba's ``workqueue`` layer aborts the process when two threads enter a
    parallel region at once, so this passes only if the kernels serialize."""
    result = subprocess.run(
        [sys.executable, "-c", _CODE],
        capture_output=True,
        text=True,
        env={**os.environ, "NUMBA_THREADING_LAYER": "workqueue"},
    )
    assert result.returncode == 0, result.stderr
