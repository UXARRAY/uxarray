import os
import subprocess
import sys

_CODE = """
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from uxarray.grid.coordinates import _construct_face_centroids

rng = np.random.default_rng(0)
n_node, n_face = 100_000, 400_000
nodes = [rng.random(n_node) for _ in range(3)]
face_nodes = rng.integers(0, n_node, (n_face, 4))
args = (*nodes, face_nodes, np.full(n_face, 4))

expected = _construct_face_centroids(*args)
with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(lambda _: _construct_face_centroids(*args), range(32)))
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
