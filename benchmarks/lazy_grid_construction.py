"""What ``Grid`` construction pulls off disk before the caller asks for it.

A chunked ``open_grid`` is supposed to cost metadata and nothing else. It did
not: ``_set_desired_longitude_range`` decided whether to wrap by asking
``lon.max() > 180``, and on a dask-backed coordinate that reduction is a
compute -- one whole longitude array read and reduced per constructor call,
and the constructor runs again on every ``isel`` and every ``copy()``.

Wall time will not show this on a test-sized mesh, and it is the wrong
instrument anyway: the quantity that changed is discrete. So the number here
is a count of dask graph executions, which is exact, has no variance, and
moves by one the moment the reduction comes back.
"""

import os
from pathlib import Path

from dask.callbacks import Callback

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]

grid_path = current_path / "test" / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"

#: Small enough to leave several chunks per coordinate, which is what makes a
#: reduction over one visible as a scheduled graph rather than a fused no-op.
CHUNKS = {"n_node": 1000}


class _CountComputes(Callback):
    def __init__(self):
        self.n = 0

    def _start(self, dsk):
        self.n += 1


class LazyGridConstruction:
    def setup(self):
        # Opening a grid for the first time in a process pulls in xarray's
        # backend machinery and the netCDF library
        ux.open_grid(grid_path, chunks=CHUNKS)

    def track_computes_open_grid_chunked(self):
        """Dask graph executions during a chunked ``open_grid``.

        Three before the longitude wrap went elementwise, two after. The two
        that remain are ``_standardize_connectivity``'s ``conn.isnull().any()``
        in ``io/_ugrid.py``, reached once from ``match_chunks_to_ugrid`` and
        once from ``Grid.from_dataset`` -- a separate site, on connectivity
        rather than coordinates, and not addressed here.
        """
        counter = _CountComputes()
        with counter:
            ux.open_grid(grid_path, chunks=CHUNKS)
        return counter.n

    def track_computes_isel(self):
        """Dask graph executions during a subset of an already-open grid.

        ``isel`` builds a new ``Grid``, so it paid the constructor's reduction
        on every call. Two before, one after. What remains is
        ``_slice_face_indices`` (``grid/slice.py``) materializing the
        face-node connectivity it slices by -- again connectivity, not
        coordinates.
        """
        uxgrid = ux.open_grid(grid_path, chunks=CHUNKS)
        counter = _CountComputes()
        with counter:
            uxgrid.isel(n_face=slice(0, 100))
        return counter.n
