"""What ``Grid`` construction pulls off disk before the caller asks for it.

A chunked ``open_grid`` is supposed to cost metadata and nothing else. It did
not: ``_set_desired_longitude_range`` decided whether to wrap by asking
``lon.max() > 180``, and on a dask-backed coordinate that reduction is a
compute -- one whole longitude array read and reduced per constructor call,
and the constructor runs again on every ``isel`` and every ``copy()``.

Two instruments. ``LazyGridConstruction`` counts dask graph executions on a
test-sized mesh: exact, no variance, and it moves by one the moment a
reduction comes back. ``OpenGridChunked`` is the wall-time and peak-memory
view of the same thing across two resolutions, which is what the rest of the
suite -- every other ``open_grid`` in it is eager -- cannot see.
"""

import math
import os
import warnings
from pathlib import Path

from dask.callbacks import Callback

import uxarray as ux

from .helpers._fixtures import OQU_GRIDS, OQU_RESOLUTIONS
from .helpers._peakmem import peak_allocated

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


#: Chunks per grid dimension, held fixed across resolutions so the graph is
#: the same shape at both and only the data under it grows.
N_CHUNKS = 4


class OpenGridChunked:
    """``open_grid`` with ``chunks=``, across both oQU resolutions.

    Reads the file directly rather than through ``CachedFixtures``, because
    reading the file is the subject here. The 120km mesh is ~16x the 480km one
    (59,329 nodes against 3,947).

    Track each resolution against its own history, not against the other. The
    two files are different formats -- oQU480.grid.nc is netCDF4/HDF5,
    oQU120.grid.nc is netCDF3 -- and the HDF5 open costs more, so 480km reads
    *slower* than 120km despite a sixteenth of the data.
    """

    param_names = ["resolution"]
    params = [OQU_RESOLUTIONS]

    def setup(self, resolution):
        self.grid_path = OQU_GRIDS[resolution]
        # An eager open for the sizes, which also pays the first-open cost of
        # the netCDF backend here rather than in the first sample.
        uxgrid = ux.open_grid(self.grid_path)
        self.chunks = {
            "n_node": math.ceil(uxgrid.n_node / N_CHUNKS),
            "n_face": math.ceil(uxgrid.n_face / N_CHUNKS),
        }
        self._open()

    def _open(self):
        with warnings.catch_warnings():
            # oQU480 stores layerThickness, ssh and zMid as one chunk of all
            # 1791 cells, so any n_face chunking splits them and xarray warns
            # once per open. They are data variables the grid reader drops.
            warnings.filterwarnings(
                "ignore", message="The specified chunks separate the stored chunks"
            )
            # A copy: open_grid adds the source-format dimension names to the
            # dict it is handed (match_chunks_to_ugrid), so reusing one would
            # have every sample after the first open with a different argument.
            return ux.open_grid(self.grid_path, chunks=dict(self.chunks))

    def time_open_grid(self, resolution):
        self._open()

    def track_peakmem_open_grid(self, resolution):
        """Transient high-water allocation of a chunked ``open_grid``."""
        return peak_allocated(self._open)

    track_peakmem_open_grid.unit = "bytes"
