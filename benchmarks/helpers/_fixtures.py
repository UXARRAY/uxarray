"""Cached inputs for benchmarks whose subject is not reading a file.

ASV gives every benchmark its own process, so a grid opened in ``setup`` is
opened once per benchmark rather than once per run. This module provides flexible
access to files across benchmark runs by caching the needed files.

Benchmarks that do measure opening a file behave as before.

Two flavors:

``topology``
    the three arrays ``Grid.from_topology`` needs and nothing else
``grid`` / ``dataset``
    everything the reader produced from ``Grid.open_grid`` and
    ``Grid.open_dataset``

Artifacts are keyed on both the uxarray build and the files, because an
artifact is one version's reader output and ASV diffs commits. Likewise, there's a
fresh read per commit. ``prime`` therefore leaves the dyamond grids out unless
asked, and there is a CLI (``python -m benchmarks.helpers._fixtures``) to fill
the cache from a batch script instead of from inside a benchmark.

Three environment variables tune all of this:

``UXARRAY_BENCH_CACHE_DIR``
    root the ``_io_cache`` directory is created under, in place of
    ``benchmarks/`` -- worth pointing at local scratch when the checkout itself
    lives on a shared filesystem
``UXARRAY_BENCH_PRIME``
    ``all`` primes the dyamond grids as well, which ``prime`` skips by default
``UXARRAY_BENCH_PRELOAD``
    any non-empty value has ``preload_topologies`` do its work; unset, it is a
    no-op, since preloading only pays off under ``launch_method: forkserver``
"""

import hashlib
import multiprocessing
import os
import urllib.request
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

import uxarray as ux

__all__ = [
    "ALL_RESOLUTIONS",
    "DYAMOND_AVAILABLE",
    "DYAMOND_GRIDS",
    "GRIDS_BY_FORMAT",
    "GRIDS_BY_RESOLUTION",
    "OQU_DATASETS",
    "OQU_GRIDS",
    "OQU_RESOLUTIONS",
    "QUAD_HEXAGON_DATASET",
    "CachedFixtures",
    "cache_dir",
    "cached_dataset",
    "cached_grid",
    "cached_topology",
    "preload_topologies",
    "prime",
]

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BENCHMARK_DIR.parent

_COOKBOOK_MESH_URL = (
    "https://github.com/ProjectPythia/unstructured-grid-viz-cookbook/raw/main/meshfiles"
)


def _cookbook_mesh(filename):
    """Path to a Cookbook mesh, fetched once if this checkout lacks it."""
    path = BENCHMARK_DIR / filename
    if not path.is_file():
        urllib.request.urlretrieve(f"{_COOKBOOK_MESH_URL}/{filename}", filename=path)
    return path


# Grids and grid/data pairs, by mesh resolution.
OQU_GRIDS = {
    "480km": _cookbook_mesh("oQU480.grid.nc"),
    "120km": _cookbook_mesh("oQU120.grid.nc"),
}
OQU_DATASETS = {
    "480km": (OQU_GRIDS["480km"], _cookbook_mesh("oQU480.data.nc")),
    "120km": (OQU_GRIDS["120km"], _cookbook_mesh("oQU120.data.nc")),
}

DYAMOND_GRIDS = {
    "30km": Path("/glade/campaign/cisl/vast/uxarray/data/dyamond/30km/grid.nc"),
    "15km": Path("/glade/campaign/cisl/vast/uxarray/data/dyamond/15km/grid.nc"),
    "7.5km": Path("/glade/campaign/cisl/vast/uxarray/data/dyamond/7.5km/grid.nc"),
    "3.75km": Path("/glade/campaign/cisl/vast/uxarray/data/dyamond/3.75km/grid.nc"),
}

# Find out which files are actually available
DYAMOND_AVAILABLE = all(path.exists() for path in DYAMOND_GRIDS.values())

GRIDS_BY_RESOLUTION = dict(OQU_GRIDS)
if DYAMOND_AVAILABLE:
    GRIDS_BY_RESOLUTION |= DYAMOND_GRIDS

OQU_RESOLUTIONS = list(OQU_GRIDS)
ALL_RESOLUTIONS = list(GRIDS_BY_RESOLUTION)

# Grids by source format, for benchmarks whose axis is the reader rather than the
# mesh size. ``mpas-oQU480`` is the same 1,791-face mesh as ``480km`` above,
# reached through the copy in the repo instead of the Cookbook download.
GRIDS_BY_FORMAT = {
    "ugrid-quad-hexagon": REPO_DIR / "test" / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc",
    "ugrid-geoflow": REPO_DIR / "test" / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc",
    "scrip-outCSne8": REPO_DIR / "test" / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc",
    "mpas-oQU480": REPO_DIR / "test" / "meshfiles" / "mpas" / "QU" / "oQU480.231010.nc",
}

QUAD_HEXAGON_DATASET = (
    GRIDS_BY_FORMAT["ugrid-quad-hexagon"],
    REPO_DIR / "test" / "meshfiles" / "ugrid" / "quad-hexagon" / "data.nc",
)

# Per path, which artifacts are actually loaded
_loaded = {}


def cache_dir():
    """Directory the cached artifacts live in: ``benchmarks/_io_cache``."""
    root = Path(os.environ.get("UXARRAY_BENCH_CACHE_DIR") or BENCHMARK_DIR)
    cached = root / "_io_cache"
    cached.mkdir(parents=True, exist_ok=True)
    return cached


def _artifact(source, flavor, suffix):
    """Where ``flavor`` of ``source`` is cached.

    ``source`` is a grid path, or a (grid, data) pair. Keyed on each file's size
    and mtime as well as its path, so a replaced source misses rather than being
    served something stale.
    """
    parts = []
    for path in source:
        stat = os.stat(path)
        parts.append(f"{os.path.realpath(path)}:{stat.st_size}:{stat.st_mtime_ns}")
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    # The dyamond grids are all named ``grid.nc``; the directory above them is
    # what tells one resolution from the next.
    grid_path = Path(source[0])
    stem = f"{grid_path.parent.name}-{grid_path.stem}"
    return cache_dir() / f"{stem}-{flavor}-{digest}{suffix}"


def _write(dataset, artifact_path, writer):
    """Writes ``dataset`` to ``artifact_path`` atomically.

    Via a scratch name in the same directory, so a process racing this one sees
    either no artifact or a complete one.
    """
    if artifact_path.exists():
        # If the path exists, someone else is already working
        return
    scratch = artifact_path.with_name(
        f"{artifact_path.stem}.{uuid.uuid4().hex}.tmp{artifact_path.suffix}"
    )
    writer(dataset, scratch)
    os.replace(scratch, artifact_path)


def _read_dataset(artifact_path):
    """Reads back a cached ``xr.Dataset``."""
    return xr.open_dataset(artifact_path, mask_and_scale=False).load()


def _build(source):
    """Reads ``source`` and writes every flavor of it, from the one read."""
    grid_path = source[0]
    if len(source) == 1:
        uxgrid = ux.open_grid(grid_path)
        uxgrid._ds.load()
        data_ds = None
    else:
        uxds = ux.open_dataset(*source)
        uxds.load()
        uxgrid = uxds.uxgrid
        uxgrid._ds.load()
        # A ``UxDataset`` grid is cached separately.
        data_ds = uxds

    _write(
        {name: getattr(uxgrid, name).data for name in ["node_lon", "node_lat", "face_node_connectivity"]},
        _artifact(source[:1], "topology", ".npz"),
        # Uncompressed: written once, read by every benchmark process after.
        lambda arrays, path: np.savez(path, **arrays),
    )
    _write(uxgrid._ds, _artifact(source[:1], "grid", ".nc"), lambda ds, path: ds.to_netcdf(path))
    if data_ds is not None:
        _write(data_ds, _artifact(source, "data", ".nc"), lambda ds, path: ds.to_netcdf(path))


def _ensure(source, flavor, suffix):
    """Path to a cached artifact, building the source's artifacts if need be."""
    artifact_path = _artifact(source, flavor, suffix)
    if not artifact_path.exists():
        _build(source if flavor == "data" else source[:1])
    return artifact_path


def cached_topology(grid_path):
    """``(node_lon, node_lat, face_node_connectivity)`` for ``grid_path``.

    The arrays are shared rather than copied: ``Grid.from_topology`` wraps
    ``node_lon`` and ``node_lat`` without copying them, and the construction
    routines assign to ``Grid._ds`` rather than writing through their inputs,
    which is what makes one copy per process safe. Treat them as read-only.
    """
    artifact_path = _ensure((Path(grid_path),), "topology", ".npz")
    if artifact_path not in _loaded:
        with np.load(artifact_path) as cached:
            _loaded[artifact_path] = tuple(cached[name] for name in ["node_lon", "node_lat", "face_node_connectivity"])
    return _loaded[artifact_path]


def _cached_grid_ds(grid_path):
    """The cached internal dataset of ``grid_path``, held for this process."""
    artifact_path = _ensure((Path(grid_path),), "grid", ".nc")
    if artifact_path not in _loaded:
        _loaded[artifact_path] = _read_dataset(artifact_path)
    return _loaded[artifact_path]


def cached_grid(grid_path):
    """A fresh ``Grid`` carrying everything the reader found in ``grid_path`` via shallow copy."""
    return ux.Grid(_cached_grid_ds(grid_path).copy())


def cached_dataset(grid_path, data_path):
    """A ``UxDataset`` over ``data_path``, on the cached grid."""
    source = (Path(grid_path), Path(data_path))
    artifact_path = _ensure(source, "data", ".nc")
    if artifact_path not in _loaded:
        _loaded[artifact_path] = _read_dataset(artifact_path)
    return ux.UxDataset(_loaded[artifact_path].copy(), uxgrid=cached_grid(grid_path))


def prime(include_dyamond=None, workers=1):
    """Fills the cache for every source the fixtures can serve.

    Returns the sources it had to read. Idempotent, and once warm costs a
    ``stat`` per file, so it is cheap to call ahead of every run.

    ``workers`` reads that many sources at once, which is worth having when the
    sources differ wildly in size and live somewhere slow: the small ones finish
    while a larger grid is still being read, instead of queueing behind all the
    file fetches. Only pays off when a read is slower than a process start due to
    the new interpreter startup. Processes rather than threads because the standard
    netCDF/HDF5 stack is not generally thread-safe for concurrent opens.
    """
    if include_dyamond is None:
        include_dyamond = os.environ.get("UXARRAY_BENCH_PRIME", "").lower() == "all"

    sources = [(path,) for path in OQU_GRIDS.values()]
    sources += [(path,) for path in GRIDS_BY_FORMAT.values()]
    sources += list(OQU_DATASETS.values())
    if include_dyamond and DYAMOND_AVAILABLE:
        sources += [(path,) for path in DYAMOND_GRIDS.values()]

    missing = []
    for source in sources:
        flavor, suffix = ("data", ".nc") if len(source) == 2 else ("grid", ".nc")
        if not _artifact(source, flavor, suffix).exists():
            missing.append(source)

    # Reading a (grid, data) pair produces that grid's artifacts too
    paired_grids = {source[0] for source in missing if len(source) == 2}
    missing = [
        source for source in missing if len(source) == 2 or source[0] not in paired_grids
    ]

    if workers > 1 and len(missing) > 1:
        # Largest first: with a pool, the longest read should start earliest, or
        # it lands last and everything waits on it.
        missing.sort(key=lambda source: -sum(os.path.getsize(path) for path in source))

        # Spawned. This process has the netCDF/HDF5 library loaded by the time it primes,
        # and a fresh interpreter per worker keeps that state out of the children.
        with ProcessPoolExecutor(
            max_workers=min(workers, len(missing)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            list(pool.map(_build, missing))
    else:
        for source in missing:
            _build(source)
    return missing


def preload_topologies(grid_paths):
    """Loads topologies here so forked benchmarks inherit them.

    Reading an artifact caches it on disk, not in the next process: under
    ``launch_method: forkserver`` each benchmark is forked from the interpreter
    that imported the suite, so it starts with whatever *that* process holds and
    reads its own copy of everything else.

    Safe to call at import: reading arrays starts no numba thread pool, which is
    what a forked child cannot inherit (see :mod:`benchmarks.helpers._warmup`).
    """
    if not os.environ.get("UXARRAY_BENCH_PRELOAD"):
        return 0
    loaded = 0
    for grid_path in grid_paths:
        cached_topology(grid_path)  # held by the process-level memo from here on
        loaded += 1
    return loaded


class CachedFixtures:
    """Mixin for benchmarks whose subject is not reading a file.

    Holds the one canonical ``setup_cache`` the suite shares. ASV keys ``setup_cache``
    on where it is defined and groups benchmarks by that key, so this single
    definition runs once per ``asv run`` rather than once per class. It returns
    ``None``, which asv reads as "no cache argument".

    The accessors are re-exported as methods so a ``setup`` reads as
    ``self.cached_grid(...)`` instead of importing the module's functions
    alongside its registries.
    """

    def setup_cache(self):
        prime()

    # Reading grids off campaign storage is the point of the cache, and does not
    # fit in a benchmark-sized timeout.
    setup_cache.timeout = 7200

    cached_topology = staticmethod(cached_topology)
    cached_grid = staticmethod(cached_grid)
    cached_dataset = staticmethod(cached_dataset)


if __name__ == "__main__":
    # Fills the cache ahead of ``asv run``, so no benchmark -- and not even
    # ``setup_cache`` -- pays for reading a source grid. Worth a line in a batch
    # script whenever the dyamond grids are in play.
    print(f"fixture cache: {cache_dir()}", flush=True)

    # Four at a time only where the dyamond grids are readable, since those are
    # the reads worth overlapping: on the oQU pair alone, priming in parallel is
    # slower than doing it sequentially (1.69s against 0.52s), because starting
    # an interpreter costs more than reading a small local file.
    for source in prime(include_dyamond=True, workers=4 if DYAMOND_AVAILABLE else 1) or [None]:
        print(f"  read {' + '.join(Path(p).name for p in source)}" if source else "  nothing to do", flush=True)
