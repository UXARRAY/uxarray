"""Cached inputs for benchmarks whose subject is not reading a file.

asv gives every benchmark its own process, so a grid opened in ``setup`` is
opened once per benchmark rather than once per run -- around 160 times across
the suite once the dyamond resolutions are visible. For a mesh on local disk
that is a rounding error. For one on campaign storage it is most of the run, and
it is spent inside the benchmark's own timeout.

Benchmarks that do measure opening a file keep reading the real thing:
``quad_hexagon``, ``OpenGrid`` in ``mpas_dyamond``, ``import``, and the
cold-start peak-memory benchmarks. They take their paths from the registries
here too, so the suite declares its inputs in one place either way.

Two flavours, picked per benchmark:

``topology``
    the three arrays ``Grid.from_topology`` needs and nothing else, for
    benchmarks that mean to build the rest themselves -- 2.3MB of the 102MB
    120km MPAS file.
``grid`` / ``dataset``
    everything the reader produced, so a benchmark still gets the
    ``face_areas`` and connectivity variables an MPAS file carries on disk
    rather than silently measuring their construction.

One source read produces both, so choosing between them costs nothing.

Artifacts are keyed on the uxarray build as well as on the file, because an
artifact is one version's reader output and asv walks commits. That means a
fresh read per commit; ``prime`` therefore leaves the dyamond grids out unless
asked, and there is a CLI (``python -m benchmarks.helpers._fixtures``) to fill
the cache from a batch script instead of from inside a benchmark.
"""

import hashlib
import os
import tempfile
import urllib.request
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
    "prime",
]

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BENCHMARK_DIR.parent

_COOKBOOK_URL = (
    "https://github.com/ProjectPythia/unstructured-grid-viz-cookbook/raw/main/meshfiles"
)


def _cookbook(filename):
    """Path to a Cookbook mesh, fetched once if this checkout lacks it."""
    path = BENCHMARK_DIR / filename
    if not path.is_file():
        urllib.request.urlretrieve(f"{_COOKBOOK_URL}/{filename}", filename=path)
    return path


# Grids, and grid/data pairs, by mesh resolution.
OQU_GRIDS = {
    "480km": _cookbook("oQU480.grid.nc"),
    "120km": _cookbook("oQU120.grid.nc"),
}
OQU_DATASETS = {
    "480km": (OQU_GRIDS["480km"], _cookbook("oQU480.data.nc")),
    "120km": (OQU_GRIDS["120km"], _cookbook("oQU120.data.nc")),
}

DYAMOND_GRIDS = {
    "30km": Path("/glade/campaign/cisl/vast/uxarray/data/dyamond/30km/grid.nc"),
    "15km": Path("/glade/campaign/cisl/vast/uxarray/data/dyamond/15km/grid.nc"),
    "7.5km": Path("/glade/campaign/cisl/vast/uxarray/data/dyamond/7.5km/grid.nc"),
    "3.75km": Path("/glade/campaign/cisl/vast/uxarray/data/dyamond/3.75km/grid.nc"),
}

# Asked once here, rather than separately in each module that cares.
DYAMOND_AVAILABLE = all(path.exists() for path in DYAMOND_GRIDS.values())

GRIDS_BY_RESOLUTION = dict(OQU_GRIDS)
if DYAMOND_AVAILABLE:
    GRIDS_BY_RESOLUTION |= DYAMOND_GRIDS

# Two ladders rather than one, because which of them a benchmark belongs on is a
# per-benchmark decision: an algorithm that does not care which model wrote the
# mesh can take the wide one, while anything tied to the oQU pair -- or too slow
# to run four dyamond resolutions of -- stays on the narrow one.
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

CACHE_DIR_VAR = "UXARRAY_BENCH_CACHE_DIR"
PRIME_VAR = "UXARRAY_BENCH_PRIME"

# The arguments ``ux.Grid.from_topology`` takes, in order.
_TOPOLOGY_ARRAYS = ("node_lon", "node_lat", "face_node_connectivity")

# Artifact path -> what it holds, for this process.
_loaded = {}


def cache_dir():
    """Directory the cached artifacts live in.

    ``UXARRAY_BENCH_CACHE_DIR`` overrides the default, and on a cluster it
    should: the cache only pays off on a filesystem faster than the one holding
    the source grids, and putting it somewhere that outlives the job means each
    grid is read once per machine rather than once per job.
    """
    root = Path(os.environ.get(CACHE_DIR_VAR) or tempfile.gettempdir())
    cached = root / "uxarray-bench-fixtures"
    cached.mkdir(parents=True, exist_ok=True)
    return cached


def _artifact(source, flavour, suffix):
    """Where ``flavour`` of ``source`` is cached.

    ``source`` is a grid path, or a (grid, data) pair. Keyed on each file's size
    and mtime as well as its path, so a replaced source misses rather than being
    served something stale, and on the uxarray version, because the artifact is
    that version's reader output.
    """
    parts = [ux.__version__]
    for path in source:
        stat = os.stat(path)
        parts.append(f"{os.path.realpath(path)}:{stat.st_size}:{stat.st_mtime_ns}")
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    # The dyamond grids are all named ``grid.nc``; the directory above them is
    # what tells one resolution from the next.
    grid_path = Path(source[0])
    stem = f"{grid_path.parent.name}-{grid_path.stem}"
    return cache_dir() / f"{stem}-{flavour}-{digest}{suffix}"


def _write(dataset, artifact_path, writer):
    """Writes ``dataset`` to ``artifact_path``, atomically.

    Via a scratch name in the same directory, so a process racing this one sees
    either no artifact or a complete one, never a half-written file.
    """
    scratch = artifact_path.with_name(
        f"{artifact_path.stem}.{os.getpid()}.tmp{artifact_path.suffix}"
    )
    writer(dataset, scratch)
    os.replace(scratch, artifact_path)


def _read_dataset(artifact_path):
    """Reads back a cached ``xr.Dataset``.

    ``mask_and_scale=False`` is load-bearing: the connectivity variables carry
    ``_FillValue``, which xarray would otherwise consume, handing back float64
    where uxarray's njit kernels require int64 -- they fail to type rather than
    returning something wrong, but they do fail.
    """
    return xr.open_dataset(artifact_path, mask_and_scale=False).load()


def _build(source):
    """Reads ``source`` and writes every flavour of it, from the one read."""
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
        # A ``UxDataset`` is an ``xr.Dataset`` subclass, so it writes itself; the
        # grid half is cached separately just below.
        data_ds = uxds

    _write(
        {name: getattr(uxgrid, name).data for name in _TOPOLOGY_ARRAYS},
        _artifact(source[:1], "topology", ".npz"),
        # Uncompressed: written once, read by every benchmark process after.
        lambda arrays, path: np.savez(path, **arrays),
    )
    _write(uxgrid._ds, _artifact(source[:1], "grid", ".nc"), lambda ds, path: ds.to_netcdf(path))
    if data_ds is not None:
        _write(data_ds, _artifact(source, "data", ".nc"), lambda ds, path: ds.to_netcdf(path))


def _ensure(source, flavour, suffix):
    """Path to a cached artifact, building the source's artifacts if need be."""
    artifact_path = _artifact(source, flavour, suffix)
    if not artifact_path.exists():
        _build(source if flavour == "data" else source[:1])
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
            _loaded[artifact_path] = tuple(cached[name] for name in _TOPOLOGY_ARRAYS)
    return _loaded[artifact_path]


def _cached_grid_ds(grid_path):
    """The cached internal dataset of ``grid_path``, held for this process."""
    artifact_path = _ensure((Path(grid_path),), "grid", ".nc")
    if artifact_path not in _loaded:
        _loaded[artifact_path] = _read_dataset(artifact_path)
    return _loaded[artifact_path]


def cached_grid(grid_path):
    """A ``Grid`` carrying everything the reader found in ``grid_path``.

    A fresh ``Grid`` over a shallow copy each call, so a benchmark that
    populates or normalizes something does not hand its leftovers to the next
    repeat: uxarray assigns new variables into ``_ds`` rather than writing
    through the arrays, so the copy isolates that while the data stays shared.
    """
    return ux.Grid(_cached_grid_ds(grid_path).copy())


def cached_dataset(grid_path, data_path):
    """A ``UxDataset`` over ``data_path``, on the cached grid."""
    source = (Path(grid_path), Path(data_path))
    artifact_path = _ensure(source, "data", ".nc")
    if artifact_path not in _loaded:
        _loaded[artifact_path] = _read_dataset(artifact_path)
    return ux.UxDataset(_loaded[artifact_path].copy(), uxgrid=cached_grid(grid_path))


def prime(include_dyamond=None):
    """Fills the cache for every source the fixtures can serve.

    Returns the sources it had to read. Idempotent, and once warm costs a
    ``stat`` per file, so it is cheap to call ahead of every run.

    The dyamond grids are left out unless ``UXARRAY_BENCH_PRIME=all`` (or
    ``include_dyamond``) asks for them, because a filtered run should not pay
    for reading four grids off campaign storage that it will never touch. On a
    machine that does have them, prime from the CLI before ``asv run``.
    """
    if include_dyamond is None:
        include_dyamond = os.environ.get(PRIME_VAR, "").lower() == "all"

    sources = [(path,) for path in OQU_GRIDS.values()]
    sources += [(path,) for path in GRIDS_BY_FORMAT.values()]
    sources += list(OQU_DATASETS.values())
    if include_dyamond and DYAMOND_AVAILABLE:
        sources += [(path,) for path in DYAMOND_GRIDS.values()]

    read = []
    for source in sources:
        flavour, suffix = ("data", ".nc") if len(source) == 2 else ("grid", ".nc")
        if not _artifact(source, flavour, suffix).exists():
            _build(source)
            read.append(source)
    return read


class CachedFixtures:
    """Mixin for benchmarks whose subject is not reading a file.

    Holds the one ``setup_cache`` the suite shares. asv keys ``setup_cache`` on
    where it is defined and groups benchmarks by that key, so this single
    definition -- inherited by every such class in every module -- runs once per
    ``asv run`` rather than once per class. It returns ``None``, which asv reads
    as "no cache argument", so the benchmark signatures stay as they are.

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
    for source in prime(include_dyamond=True) or [None]:
        # Unbuffered and one line per source, so a batch log shows how far the
        # reading has got.
        print(f"  read {' + '.join(Path(p).name for p in source)}" if source else "  nothing to do", flush=True)
