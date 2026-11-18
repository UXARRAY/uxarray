"""Delete numba's on-disk cache for ``uxarray``.

Numba stamps each ``cache=True`` index (``*.nbi``) with the ``(mtime, size)`` of the
source file that *defines* the function, and nothing else. A kernel that calls a jitted
function from another module has that callee's code linked into its own cached object,
so editing the callee leaves the caller's cache entry "valid" with the old body baked in.

The test suite calls ``clear_if_shared_modules_changed`` at session start, which clears
the cache only when the jitted modules that others call into (``SHARED_MODULES``) differ
from the last run. Outside pytest (notebooks, scripts), run this by hand after editing
one of them::

    python ci/clean_numba_cache.py [--dry-run]
"""

import argparse
import hashlib
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "uxarray"
CACHE_SUFFIXES = {".nbi", ".nbc"}
# Jitted modules whose kernels are compiled into kernels of *other* modules. Add a module
# here when another module's jitted code starts calling into it.
SHARED_MODULES = [
    "utils/computing.py",
    "utils/numba_math.py",
    "grid/angles.py",
    "grid/arcs.py",
    "grid/area.py",
    "grid/coordinates.py",
    "grid/geometry.py",
    "grid/intersections.py",
    "grid/point_in_face.py",
    "grid/utils.py",
]
# Digest of SHARED_MODULES as of the last clear; __pycache__ keeps it out of git.
STAMP = PACKAGE_DIR / "__pycache__" / "numba_shared_modules.sha256"
# Numba files caches in NUMBA_CACHE_DIR under a hash of each source directory, so they
# can't be told apart from other projects' entries; leave them alone rather than guess.
EXTERNAL_CACHE_WARNING = (
    "NUMBA_CACHE_DIR is set, so uxarray's numba cache lives there and was not cleared "
    "(to avoid touching other projects' entries); clear it by hand."
)


def _cache_files():
    """Yield numba cache files stored next to the package's sources."""
    for path in PACKAGE_DIR.rglob("__pycache__/*"):
        if path.suffix in CACHE_SUFFIXES:
            yield path


def _shared_modules_digest():
    digest = hashlib.sha256()
    for name in SHARED_MODULES:
        path = PACKAGE_DIR / name
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
    return digest.hexdigest()


def _write_stamp(digest):
    STAMP.parent.mkdir(exist_ok=True)
    STAMP.write_text(digest)


def clear(dry_run=False):
    """Delete every cache file; return how many there were."""
    files = sorted(set(_cache_files()))
    for path in files:
        if dry_run:
            print(path)
        else:
            path.unlink(missing_ok=True)

    if not dry_run:
        _write_stamp(_shared_modules_digest())
    return len(files)


def clear_if_shared_modules_changed():
    """Clear the cache if SHARED_MODULES changed since the last clear (or there is no
    record of one); return the number of files removed, or None if nothing changed.
    """
    digest = _shared_modules_digest()
    try:
        if STAMP.read_text() == digest:
            return None
    except OSError:
        pass
    return clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="list what would be removed without deleting anything",
    )
    args = parser.parse_args(argv)

    removed = clear(dry_run=args.dry_run)
    if os.environ.get("NUMBA_CACHE_DIR"):
        print(f"WARNING: {EXTERNAL_CACHE_WARNING}")
    verb = "would remove" if args.dry_run else "removed"
    print(f"{verb} {removed} numba cache file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
