"""Delete numba's on-disk cache for ``uxarray``.

Numba stamps each ``cache=True`` index (``*.nbi``) with the ``(mtime, size)`` of the
source file that *defines* the function, and nothing else. A kernel that calls a jitted
function from another module has that callee's code linked into its own cached object,
so editing the callee leaves the caller's cache entry "valid" with the old body baked in.

Run this after editing any shared jitted module, or let the pre-commit hook do it::

    python ci/clean_numba_cache.py [--dry-run]
"""

import argparse
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "uxarray"
CACHE_SUFFIXES = {".nbi", ".nbc"}


def _cache_files():
    """Yield numba cache files stored next to the package's sources."""
    for path in PACKAGE_DIR.rglob("__pycache__/*"):
        if path.suffix in CACHE_SUFFIXES:
            yield path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="list what would be removed without deleting anything",
    )
    args = parser.parse_args(argv)

    files = sorted(set(_cache_files()))
    for path in files:
        if args.dry_run:
            print(path)
        else:
            path.unlink(missing_ok=True)

    if os.environ.get("NUMBA_CACHE_DIR"):
        # Numba files caches there under a hash of each source directory, so they cannot
        # be told apart from other projects' entries; leave them alone rather than guess.
        print("NUMBA_CACHE_DIR is set: clear it by hand, it is not touched here")

    verb = "would remove" if args.dry_run else "removed"
    print(f"{verb} {len(files)} numba cache file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
