"""Deterministic memory-size accounting for the benchmark suite.

This module backs the ``track_nbytes_*`` benchmarks that replaced the former
``peakmem_*`` ones.

Why ``peakmem_*`` was removed
-----------------------------
asv's ``peakmem_*`` records the maximum resident set size of the *whole
process*, and per asv's own docs it "also counts memory usage during the setup
routine". For uxarray that number is almost entirely fixed overhead:

    grid                    import  +open_grid  +.bounds   attributable to op
    quad-hexagon  ( 24K)     236MB      265MB     301MB     36MB (12%)
    geoflow-small (1.1M)     236MB      263MB     487MB    224MB (46%)
    outCSne8      ( 48K)     235MB      281MB     317MB     35MB (11%)
    oQU480        (4.6M)     236MB      270MB     306MB     36MB (12%)

``import uxarray`` alone is ~226 MB and constant to within 1 MB. oQU480 is 190x
larger than quad-hexagon yet both attributed ~36 MB to ``Grid.bounds`` -- the
benchmark was nearly insensitive to its own workload. The part that did vary was
numba: with a cold JIT cache the quad-hexagon case peaked at 599 MB, with a warm
one 298 MB. Which side of an ``asv continuous`` comparison paid the compile cost
depended on run order, so unrelated PRs kept reporting identical ~0.55-0.77
"improvements".

Why not sampled peak RSS
------------------------
Sampling current RSS around the call and reporting the delta is immune to
process history, and it validates cleanly (a known 200 MB allocation measures as
200.0 MB). But after a warm-up call -- which is required to keep JIT compilation
out of the measurement -- the allocator reuses the pages it just freed, so RSS
*growth* systematically undercounts. Every operation the old benchmarks covered
measured 0.0-0.1 MB that way, including on the 98 MB / 28,571-face oQU120 mesh.

Why not tracemalloc
-------------------
It measures allocation volume rather than RSS growth, which would sidestep the
page-reuse problem, but it does not observe numba NRT allocations -- and that is
where uxarray's array memory is allocated.

What is left
------------
Deterministic size accounting. It is bit-reproducible across runs, platforms and
JIT states (verified), scales with the mesh, and catches the memory regressions
that actually matter in an array library: dtype widening, densified
connectivity, and newly cached arrays. It does not capture transient peaks --
but no available instrument captures those reliably here, and the measurements
above show the transients are ~1 MB, far below anything worth gating on.
"""

__all__ = ["grid_nbytes", "dataset_nbytes"]


def grid_nbytes(uxgrid):
    """Total size of the arrays a ``Grid`` currently holds, in bytes.

    Counts whatever has been materialized so far, so calling this after an
    operation that caches results onto the grid (``bounds``, ``face_areas``,
    connectivity) reports that operation's contribution to the grid's footprint.
    """
    return uxgrid._ds.nbytes


def dataset_nbytes(uxds):
    """Total size of a ``UxDataset``, in bytes, including its grid."""
    return uxds.nbytes + grid_nbytes(uxds.uxgrid)
