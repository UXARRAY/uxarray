"""Policy guard for ``nogil=True`` on the package's numba kernels.

``nogil`` is released once, at the Python -> native boundary, and stays
released for the whole native call tree beneath it. Two consequences shape
where the flag belongs, and both are easy to get wrong by eye:

* On a kernel that only other jitted code calls, the flag is inert -- an
  njit -> njit call never touches the GIL, so an entry point's ``nogil``
  already covers everything it calls. Setting it anyway is not free: the
  flag adds a ``PyEval_SaveThread``/``RestoreThread`` pair to the Python
  wrapper (measured at ~40ns/call), which is invisible in normal use but
  shows up as a 1.2-1.4x regression the moment a micro-benchmark calls the
  kernel directly. A blanket sweep did exactly that and asv caught it.

* On a kernel Python calls per *element* -- inside a Python ``for`` loop --
  that same ~40ns is a real cost against a body measured in hundreds of
  nanoseconds, and buys nothing, because a serial Python loop has no second
  thread to overlap with.

So the flag belongs on exactly one population: kernels Python invokes once
per whole array, which is the set dask calls once per chunk under
``dask="parallelized"``. The tests below pin that in both directions, so
neither a missing flag (silent serialization under the threaded scheduler)
nor a stray one (a benchmark regression with no upside) can land quietly.
"""

import ast
import pathlib
from collections import namedtuple

import pytest

import uxarray

PKG = pathlib.Path(uxarray.__file__).parent

#: Decorators that compile a function to native code.
JIT_DECORATORS = frozenset({"njit", "guvectorize", "vectorize"})

#: ``guvectorize``/``vectorize`` do not accept ``nogil`` at all -- numba
#: raises ``KeyError: Unrecognized options`` at import. The gufunc machinery
#: already drops the GIL around the loop, so there is nothing to ask for.
GUFUNC_DECORATORS = frozenset({"guvectorize", "vectorize"})

#: Kernels Python reaches but which must *not* take ``nogil``, with the
#: reason. Anything added here needs a per-call justification, not a
#: preference. Stale entries are themselves a failure (see the last test):
#: when one of these stops being called per element -- most are slated to
#: become gufuncs -- drop it from the list and let the policy apply.
EXEMPT = {
    "gca_const_lat_intersection": (
        "per-edge, from the Python loops in zonal.py and integrate.py; "
        "slated to become a gufunc"
    ),
    "get_number_of_intersections": (
        "per-edge, from the same loops as gca_const_lat_intersection"
    ),
    "_compute_band_overlap_area": "per-face, from the Python loop at zonal.py:327",
    "_barycentric_coordinates": (
        "per-candidate-face, from the nested Python loop at neighbors.py:954"
    ),
    "pole_point_inside_polygon": (
        "reached njit -> njit from _construct_face_bounds, which already "
        "releases the GIL; the Python-level reference is a thin test wrapper"
    ),
    "_variance": "a reducer that _make_kernel compiles into the gufunc, not an entry point",
    "_median": "a reducer that _make_kernel compiles into the gufunc, not an entry point",
}

Kernel = namedtuple("Kernel", "name path lineno decorator nogil inline_always")


def _decorator_of(node):
    """The jit decorator on ``node``, or ``None``."""
    for dec in node.decorator_list:
        func = dec.func if isinstance(dec, ast.Call) else dec
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in JIT_DECORATORS:
            return name, dec
    return None


def _keyword(dec, arg):
    """Value of keyword ``arg`` on a decorator call, or ``None``."""
    if not isinstance(dec, ast.Call):
        return None
    for kw in dec.keywords:
        if kw.arg == arg and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


def _scan():
    """Every jitted kernel in the package, and the names Python scope reaches.

    "Reaches" covers a bare reference as well as a call: the kernels that
    matter most are *passed* to ``xr.apply_ufunc`` rather than called, e.g.
    ``_build_n_nodes_per_face`` at connectivity.py:136.
    """
    kernels, trees = [], {}
    for path in sorted(PKG.rglob("*.py")):
        # Explicit encoding: the default is the locale's, which on Windows is
        # a codepage that cannot decode the U+2010 in (e.g.) core/dataarray.py.
        tree = ast.parse(path.read_text(encoding="utf-8"))
        trees[path] = tree
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            found = _decorator_of(node)
            if found is None:
                continue
            name, dec = found
            kernels.append(
                Kernel(
                    name=node.name,
                    path=path.relative_to(PKG.parent),
                    lineno=dec.lineno if not isinstance(dec, ast.Call) else dec.func.lineno,
                    decorator=name,
                    nogil=bool(_keyword(dec, "nogil")),
                    inline_always=_keyword(dec, "inline") == "always",
                )
            )

    jit_names = {k.name for k in kernels}
    reachable = set()

    def walk(node, in_jit, path):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, in_jit or _decorator_of(child) is not None, path)
                continue
            if not in_jit:
                if isinstance(child, ast.Call):
                    func = child.func
                    nm = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                    if nm in jit_names:
                        reachable.add(nm)
                elif (
                    isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                    and child.id in jit_names
                ):
                    reachable.add(child.id)
            walk(child, in_jit, path)

    for path, tree in trees.items():
        walk(tree, False, path)

    return kernels, reachable


KERNELS, REACHABLE = _scan()


def _where(k):
    return f"{k.path}:{k.lineno} {k.name}"


def test_kernels_were_found():
    """Guard the guard: a broken scan must not silently pass everything."""
    assert len(KERNELS) > 100, "AST scan found almost no kernels -- the scan is broken"
    assert any(k.nogil for k in KERNELS), "scan found no nogil kernels at all"


def test_python_entry_points_release_the_gil():
    """A kernel Python invokes per array must release the GIL.

    Without it, ``dask="parallelized"`` on the threaded scheduler serializes
    the kernel across workers -- silently, with no error and no effect on any
    single-threaded benchmark.
    """
    missing = [
        k
        for k in KERNELS
        if k.name in REACHABLE
        and k.decorator == "njit"
        and not k.nogil
        and not k.inline_always
        and k.name not in EXEMPT
    ]
    assert not missing, (
        "These kernels are reachable from Python scope but hold the GIL:\n  "
        + "\n  ".join(_where(k) for k in missing)
        + "\n\nAdd nogil=True. If the kernel is instead called per element from a "
        "Python loop, add it to EXEMPT in this file with the reason."
    )


def test_jit_only_kernels_do_not_set_nogil():
    """A kernel only jitted code calls must not set ``nogil``.

    Its caller has already released the GIL, so the flag adds nothing but the
    ~40ns wrapper cost paid by any direct Python caller -- which in practice
    means the asv micro-benchmarks, where it reads as a regression.
    """
    stray = [k for k in KERNELS if k.nogil and k.name not in REACHABLE]
    assert not stray, (
        "These kernels set nogil but nothing in Python scope calls them:\n  "
        + "\n  ".join(_where(k) for k in stray)
        + "\n\nDrop nogil=True -- the entry point that calls them already "
        "released the GIL for the whole call tree."
    )


def test_inlined_kernels_do_not_set_nogil():
    """``inline="always"`` kernels have no call boundary to release at.

    They are inlined into their jitted callers by construction, so the flag
    only ever materializes as overhead on a direct Python call. The EFT
    primitives in utils/computing.py must stay this way: a real call boundary
    would spill their arguments (see the refactor plan, "where the guvectorize
    seam belongs").
    """
    stray = [k for k in KERNELS if k.inline_always and k.nogil]
    assert not stray, (
        "inline=always kernels must not set nogil:\n  "
        + "\n  ".join(_where(k) for k in stray)
    )


def test_gufuncs_do_not_set_nogil():
    """numba rejects ``nogil`` on ``guvectorize``/``vectorize``.

    The gufunc machinery already drops the GIL around the loop, and passing
    the option anyway raises ``KeyError: Unrecognized options: {'nogil'}``
    when the kernel compiles.

    This static check catches the literal ``nogil=`` keyword. It cannot see
    through the ``**_GUFUNC_KWARGS`` form used at neighbors.py:1231 -- but
    that path needs no guard, because numba raises there on first compile and
    any neighborhood test fails loudly.
    """
    stray = [k for k in KERNELS if k.decorator in GUFUNC_DECORATORS and k.nogil]
    assert not stray, (
        "gufuncs cannot take nogil (numba raises at import); the ufunc loop "
        "already releases the GIL:\n  " + "\n  ".join(_where(k) for k in stray)
    )


@pytest.mark.parametrize("name", sorted(EXEMPT))
def test_exemptions_are_still_live(name):
    """Keep EXEMPT from rotting.

    An exemption is only meaningful while the kernel is still reachable from
    Python and still without the flag. Once one becomes a gufunc or stops
    being called per element, its entry here is dead and should go.
    """
    sites = [k for k in KERNELS if k.name == name]
    assert sites, f"EXEMPT lists {name!r}, which no longer exists as a kernel"
    assert name in REACHABLE, (
        f"EXEMPT lists {name!r}, but nothing in Python scope reaches it any "
        "more -- the exemption is dead, remove it"
    )
    carrying = [k for k in sites if k.nogil]
    assert not carrying, (
        f"{name!r} is listed in EXEMPT but sets nogil:\n  "
        + "\n  ".join(_where(k) for k in carrying)
        + "\nRemove it from EXEMPT, or drop the flag."
    )
