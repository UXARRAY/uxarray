"""Guard the numba flags that uxarray's compensated (EFT) kernels depend on.

A kernel declared ``inline="always"`` is compiled with its *caller's* flags, not its own,
so every kernel it is inlined into, directly or transitively, needs the same flags.
Taking ``acc_sqrt_re`` (``uxarray.utils.computing``) as the example:

- ``error_model="numpy"``: division follows IEEE 754, so ``acc_sqrt_re(0.0)`` returns
  ``(0.0, nan)`` (the correction is 0/0) and the status layer's isfinite mask rejects it.
  Under numba's default ``error_model="python"`` the same call raises ZeroDivisionError.
  (A negative input gives ``(nan, nan)`` under either model; only division raises.)
- no ``fastmath``: compensated arithmetic relies on IEEE 754 forbidding reassociation.
  ``fastmath`` lets LLVM simplify two_sum's ``e = (a - (s - bp)) + (b - bp)``, with
  ``s = a + b``, to 0, discarding the rounding error the term exists to recover:
  ``acc_sqrt_re(10.0)`` returns correction -1.9e-16 normally, 0.0 under fastmath.

``test_numpy_error_model`` checks every uxarray kernel that inlines one of
``NUMPY_ERROR_MODEL_KERNELS``; ``test_no_fastmath`` checks every kernel that inlines
anything from ``uxarray.utils.computing``. ``test_shared_modules_complete`` checks that
the cache clearing in test/conftest.py watches every module whose kernels are compiled
into another module's.
"""

import ast
import importlib
import importlib.util
import inspect
import pkgutil
import sys
import textwrap
from pathlib import Path

import pytest
from numba.core.registry import CPUDispatcher

import uxarray

EFT_MODULE = "uxarray.utils.computing"
# The kernels whose own arithmetic relies on 1/0 -> inf and 0/0 -> nan. This is the
# list of *sources*: the tests scan every kernel in uxarray and flag any one these are
# inlined into that lacks error_model="numpy". It has to be maintained by hand, because
# a kernel's own flags can't tell us this: e.g. ``_try_gca_gca_intersection`` sets
# error_model="numpy" only because it inlines ``_accux_gca``, not because it needs it.
NUMPY_ERROR_MODEL_KERNELS = {
    "uxarray.utils.computing.acc_sqrt_re",
    "uxarray.grid.intersections._accux_gca",
    "uxarray.grid.intersections._accux_constlat",
}


def _qualname(kernel):
    return f"{kernel.py_func.__module__}.{kernel.py_func.__name__}"


def _callees(kernel):
    """Yield the jitted functions called by name in ``kernel``'s body."""
    namespace = kernel.py_func.__globals__
    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(kernel.py_func)))):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        # A direct function call (e.g., print())
        if isinstance(func, ast.Name):
            target = namespace.get(func.id)
        # An attribute method call (e.g. val.print())
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            target = getattr(namespace.get(func.value.id), func.attr, None)
        else:
            continue

        # Only consider Numba JIT func calls
        if isinstance(target, CPUDispatcher):
            yield target


def _get_inlined_funcs_in_call_tree(kernel):
    """Names of ``kernel`` and every kernel inlined into it, all built with its flags."""
    names, stack = {_qualname(kernel)}, [kernel]
    while stack:
        for callee in _callees(stack.pop()):
            name = _qualname(callee)
            if callee.targetoptions.get("inline") == "always" and name not in names:
                names.add(name)
                stack.append(callee)
    return names


@pytest.fixture(scope="module")
def kernels():
    """Map each module-level kernel in uxarray to ``(kernel, names compiled into it)``."""
    found = {}
    for info in pkgutil.walk_packages(uxarray.__path__, "uxarray."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # missing optional dependency: its kernels can't run either
            continue
        for obj in vars(module).values():
            if isinstance(obj, CPUDispatcher) and obj.py_func.__module__ == info.name:
                found[_qualname(obj)] = (obj, _get_inlined_funcs_in_call_tree(obj))
    return found


def test_numpy_error_model(kernels):
    """Every kernel that inlines a NUMPY_ERROR_MODEL_KERNELS entry sets error_model="numpy"."""
    # If a source kernel is renamed, moved or fails to import, nothing would match it
    # below and this test would pass without checking anything.
    missing = NUMPY_ERROR_MODEL_KERNELS - kernels.keys()
    assert not missing, f"NUMPY_ERROR_MODEL_KERNELS names unknown kernels: {missing}"

    offenders = sorted(
        name
        for name, (kernel, is_compiled) in kernels.items()
        if is_compiled & NUMPY_ERROR_MODEL_KERNELS  # inlines a kernel that needs it...
        and kernel.targetoptions.get("error_model") != "numpy"  # ...but doesn't set it
    )
    assert not offenders, (
        f"Missing error_model='numpy' on {offenders}: they inline kernels that rely on "
        f"1/0 -> inf and 0/0 -> nan, which raise ZeroDivisionError under the default model"
    )


def test_no_fastmath(kernels):
    """No kernel that inlines EFT code from uxarray.utils.computing enables fastmath."""
    eft = {
        name: kernel
        for name, (kernel, compiled) in kernels.items()
        if any(n.startswith(EFT_MODULE + ".") for n in compiled)
    }
    assert eft, f"No kernels found in {EFT_MODULE}; was it renamed?"  # else vacuous
    offenders = [
        name for name, kernel in eft.items() if kernel.targetoptions.get("fastmath")
    ]
    # Also scan the source, for kernels the import walk can't see: nested ones and the
    # ``two_prod`` variant not selected on this machine. Only a literal ``False`` is known
    # to be safe; anything else (``True``, a flag set, a variable) is flagged.
    for module in {kernel.py_func.__module__ for kernel in eft.values()}:
        tree = ast.parse(Path(sys.modules[module].__file__).read_text())
        offenders += [
            f"{module}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.keyword)
            and node.arg == "fastmath"
            and not (isinstance(node.value, ast.Constant) and node.value.value is False)
        ]
    assert not offenders, (
        f"fastmath enabled on {sorted(offenders)} (kernels, or module:line of a fastmath= "
        f"that isn't a literal False): they inline EFT code from {EFT_MODULE}, whose "
        f"error terms fastmath cancels to 0. Remove fastmath or set it to False."
    )


def test_shared_modules_complete(kernels):
    """SHARED_MODULES lists every module whose kernels another module's kernels call."""
    script = Path(__file__).parents[3] / "ci" / "clean_numba_cache.py"
    if not script.exists():
        pytest.skip("ci/clean_numba_cache.py is not available outside the repository")
    spec = importlib.util.spec_from_file_location("clean_numba_cache", script)
    cleaner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cleaner)
    shared = {
        "uxarray." + name.removesuffix(".py").replace("/", ".")
        for name in cleaner.SHARED_MODULES
    }

    # Any cross-module call counts, not only inline="always": numba links a cached
    # callee's code into the caller's cached object either way.
    called_from = {}
    for kernel, _ in kernels.values():
        caller = kernel.py_func.__module__
        for callee in _callees(kernel):
            if callee.py_func.__module__ != caller:
                called_from.setdefault(callee.py_func.__module__, set()).add(caller)
    assert called_from, "found no cross-module kernel calls; is _callees broken?"

    missing = {m: sorted(c) for m, c in sorted(called_from.items()) if m not in shared}
    assert not missing, (
        f"Add these to SHARED_MODULES in ci/clean_numba_cache.py: {missing} "
        f"(module -> its callers). Kernels in the callers compile in code from these "
        f"modules, and numba won't notice when they change, so pytest must clear the "
        f"cache whenever they do."
    )
