"""Guard the numba flags that the compensated (EFT) kernels depend on.

``fastmath`` lets LLVM reassociate ``e = (a - (s - bp)) + (b - bp)`` to 0, silently undoing
every compensated result from ``uxarray.utils.computing``. ``acc_sqrt_re``, ``_accux_gca``
and ``_accux_constlat`` need ``error_model="numpy"`` so that 1/0 and sqrt(<0) give inf/nan
for the status layer to mask, instead of raising.

An ``inline="always"`` callee is compiled with its caller's flags, so both checks apply to
every kernel this code is inlined into, not just where it is defined.
"""

import ast
import importlib
import inspect
import pkgutil
import sys
import textwrap
from pathlib import Path

import pytest
from numba.core.registry import CPUDispatcher

import uxarray

EFT_MODULE = "uxarray.utils.computing"
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


def _compiled_into(kernel):
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
                found[_qualname(obj)] = (obj, _compiled_into(obj))
    assert NUMPY_ERROR_MODEL_KERNELS <= found.keys()  # else the checks pass vacuously
    return found

# For each inlined caller kernel, look at the included/invoked callee subkernels and
# filter out caller kernels that don't use numpy error model where at least one of
# the callee subkernels uses the numpy error model. i.e., instances of
# InlinedCallerKernelNotUsingNumpyErrorModel where CalleeSubKernelUsesNumpyErrorModel
def test_numpy_error_model(kernels):
    """Test that functions use error_model==numpy if it calls inline
    functions with error_model == numpy.
    """

    offenders = sorted(
        name
        for name, (kernel, is_compiled) in kernels.items()
        if is_compiled & NUMPY_ERROR_MODEL_KERNELS # Compiled kernel is using numpy error model
        and kernel.targetoptions.get("error_model") != "numpy" # ...but the current kernel is *not*!
    )
    assert not offenders, (f"Missing error_model='numpy' in function decorations of : {offenders}. "
                           f"Error model numpy is needed for some kernels used in these functions "
                           f"(to avoid exceptions on inf/nan)")


def test_no_fastmath(kernels):
    """Test that functions don't use fastmath if they call inline functions"""
    eft = {
        name: kernel
        for name, (kernel, compiled) in kernels.items()
        if any(n.startswith(EFT_MODULE + ".") for n in compiled)
    }
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
        f"fastmath would cancel the EFT error terms: {sorted(offenders)}"
    )
