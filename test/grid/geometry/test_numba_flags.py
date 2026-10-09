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
        if isinstance(func, ast.Name):
            target = namespace.get(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            target = getattr(namespace.get(func.value.id), func.attr, None)
        else:
            continue
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


def test_numpy_error_model(kernels):
    offenders = sorted(
        name
        for name, (kernel, compiled) in kernels.items()
        if compiled & NUMPY_ERROR_MODEL_KERNELS
        and kernel.targetoptions.get("error_model") != "numpy"
    )
    assert not offenders, f"need error_model='numpy' to mask inf/nan: {offenders}"


def test_no_fastmath(kernels):
    eft = {
        name: kernel
        for name, (kernel, compiled) in kernels.items()
        if any(n.startswith(EFT_MODULE + ".") for n in compiled)
    }
    offenders = [
        name for name, kernel in eft.items() if kernel.targetoptions.get("fastmath")
    ]
    # Also scan the source, for kernels the import walk can't see: nested ones and the
    # ``two_prod`` variant not selected on this machine.
    for module in {kernel.py_func.__module__ for kernel in eft.values()}:
        tree = ast.parse(Path(sys.modules[module].__file__).read_text())
        offenders += [
            f"{module}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.keyword) and node.arg == "fastmath"
        ]
    assert not offenders, (
        f"fastmath would cancel the EFT error terms: {sorted(offenders)}"
    )
