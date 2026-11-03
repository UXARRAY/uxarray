"""
Purpose: testing tools from imports.py
"""

import os
import warnings

import pytest

from uxarray.utils.imports import (
    _optional_import_usage_throughout,
    _optional_imports_at_module_level,
)

HERE = __file__   # e.g. path0/uxarray/test/utils/test_imports
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(HERE), "..", "..", "uxarray"))
# e.g. SRC_ROOT = path0/uxarray/uxarray


def test_optional_dependency_imports_are_hinted():
    """Ensures that all optional dependencies are hinted for in the functions that use them.
    I.e., in a function which imports optional dependencies "dep1", "dep2", "dep3",
    need to call _raise_hint_if_optional_deps_missing("dep1", "dep2", "dep3").
    """
    results = _optional_import_usage_throughout(SRC_ROOT)

    missing = [r for r in results if r.missing_deps]
    extra = [r for r in results if r.extra_deps]

    for r in extra:
        warnings.warn(
            f"{r.filepath}:{r.lineno} in {r.qualname} — "
            "_raise_hint_if_optional_deps_missing() lists deps that "
            f"aren't actually imported here: {sorted(r.extra_deps)}",
            stacklevel=1,
        )

    if missing:
        details = "\n".join(
            f"  {r.filepath}:{r.lineno} in {r.qualname} — "
            f"failed to hint for these deps: {sorted(r.missing_deps)}"
            for r in missing
        )
        pytest.fail(
            "Function(s) import optional dependencies without including all of them in "
            f"_raise_hint_if_optional_deps_missing():\n{details}"
        )


def test_no_optional_dependency_imports_at_module_level():
    """Ensures that no optional dependencies are imported at the module level.
    I.e., all optional dependencies must be imported inside functions, not at the top of the file.
    """
    results = _optional_imports_at_module_level(SRC_ROOT)

    if results:
        results_str = "\n".join(
            f"  - {module} (line={lineno}) imports {optional_dep!r}"
            for module, lines in results.items()
            for lineno, optional_dep in lines
        )
        pytest.fail(
            f"Detected optional dependency imports at module level:\n{results_str}"
            "\n(Optional dependencies must only be imported inside functions, "
            "\nnever at module level, to ensure they remain optional.)"
        )
