"""
Purpose: tests related to dependencies
e.g.: hvplot is optional, so it shouldn't be imported by default.
Related issues: #1224, #1539
"""
import subprocess
import sys


def _assert_not_imported_after_import_uxarray(module_name):
    """Run ``import uxarray`` in a fresh interpreter and assert that
    ``module_name`` was not imported as a side effect.

    A subprocess is used so the check is immune to test ordering: other tests
    in the same session may have already imported optional deps (e.g. plotting
    tests import ``hvplot``), which would pollute this process's ``sys.modules``.
    """
    code = (
        "import sys; import uxarray; "
        f"assert {module_name!r} not in sys.modules, "
        f"'{module_name} was imported by `import uxarray`'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{module_name} should not be imported by default:\n{result.stderr}"
    )


def test_hvplot_optional():
    """Test that hvplot is actually optional and not imported by default.
    (hvplot in particular is "slow" to import (~1s to import hvplot.pandas),
    so it should not be imported until actually plotting something.)
    """
    _assert_not_imported_after_import_uxarray("hvplot")


def test_no_numba_kernels_built_on_import():
    """Test that `import uxarray` does not build any numba kernel.

    ``guvectorize`` compiles at decoration time when it is given explicit
    signatures, so a kernel assigned at module scope is built during the
    import. This compilation can dominate the uxarray import, and building a
    ``target="parallel"`` kernel starts numba's threading layer, which
    leaves a thread pool running, making forks unsafe.
    """
    code = (
        "import numba, uxarray\n"
        "try:\n"
        "    layer = numba.threading_layer()\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError(\n"
        "        f'`import uxarray` started numba threading layer {layer!r}. '\n"
        "        'Something it imports builds a parallel kernel at module '\n"
        "        'scope; build it on first use instead.'\n"
        "    )\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_py_typed_marker_is_installed(tmp_path):
    """Test that the PEP 561 ``py.typed`` marker ships with the package.

    uxarray annotates its public API, but a type checker is only allowed to
    use those annotations if the package declares itself typed by shipping an
    (empty) ``py.typed`` file. Without it, mypy and pyright silently treat
    every uxarray import as ``Any``, and ``--strict`` users get an error on
    ``import uxarray``.

    This asserts against the *installed* package rather than the source tree,
    since only the installed package can catch a packaging regression.

    The import is done in a subprocess run from a directory other than the
    repository root. ``python -m pytest`` prepends the current directory to
    ``sys.path``, so running it from the root would import ``uxarray/`` from
    the source tree, where the marker is always present, and the test would
    pass even if the build config had dropped it.
    """
    code = (
        "import pathlib, uxarray; "
        "p = pathlib.Path(uxarray.__file__).parent / 'py.typed'; "
        "print(p); "
        "raise SystemExit(0 if p.is_file() else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, (
        f"PEP 561 marker missing from the installed package at "
        f"{result.stdout.strip()}. Type checkers will ignore uxarray's "
        "annotations. See [tool.setuptools.package-data] in pyproject.toml "
        f"and MANIFEST.in.\n{result.stderr}"
    )


# TODO: similar tests for cartopy, holoviews, and other optional deps.
