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


# TODO: similar tests for cartopy, holoviews, and other optional deps.
