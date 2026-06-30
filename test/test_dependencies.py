"""
Purpose: tests related to dependencies
e.g.: hvplot is optional, so it shouldn't be imported by default.
Related issues: #1224, #1539
"""
import sys


def test_hvplot_optional():
    """Test that hvplot is actually optional and not imported by default.
    (hvplot in particular is "slow" to import (~1s to import hvplot.pandas),
    so it should not be imported until actually plotting something.)
    """
    assert "hvplot" not in sys.modules  # else, it's a test environment issue..?
    import uxarray as ux
    assert "hvplot" not in sys.modules


# TODO: similar tests for cartopy, holoviews, and other optional deps.
