"""
Purpose: test expected behaviors when installed without any optional dependencies.
Tests should all pass if and only if installed accordingly, i.e. something like:
    pip install "."
"""

import pytest
from _optional_deps_helpers import (
    check_requires_no_opts,
    check_requires_only_geo,
    check_requires_only_viz,
    check_requires_viz_and_geo,
)


def test_check_requires_no_opts():
    """ensure success for checks which should not require any optional dependencies"""
    check_requires_no_opts()


def test_check_requires_only_viz():
    """ensure failure for checks which should require viz optional dependencies"""
    with pytest.raises(ImportError, match=r'pip install "uxarray\[viz\]"'):
        check_requires_only_viz()


def test_check_requires_only_geo():
    """ensure failure for checks which should require geo optional dependencies"""
    with pytest.raises(ImportError, match=r'pip install "uxarray\[geo\]"'):
        check_requires_only_geo()


def test_check_requires_viz_and_geo():
    """ensure failure for checks which should require both viz and geo optional dependencies"""
    with pytest.raises(ImportError):
        # ^no match "uxarray[geo,viz]" here; might crash in a viz-only or a geo-only method,
        # even though the check itself ultimately requires both viz and geo.
        check_requires_viz_and_geo()
