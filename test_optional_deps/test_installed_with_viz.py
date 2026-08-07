"""
Purpose: test expected behaviors when installed with only viz optional dependency.
Tests should all pass if and only if installed accordingly, i.e. something like:
    pip install ".[viz]"
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
    """ensure success for checks which should require viz optional dependencies"""
    check_requires_only_viz()


def test_check_requires_only_geo():
    """ensure failure for checks which should require geo optional dependencies"""
    with pytest.raises(ImportError, match=r"pip install uxarray\[geo\]"):
        check_requires_only_geo()


def test_check_requires_viz_and_geo():
    """ensure failure for checks which should require both viz and geo optional dependencies"""
    with pytest.raises(ImportError):
        check_requires_viz_and_geo()
