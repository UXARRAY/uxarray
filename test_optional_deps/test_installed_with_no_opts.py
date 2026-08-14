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


def test_messages_of_raise_hint_if_optional_deps_missing():
    """additional tests for reasonable-looking messages from _raise_hint_if_optional_deps_missing.
    Hard-codes expected messages to prove it is working as expected in a variety of cases.
    Only including this test in the no_opts case because it covers all kinds of messages.
    (The "no error raised" case already gets covered by the other optional deps test files,
        so this test here is just about testing cases where an error is actually raised.)
    """
    import uxarray as ux
    import uxarray.utils.imports

    # first, check that expected error messages get raised:
    # (A) check that the stack is an OptionalDependencyNotFoundError on top of a ModuleNotFoundError.
    try:
        uxarray.utils.imports._raise_hint_if_optional_deps_missing("healpix")
    except ux.errors.OptionalDependencyNotFoundError as err:
        assert isinstance(err.__cause__, ModuleNotFoundError)

    # (B) check that the error is instead a ValueError if an unrecognized package name is provided.
    with pytest.raises(ValueError, match="Unrecognized package names"):
        uxarray.utils.imports._raise_hint_if_optional_deps_missing("_unrecognized_package_name_")

    # next, check error messages. Hard-code expected messages to make this test easier to read & maintain later.
    def _get_errmsg(*packages):
        try:
            uxarray.utils.imports._raise_hint_if_optional_deps_missing(*packages)
        except ux.errors.OptionalDependencyNotFoundError as err:
            errmsg = str(err)
        else:
            assert False, f"Expected OptionalDependencyNotFoundError, got no error. packages={packages}"
        return errmsg

    assert _get_errmsg("hvplot") == ('Failed to import: hvplot.'
        '\nConsider running ``pip install "uxarray[viz]"``, then try again.')
    assert _get_errmsg("holoviews", "geoviews") == ('Failed to import: geoviews, holoviews.'
        '\nConsider running ``pip install "uxarray[viz]"``, then try again.')
    assert _get_errmsg("healpix", "pyproj", "geopandas") == ('Failed to import: geopandas, healpix, pyproj.'
        '\nConsider running ``pip install "uxarray[geo]"``, then try again.')
    assert _get_errmsg("hvplot", "geopandas") == ('Failed to import: geopandas, hvplot.'
        '\nConsider running ``pip install "uxarray[geo,viz]"`` or ``pip install "uxarray[all]"``, then try again.')
    assert _get_errmsg("cartopy", "geopandas") == ('Failed to import: cartopy, geopandas.'
        '\nConsider running ``pip install "uxarray[geo]"``, then try again.')
    assert _get_errmsg("cartopy", "hvplot") == ('Failed to import: cartopy, hvplot.'
        '\nConsider running ``pip install "uxarray[viz]"``, then try again.')
    assert _get_errmsg("cartopy", "geopandas", "hvplot") == ('Failed to import: cartopy, geopandas, hvplot.'
        '\nConsider running ``pip install "uxarray[geo,viz]"`` or ``pip install "uxarray[all]"``, then try again.')

    # to fully test the _raise_hint_if_optional_deps_missing() function,
    # need to check cases with more than 2 extras. Add corresponding "fake packages" here.
    uxarray.utils.imports._OPTIONAL_DEPS_TO_EXTRAS.update({
        "_fakepackage1_": "_fakeextra1_",
        "_fakepackage2_": ("_fakeextra1_", "_fakeextra2_"),
        "_fakepackage3_": ("_fakeextra1_", "_fakeextra2_", "_fakeextra3_"),
    })

    assert  _get_errmsg("matplotlib", "spatialpandas", "_fakepackage1_") == (
        'Failed to import: _fakepackage1_, matplotlib, spatialpandas.\n'
        'Consider running ``pip install "uxarray[_fakeextra1_,geo,viz]"`` or '
        '``pip install "uxarray[all]"``, then try again.')
    assert _get_errmsg("_fakepackage1_", "_fakepackage2_") == ('Failed to import: _fakepackage1_, _fakepackage2_.'
        '\nConsider running ``pip install "uxarray[_fakeextra1_]"``, then try again.')
    assert _get_errmsg("_fakepackage3_", "_fakepackage2_") == ('Failed to import: _fakepackage2_, _fakepackage3_.'
        '\nConsider running ``pip install "uxarray[all]"``, then try again.')
    assert _get_errmsg("_fakepackage3_") == ('Failed to import: _fakepackage3_.\n'
        'Consider running ``pip install "uxarray[_fakeextra1_]"`` or pip install with '
        '[_fakeextra2_], [_fakeextra3_], or [all], then try again.')
