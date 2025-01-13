import sys
import xarray as xr

# Import from directory structure if coverage test, or from installed packages otherwise
if "--cov" in str(sys.argv):
    import uxarray
else:
    import uxarray

def test_placeholder():
    """Placeholder test that currently does nothing."""
    pass
