import sys
from unittest import TestCase

import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    import uxarray
else:
    import uxarray


class test_placeholder(TestCase):

    def test_placeholder(self):
        pass
