import sys
from unittest import TestCase


# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    pass
else:
    pass


class test_placeholder(TestCase):
    def test_placeholder(self):
        pass
