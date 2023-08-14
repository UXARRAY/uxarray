from uxarray.grid import Grid

from uxarray.constants import (INT_DTYPE, INT_FILL_VALUE)
from uxarray.grid import Grid

from uxarray.core.api import (open_grid, open_dataset, open_mfdataset)
from uxarray.core.dataarray import UxDataArray
from uxarray.core.dataset import UxDataset

# Sets the version of uxarray currently installeds
# Attempt to import the needed modules
try:
    from importlib.metadata import version as _version
except Exception:
    from importlib_metadata import version as _version

try:
    __version__ = _version("uxarray")
except Exception:
    # Placeholder version incase an error occurs, such as the library isn't installed
    __version__ = "999"
