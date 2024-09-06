# Sets the version of uxarray currently installeds
# Attempt to import the needed modules

import uxarray.constants

from .core.api import open_grid, open_dataset, open_mfdataset

from .core.dataset import UxDataset
from .core.dataarray import UxDataArray
from .grid import Grid

from .constants import INT_DTYPE, INT_FILL_VALUE
from .cmaps import sequential, diverging, sequential_blue, sequential_green

try:
    from importlib.metadata import version as _version
except Exception:
    from importlib_metadata import version as _version

try:
    __version__ = _version("uxarray")
except Exception:
    # Placeholder version incase an error occurs, such as the library isn't installed
    __version__ = "999"

# Flag for enabling FMA instructions across the package


def enable_fma():
    """Enables Fused-Multiply-Add (FMA) instructions using the ``pyfma``
    package."""
    uxarray.constants.ENABLE_FMA = True


def disable_fma():
    """Disable Fused-Multiply-Add (FMA) instructions using the ``pyfma``
    package."""
    uxarray.constants.ENABLE_FMA = False


__all__ = (
    "open_grid",
    "open_dataset",
    "open_mfdataset",
    "UxDataset",
    "UxDataArray",
    "INT_DTYPE",
    "INT_FILL_VALUE",
    "Grid",
    "sequential",
    "diverging",
    "sequential_blue",
    "sequential_green",
    "enable_fma",
    "disable_fma",
)
