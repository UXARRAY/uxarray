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
)
