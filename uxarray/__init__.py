from .constants import INT_DTYPE, INT_FILL_VALUE
from .core.api import (
    concat,
    list_grid_names,
    open_dataset,
    open_grid,
    open_mfdataset,
    open_multigrid,
)
from .core.dataarray import UxDataArray
from .core.dataset import UxDataset
from .grid import Grid

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
    "open_multigrid",
    "list_grid_names",
    "concat",
    "UxDataset",
    "UxDataArray",
    "INT_DTYPE",
    "INT_FILL_VALUE",
    "Grid",
)
