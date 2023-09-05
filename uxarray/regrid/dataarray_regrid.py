from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray
    from uxarray.grid import Grid

from uxarray.regrid import nearest_neighbor


def _nearest_neighbor(source_uxda: UxDataArray,
                      destination_obj: Union[Grid, UxDataArray]):

    pass
