from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.grid import Grid


class GridCrossSectionAccessor:
    """Accessor for cross-section operations on a ``Grid``"""

    def __init__(self, uxgrid: Grid) -> None:
        self.uxgrid = uxgrid

    def constant_latitude(self, *args, **kwargs):
        warnings.warn(
            "The ‘.cross_section.constant_latitude’ method is deprecated and will be removed in a future release; "
            "please use the `.subset.constant_latitude` accessor instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.uxgrid.subset.constant_latitude(*args, **kwargs)

    def constant_longitude(self, *args, **kwargs):
        warnings.warn(
            "The ‘.cross_section.constant_longitude’ method is deprecated and will be removed in a future release; "
            "please use the `.subset.constant_longitude` accessor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.uxgrid.subset.constant_longitude(*args, **kwargs)

    def constant_latitude_interval(self, *args, **kwargs):
        warnings.warn(
            "The ‘.cross_section.constant_latitude_interval’ method is deprecated and will be removed in a future release; "
            "please use the `.subset.constant_latitude_interval` accessor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.uxgrid.subset.constant_latitude_interval(*args, **kwargs)

    def constant_longitude_interval(self, *args, **kwargs):
        warnings.warn(
            "The ‘.cross_section.constant_longitude_interval’ method is deprecated and will be removed in a future release; "
            "please use the `.subset.constant_longitude_interval` accessor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.uxgrid.subset.constant_longitude_interval(*args, **kwargs)
