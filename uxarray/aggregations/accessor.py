from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Tuple

import numpy as np
from xarray.core.types import Dims

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray


class DataArrayAggregateAccessor:
    def __init__(self, ux_obj: UxDataArray):
        self.ux_obj = ux_obj

    def __repr__(self) -> str:
        prefix = f"<{type(self.ux_obj).__name__}.remap>\n"
        return (
            prefix
            + "Supported methods:\n"
            + "  • nearest_neighbor(destination_grid, remap_to='faces')\n"
            + "  • inverse_distance_weighted(destination_grid, remap_to='faces', power=2, k=8)\n"
        )  # TODO

    def __call__(self):
        """Aggregates this data array ..."""
        pass

    def topological(
        self,
        func: Callable[..., Any],
        dim: Dims = None,
        *,
        destination: Literal["node", "edge", "face"],
        keep_attrs: bool | None = None,
        keepdims: bool = False,
        **kwargs: Any,
    ):
        """Aggregates this data array along its spatial dimension (and any other provided ones) by indexing
        connectivity arrays and applying one or more operations.

        """
        pass

    def azimuthal(
        self,
        func: Callable[..., Any] = np.mean,
        *,
        center_coord: Tuple[float, float],  # TODO: type
        outer_radius: int | float,
        radius_step: int | float,
        return_hit_counts: bool = False,
        keep_attrs: bool | None = None,
        keepdims: bool = False,
    ):
        """Aggregates this data array along its spatial dimension (and any other provided ones) by selecting elements
        that fall within circles of constant great-circle distance from a given point
        and applying one or more operations."""
        pass

    def neighborhood(self, func: Callable[..., Any], *, r: int | float):
        """Aggregates this data array along its spatial dimension (and any other provided ones) by collecting a
        neighborhood or elements and applying one or more operations."""
        pass
