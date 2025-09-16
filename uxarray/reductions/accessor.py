from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Tuple

import numpy as np
from xarray.core.types import Dims

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray


class DataArrayReduceAccessor:
    def __init__(self, ux_obj: UxDataArray):
        self.ux_obj = ux_obj

    def __repr__(self) -> str:
        prefix = f"<{type(self.ux_obj).__name__}.remap>\n"
        return (
            prefix
            + "Supported methods:\n"
            + "  • nearest_neighbor(destination_grid, remap_to='faces')\n"
            + "  • inverse_distance_weighted(destination_grid, remap_to='faces', power=2, k=8)\n"
        )

    def __call__(
        self,
        func: Callable[..., Any],
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        keep_attrs: bool | None = None,
        keepdims: bool = False,
        **kwargs: Any,
    ):
        """TODO: Use Xarray's default docstring"""

        return super(UxDataArray, self.ux_obj).reduce(
            func=func, dim=dim, keep_attrs=keep_attrs, keepdims=keepdims, **kwargs
        )

    def topological(
        self,
        func: Callable[..., Any],
        destination: Literal["node", "edge", "face"],
        **kwargs: Any,
    ):
        pass

    def azimuthal(
        self,
        func: Callable[..., Any] = np.mean,
        *,
        center_coord: Tuple[float, float],  # TODO: type
        outer_radius: int | float,
        radius_step: int | float,
        return_hit_counts: bool = False,
    ):  #
        pass

    def neighborhood(self, func: Callable[..., Any], *, r: int | float):
        pass
