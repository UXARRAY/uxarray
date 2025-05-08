from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray
    from uxarray.core.dataset import UxDataset
    from uxarray.grid.grid import Grid

from uxarray.remap.inverse_distance_weighted import _inverse_distance_weighted_remap
from uxarray.remap.nearest_neighbor import _nearest_neighbor_remap


class RemapAccessor:
    def __init__(self, ux_obj: UxDataArray | UxDataset):
        self.ux_obj = ux_obj

    def __repr__(self) -> str:
        prefix = f"<{type(self.ux_obj)}.remap>\n"
        methods_heading = "Supported Methods:\n"
        methods_heading += "  * nearest_neighbor(destination_obj, remap_to)\n"
        methods_heading += (
            "  * inverse_distance_weighted(destination_obj, remap_to, power, k)\n"
        )

        return prefix + methods_heading

    def __call__(self, *args, **kwargs) -> UxDataArray | UxDataset:
        """Default Remapping (Nearest Neighbor)"""
        return self.nearest_neighbor(*args, **kwargs)

    def nearest_neighbor(
        self, destination_grid: Grid, remap_to: str = "faces", **kwargs
    ) -> UxDataArray | UxDataset:
        """Nearest Neighbor Remapping.

        Parameters
        ---------
        destination_grid : Grid
            Destination Grid for remapping
        remap_to : str, default="nodes"
            Location of where to map data, either "nodes", "edges" or "faces"

        Returns
        -------
        remapped: UxDataArray or UxDataset
            TODO

        """

        return _nearest_neighbor_remap(self.ux_obj, destination_grid, remap_to)

    def inverse_distance_weighted(
        self, destination_grid: Grid, remap_to: str = "faces", power=2, k=8, **kwargs
    ) -> UxDataArray | UxDataset:
        """Inverse Distance Weighted Remapping.

        Parameters
        ---------
        destination_grid : Grid
            Destination Grid for remapping
        remap_to : str, default="nodes"
            Location of where to map data, either "nodes", "edges" or "faces"
        power : int, default=2
            Power parameter for inverse distance weighting. This controls how local or global the remapping is, a higher
            power causes points that are further away to have less influence
        k : int, default=8
            Number of nearest neighbors to consider in the weighted calculation.

        Returns
        -------
        remapped: UxDataArray or UxDataset
            TODO
        """

        return _inverse_distance_weighted_remap(
            self.ux_obj, destination_grid, remap_to, power, k
        )
