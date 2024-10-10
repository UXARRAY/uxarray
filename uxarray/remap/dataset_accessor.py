from __future__ import annotations
from typing import TYPE_CHECKING

from uxarray.remap.nearest_neighbor import _nearest_neighbor_uxds
from uxarray.remap.inverse_distance_weighted import (
    _inverse_distance_weighted_remap_uxds,
)

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset

from uxarray.grid import Grid


class UxDatasetRemapAccessor:
    """Remapping accessor for ``UxDataset``"""

    def __init__(self, uxds: UxDataset):
        self.uxds = uxds

    def __repr__(self):
        prefix = "<uxarray.UxDataset.remap>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += (
            "  * nearest_neighbor(destination_obj, remap_to, coord_type)\n"
        )
        methods_heading += "  * inverse_distance_weighted(destination_obj, remap_to, coord_type, power, k)\n"

        return prefix + methods_heading

    def nearest_neighbor(
        self,
        destination_grid: Grid,
        remap_to: str = "face centers",
        coord_type: str = "spherical",
    ):
        """Nearest Neighbor Remapping between a source ``UxDataset`` and
        destination ``Grid``

        Parameters
        ---------
        destination_grid : Grid
            Destination Grid for remapping
        remap_to : str, default="nodes"
            Location of where to map data, either "nodes", "edge centers", or "face centers"
        coord_type : str, default="spherical"
            Indicates whether to remap using on spherical or cartesian coordinates
        """

        return _nearest_neighbor_uxds(self.uxds, destination_grid, remap_to, coord_type)

    def inverse_distance_weighted(
        self,
        destination_grid: Grid,
        remap_to: str = "face centers",
        coord_type: str = "spherical",
        power=2,
        k=8,
    ):
        """Inverse Distance Weighted Remapping between a source ``UxDataset``
        and destination ``Grid``

        Parameters
        ---------
        destination_grid : Grid
            Destination Grid for remapping
        remap_to : str, default="nodes"
            Location of where to map data, either "nodes", "edge centers", or "face centers"
        coord_type : str, default="spherical"
            Indicates whether to remap using on spherical or cartesian coordinates
        power : int, default=2
            Power parameter for inverse distance weighting. This controls how local or global the remapping is, a higher
            power causes points that are further away to have less influence
        k : int, default=8
            Number of nearest neighbors to consider in the weighted calculation.
        """

        return _inverse_distance_weighted_remap_uxds(
            self.uxds, destination_grid, remap_to, coord_type, power, k
        )
