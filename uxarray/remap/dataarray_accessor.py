from __future__ import annotations
from typing import TYPE_CHECKING, Union

from uxarray.remap.nearest_neighbor import _nearest_neighbor_uxda
from uxarray.remap.inverse_distance_weighted import (
    _inverse_distance_weighted_remap_uxda,
)

if TYPE_CHECKING:
    from uxarray.core.dataset import UxDataset
    from uxarray.core.dataarray import UxDataArray

from uxarray.grid import Grid


class UXDataArrayRemapAccessor:
    def __init__(self, uxda: UxDataArray):
        self.uxda = uxda

    def nearest_neighbor(
        self,
        destination_obj: Union[Grid, UxDataArray, UxDataset],
        remap_to: str = "nodes",
        coord_type: str = "spherical",
    ):
        """Nearest Neighbor Remapping between a source (``UxDataArray``) and
        destination.`.

        Parameters
        ---------
        destination_obj : Grid, UxDataArray, UxDataset
            Destination for remapping
        remap_to : str, default="nodes"
            Location of where to map data, either "nodes" or "face centers"
        coord_type : str, default="spherical"
            Indicates whether to remap using on spherical or cartesian coordinates
        """

        return _nearest_neighbor_uxda(self.uxda, destination_obj, remap_to, coord_type)

    def inverse_distance_weighted(
        self,
        destination_obj: Union[Grid, UxDataArray, UxDataset],
        remap_to: str = "nodes",
        coord_type: str = "spherical",
        power=2,
        k=8,
    ):
        """Inverse Distance Weighted Remapping between a source
        (``UxDataArray``) and destination.`.

        Parameters
        ---------
        destination_obj : Grid, UxDataArray, UxDataset
            Destination for remapping
        remap_to : str, default="nodes"
            Location of where to map data, either "nodes" or "face centers"
        coord_type : str, default="spherical"
            Indicates whether to remap using on spherical or cartesian coordinates
        power : int, default=2
            Power parameter for inverse distance weighting. This controls how local or global the remapping is, a higher
            power causes points that are further away to have less influence
        k : int, default=8
            Number of nearest neighbors to consider in the weighted calculation.
        """

        return _inverse_distance_weighted_remap_uxda(
            self.uxda, destination_obj, remap_to, coord_type, power, k
        )
