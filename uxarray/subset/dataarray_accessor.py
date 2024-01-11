from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING, Union, Tuple, List, Optional

if TYPE_CHECKING:
    pass


class DataArraySubsetAccessor:
    """Accessor for performing unstructured grid subsetting with a data
    variable, accessed through ``UxDataArray.subset``"""

    def __init__(self, uxda) -> None:
        self.uxda = uxda

    def __repr__(self):
        prefix = "<uxarray.UxDataArray.subset>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += "  * nearest_neighbor(center_coord, k, element, **kwargs)\n"
        methods_heading += "  * bounding_circle(center_coord, r, element, **kwargs)\n"
        methods_heading += (
            "  * bounding_box(lon_bounds, lat_bounds, element, method, **kwargs)\n"
        )

        return prefix + methods_heading

    def bounding_box(
        self,
        lon_bounds: Union[Tuple, List, np.ndarray],
        lat_bounds: Union[Tuple, List, np.ndarray],
        element: Optional[str] = "nodes",
        method: Optional[str] = "coords",
        **kwargs,
    ):
        """Subsets an unstructured grid between two latitude and longitude
        points which form a bounding box.

        A bounding box may span the antimeridian, when the pair of longitude points is given in descending order (
        i.e. the first longitude point is greater than the second).

        Parameters
        ----------
        lon_bounds: tuple, list, np.ndarray
            (lon_left, lon_right) where lon_left < lon_right when the bounding box does not span
            the antimeridian, otherwise lon_left > lon_right, both between [-180, 180]
        lat_bounds: tuple, list, np.ndarray
            (lat_bottom, lat_top) where lat_top > lat_bottom and between [-90, 90]
        method: str
            Bounding Box Method, currently supports 'coords', which ensures the coordinates of the corner nodes,
            face centers, or edge centers lie within the bounds.
        element: str
            Element for use with `coords` comparison, one of `nodes`, `face centers`, or `edge centers`
        """
        grid = self.uxda.uxgrid.subset.bounding_box(
            lon_bounds, lat_bounds, element, method
        )

        return self.uxda._slice_from_grid(grid)

    def bounding_circle(
        self,
        center_coord: Union[Tuple, List, np.ndarray],
        r: Union[float, int],
        element: Optional[str] = "nodes",
        **kwargs,
    ):
        """Subsets an unstructured grid by returning all elements within some
        radius (in degrees) from a center coord.

        Parameters
        ----------
        center_coord : tuple, list, np.ndarray
            Longitude and latitude of the center of the bounding circle
        r: scalar, int, float
            Radius of bounding circle (in degrees)
        element: str
            Element for use with `coords` comparison, one of `nodes`, `face centers`, or `edge centers`
        """
        grid = self.uxda.uxgrid.subset.bounding_circle(
            center_coord, r, element, **kwargs
        )
        return self.uxda._slice_from_grid(grid)

    def nearest_neighbor(
        self,
        center_coord: Union[Tuple, List, np.ndarray],
        k: int,
        element: Optional[str] = "nodes",
        **kwargs,
    ):
        """Subsets an unstructured grid by returning the ``k`` closest
        neighbors from a center coordinate.

        Parameters
        ----------
        center_coord : tuple, list, np.ndarray
            Longitude and latitude of the center of the bounding circle
        k: int
            Number of neighbors to query
        element: str
            Element for use with `coords` comparison, one of `nodes`, `face centers`, or `edge centers`
        """

        grid = self.uxda.uxgrid.subset.nearest_neighbor(
            center_coord, k, element, **kwargs
        )

        return self.uxda._slice_from_grid(grid)
