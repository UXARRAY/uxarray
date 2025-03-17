from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING, Union, Tuple, List, Optional, Set

if TYPE_CHECKING:
    from uxarray.grid import Grid


class GridSubsetAccessor:
    """Accessor for performing unstructured grid subsetting, accessed through
    ``Grid.subset``"""

    def __init__(self, uxgrid: Grid) -> None:
        self.uxgrid = uxgrid

    def __repr__(self):
        prefix = "<uxarray.Grid.subset>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += "  * nearest_neighbor(center_coord, k, element, inverse_indices, **kwargs)\n"
        methods_heading += (
            "  * bounding_circle(center_coord, r, element, inverse_indices, **kwargs)\n"
        )
        methods_heading += "  * bounding_box(lon_bounds, lat_bounds, inverse_indices)\n"

        return prefix + methods_heading

    def bounding_box(
        self,
        lon_bounds: Tuple[float, float],
        lat_bounds: Tuple[float, float],
        inverse_indices: Union[List[str], Set[str], bool] = False,
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
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)
        """

        faces_between_lons = self.uxgrid.get_faces_between_longitudes(lon_bounds)
        face_between_lats = self.uxgrid.get_faces_between_latitudes(lat_bounds)

        faces = np.intersect1d(faces_between_lons, face_between_lats)

        return self.uxgrid.isel(n_face=faces, inverse_indices=inverse_indices)

    def bounding_circle(
        self,
        center_coord: Union[Tuple, List, np.ndarray],
        r: Union[float, int],
        element: Optional[str] = "face centers",
        inverse_indices: Union[List[str], Set[str], bool] = False,
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
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)
        """

        coords = np.asarray(center_coord)

        tree = self._get_tree(coords, element)

        ind = tree.query_radius(coords, r)

        if len(ind) == 0:
            raise ValueError(
                f"No elements founding within the bounding circle with radius {r} when querying {element}"
            )

        return self._index_grid(ind, element, inverse_indices)

    def nearest_neighbor(
        self,
        center_coord: Union[Tuple, List, np.ndarray],
        k: int,
        element: Optional[str] = "face centers",
        inverse_indices: Union[List[str], Set[str], bool] = False,
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
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)
        """

        coords = np.asarray(center_coord)

        tree = self._get_tree(coords, element)

        _, ind = tree.query(coords, k)

        return self._index_grid(ind, element, inverse_indices=inverse_indices)

    def _get_tree(self, coords, tree_type):
        """Internal helper for obtaining the desired KDTree or BallTree."""
        if coords.ndim > 1:
            raise ValueError("Coordinates must be one-dimensional")

        if len(coords) == 2:
            # Spherical coordinates
            tree = self.uxgrid.get_ball_tree(tree_type)
        elif len(coords) == 3:
            # Cartesian coordinates
            tree = self.uxgrid.get_kd_tree(tree_type)
        else:
            raise ValueError("Unsupported coordinates provided.")

        return tree

    def _index_grid(self, ind, tree_type, inverse_indices=False):
        """Internal helper for indexing a grid with indices based off the
        provided tree type."""
        if tree_type == "nodes":
            return self.uxgrid.isel(inverse_indices, n_node=ind)
        elif tree_type == "edge centers":
            return self.uxgrid.isel(inverse_indices, n_edge=ind)
        else:
            return self.uxgrid.isel(inverse_indices, n_face=ind)
