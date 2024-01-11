from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING, Union, Tuple, List, Optional

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

        if method == "coords":
            # coordinate comparison bounding box

            if element == "nodes":
                lat, lon = self.uxgrid.node_lat.values, self.uxgrid.node_lon.values
            elif element == "face centers":
                lat, lon = self.uxgrid.face_lat.values, self.uxgrid.face_lon.values
            elif element == "edge centers":
                lat, lon = self.uxgrid.edge_lat.values, self.uxgrid.edge_lon.values
            else:
                raise ValueError("TODO")

            if lon_bounds[0] > lon_bounds[1]:
                # split across antimeridian

                lon_indices_lhs = np.argwhere(
                    np.logical_and(lon >= -180, lon < lon_bounds[1])
                )

                lon_indices_rhs = np.argwhere(
                    np.logical_and(lon >= lon_bounds[0], lon < 180)
                )

                lon_indices = np.union1d(
                    lon_indices_lhs.squeeze(), lon_indices_rhs.squeeze()
                )
            else:
                # continuous bound

                lon_indices = np.argwhere(
                    np.logical_and(lon > lon_bounds[0], lon < lon_bounds[1])
                )

            lat_indices = np.argwhere(
                np.logical_and(lat > lat_bounds[0], lat < lat_bounds[1])
            )

            # treat both indices as a set, find the intersection of both
            indices = np.intersect1d(lat_indices, lon_indices)

            if len(indices) == 0:
                raise ValueError(
                    f"No elements founding within the bounding box when querying {element}"
                )

            if element == "nodes":
                return self.uxgrid.isel(n_node=indices)
            elif element == "face centers":
                return self.uxgrid.isel(n_face=indices)
            elif element == "edge centers":
                return self.uxgrid.isel(n_edge=indices)

        else:
            raise ValueError(f"Method '{method}' not supported.")

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

        coords = np.asarray(center_coord)

        tree = self._get_tree(coords, element)

        ind = tree.query_radius(coords, r)

        if len(ind) == 0:
            raise ValueError(
                f"No elements founding within the bounding circle with radius {r} when querying {element}"
            )

        return self._index_grid(ind, element)

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

        coords = np.asarray(center_coord)

        tree = self._get_tree(coords, element)

        _, ind = tree.query(coords, k)

        return self._index_grid(ind, element)

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

    def _index_grid(self, ind, tree_type):
        """Internal helper for indexing a grid with indices based off the
        provided tree type."""
        if tree_type == "nodes":
            return self.uxgrid.isel(n_node=ind)
        elif tree_type == "edge centers":
            return self.uxgrid.isel(n_edge=ind)
        else:
            return self.uxgrid.isel(n_face=ind)
