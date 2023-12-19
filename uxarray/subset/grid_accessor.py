from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.grid import Grid


class GridSubsetAccessor:

    def __init__(self, uxgrid: Grid) -> None:
        self.uxgrid = uxgrid

    def __call__(self):
        return repr(self)

    def __repr__(self):
        prefix = "<uxarray.Grid.subset>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += "  * nearest_neighbor(center_coord, k, tree_type, **kwargs)\n"
        methods_heading += "  * bounding_circle(center_coord, r, tree_type, **kwargs)\n"
        methods_heading += "  * bounding_box(lon_bounds, lat_bounds, method, element)\n"

        return prefix + methods_heading

    def bounding_box(self,
                     lon_bounds,
                     lat_bounds,
                     method='naive',
                     element='nodes'):
        """Subsets an unstructured grid bounded by a pair of latitude and
        longitude bounds (i.e. (-180, 180), (-10, 10)) and returns a new
        ``Grid``.

        Parameters
        ----------
        TODO:
        """

        if method == "naive":

            if element == "nodes":
                lat, lon = self.uxgrid.node_lat.values, self.uxgrid.node_lon.values
            elif element == "face centers":
                lat, lon = self.uxgrid.face_lat.values, self.uxgrid.face_lon.values
            elif element == "edge centers":
                lat, lon = self.uxgrid.edge_lat.values, self.uxgrid.edge_lon.values
            else:
                raise ValueError("TODO")

            # obtain all lat/lon indices that are within the bounds
            lat_indices = np.argwhere(
                np.logical_and(lat > lat_bounds[0], lat < lat_bounds[1]))
            lon_indices = np.argwhere(
                np.logical_and(lon > lon_bounds[0], lon < lon_bounds[1]))

            # treat both indices as a set, find the intersection of both
            indices = np.intersect1d(lat_indices, lon_indices)

            if element == "nodes":
                return self.uxgrid.isel(n_node=indices)
            elif element == "face centers":
                return self.uxgrid.isel(n_face=indices)
            elif element == "edge centers":
                return self.uxgrid.isel(n_edge=indices)

        else:
            raise ValueError(f"Method '{method}' not supported.")

    def bounding_circle(self, center_coord, r, tree_type='nodes', **kwargs):
        """Subsets an unstructured grid by returning all elements within some
        radius (in degrees) from a center coord.

        Parameters
        ----------
        center_coord : tuple
            Longitude and latitude of the center of the bounding circle
        r: scalar
            Radius of bounding circle (in degrees)
        tree_type: str
            Tree type (either `nodes` or `face centers`) for internal nearest neighbor computations
        """

        coords = np.asarray(center_coord)

        tree = self._get_tree(coords, tree_type)

        _, ind = tree.query_radius(coords, r)

        return self._index_grid(ind, tree_type)

    def nearest_neighbor(self, center_coord, k, tree_type='nodes', **kwargs):
        """Subsets an unstructured grid by returning the ``k`` closest
        neighbors from a center coordinate.

        Parameters
        ----------
        center_coord : tuple
            Longitude and latitude of the center of the bounding circle
        k: int
            Number of neighbors to query
        tree_type: str
            Tree type (either `nodes` or `face centers`) for internal nearest neighbor computations
        """

        coords = np.asarray(center_coord)

        tree = self._get_tree(coords, tree_type)

        _, ind = tree.query(coords, k)

        return self._index_grid(ind, tree_type)

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
