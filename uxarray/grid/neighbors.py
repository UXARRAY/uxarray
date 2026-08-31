import functools
import warnings
from typing import Callable

import numpy as np
import xarray as xr
from numba import guvectorize, njit
from numpy import deg2rad

from uxarray.constants import (
    ERROR_TOLERANCE,
    GRID_DIMS,
    INT_DTYPE,
    INT_FILL_VALUE,
)
from uxarray.errors import DimensionError


class KDTree:
    """Custom KDTree data structure written around the
    ``sklearn.neighbors.KDTree`` implementation for use with corner
    (``node_x``, ``node_y``, ``node_z``) and (``node_lon``, ``node_lat``), edge
    (``edge_x``, ``edge_y``, ``edge_z``) and (``edge_lon``, ``edge_lat``), or
    center (``face_x``, ``face_y``, ``face_z``) and (``face_lon``,
    ``face_lat``) nodes of the inputted unstructured grid.

    Parameters
    ----------
    grid : ux.Grid
        Source grid used to construct the KDTree
    coordinates : str, default="nodes"
            Identifies which tree to construct or select, with "nodes" selecting the corner nodes, "face centers" selecting the face
            centers of each face, and "edge centers" selecting the centers of each edge of a face
    coordinate_system : str, default="cartesian"
            Sets the coordinate type used to construct the KDTree, either cartesian coordinates or spherical coordinates.
    distance_metric : str, default="minkowski"
        Distance metric used to construct the KDTree, available options include:
        'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity'
    reconstruct : bool, default=False
        If true, reconstructs the tree

    Notes
    -----
    See `sklearn.neighbors.KDTree <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html>`__
    for further information about the wrapped data structures.
    """

    def __init__(
        self,
        grid,
        coordinates: str | None = "face centers",
        coordinate_system: str | None = "cartesian",
        distance_metric: str | None = "minkowski",
        reconstruct: bool = False,
    ):
        # Set up references
        self._source_grid = grid
        self._coordinates = coordinates
        self.coordinate_system = coordinate_system
        self.distance_metric = distance_metric
        self.reconstruct = reconstruct

        self._tree_from_nodes = None
        self._tree_from_face_centers = None
        self._tree_from_edge_centers = None

        # Build the tree based on nodes, face centers, or edge centers
        if coordinates == "nodes":
            self._tree_from_nodes = self._build_from_nodes()
            self._n_elements = self._source_grid.n_node
        elif coordinates == "face centers":
            self._tree_from_face_centers = self._build_from_face_centers()
            self._n_elements = self._source_grid.n_face
        elif coordinates == "edge centers":
            self._tree_from_edge_centers = self._build_from_edge_centers()
            self._n_elements = self._source_grid.n_edge
        else:
            raise ValueError(
                f"Unknown coordinates location, {self._coordinates}, use either 'nodes', 'face centers', "
                f"or 'edge centers'"
            )

    def _build_from_nodes(self):
        """Internal``sklearn.neighbors.KDTree`` constructed from corner
        nodes."""
        from sklearn.neighbors import KDTree as SKKDTree

        if self._tree_from_nodes is None or self.reconstruct:
            # Sets which values to use for the tree based on the coordinate_system
            if self.coordinate_system == "cartesian":
                coords = np.stack(
                    (
                        self._source_grid.node_x.values,
                        self._source_grid.node_y.values,
                        self._source_grid.node_z.values,
                    ),
                    axis=-1,
                )

            elif self.coordinate_system == "spherical":
                coords = np.vstack(
                    (
                        deg2rad(self._source_grid.node_lat.values),
                        deg2rad(self._source_grid.node_lon.values),
                    )
                ).T

            else:
                raise ValueError(
                    f"Unknown coordinate_system, {self.coordinate_system}, use either 'cartesian' or "
                    f"'spherical'"
                )

            self._tree_from_nodes = SKKDTree(coords, metric=self.distance_metric)

        return self._tree_from_nodes

    def _build_from_face_centers(self):
        """Internal``sklearn.neighbors.KDTree`` constructed from face
        centers."""
        from sklearn.neighbors import KDTree as SKKDTree

        if self._tree_from_face_centers is None or self.reconstruct:
            # Sets which values to use for the tree based on the coordinate_system
            if self.coordinate_system == "cartesian":
                coords = np.stack(
                    (
                        self._source_grid.face_x.values,
                        self._source_grid.face_y.values,
                        self._source_grid.face_z.values,
                    ),
                    axis=-1,
                )

            elif self.coordinate_system == "spherical":
                coords = np.vstack(
                    (
                        deg2rad(self._source_grid.face_lat.values),
                        deg2rad(self._source_grid.face_lon.values),
                    )
                ).T

            else:
                raise ValueError(
                    f"Unknown coordinate_system, {self.coordinate_system}, use either 'cartesian' or "
                    f"'spherical'"
                )

            self._tree_from_face_centers = SKKDTree(coords, metric=self.distance_metric)

        return self._tree_from_face_centers

    def _build_from_edge_centers(self):
        """Internal``sklearn.neighbors.KDTree`` constructed from edge
        centers."""
        from sklearn.neighbors import KDTree as SKKDTree

        if self._tree_from_edge_centers is None or self.reconstruct:
            # Sets which values to use for the tree based on the coordinate_system
            if self.coordinate_system == "cartesian":
                if self._source_grid.edge_x is None:
                    raise ValueError("edge_x isn't populated")

                coords = np.stack(
                    (
                        self._source_grid.edge_x.values,
                        self._source_grid.edge_y.values,
                        self._source_grid.edge_z.values,
                    ),
                    axis=-1,
                )

            elif self.coordinate_system == "spherical":
                if self._source_grid.edge_lat is None:
                    raise ValueError("edge_lat isn't populated")

                coords = np.vstack(
                    (
                        deg2rad(self._source_grid.edge_lat.values),
                        deg2rad(self._source_grid.edge_lon.values),
                    )
                ).T

            else:
                raise ValueError(
                    f"Unknown coordinate_system, {self.coordinate_system}, use either 'cartesian' or "
                    f"'spherical'"
                )

            self._tree_from_edge_centers = SKKDTree(coords, metric=self.distance_metric)

        return self._tree_from_edge_centers

    def _current_tree(self):
        """Creates and returns the current tree."""
        _tree = None

        if self._coordinates == "nodes":
            _tree = self._tree_from_nodes
        elif self._coordinates == "face centers":
            _tree = self._tree_from_face_centers
        elif self._coordinates == "edge centers":
            _tree = self._tree_from_edge_centers
        else:
            raise ValueError(
                f"Unknown coordinates location, {self._coordinates}, use either 'nodes', 'face centers', "
                f"or 'edge centers'"
            )

        return _tree

    def query(
        self,
        coords: np.ndarray | list | tuple,
        k: int | None = 1,
        return_distance: bool | None = True,
        in_radians: bool | None = False,
        dualtree: bool | None = False,
        breadth_first: bool | None = False,
        sort_results: bool | None = True,
    ):
        """Queries the tree for the ``k`` nearest neighbors.

        Parameters
        ----------
        coords : array_like
            coordinate pairs in cartesian (x, y, z) or spherical (lat, lon) to query
        k: int, default=1
            The number of nearest neighbors to return
        return_distance : bool, optional
            Indicates whether distances should be returned
        in_radians : bool, optional
            if True, queries assuming coords are inputted in radians, not degrees. Only applies for spherical coordinates
        dualtree : bool, default=False
            Indicates whether to use the dual-tree formalism for node queries
        breadth_first : bool, default=False
            Indicates whether to query nodes in a breadth-first manner
        sort_results : bool, default=True
            Indicates whether distances should be sorted

        Returns
        -------
        d : ndarray of shape (xyz.shape[0], k), dtype=double
            Distance array that keeps the distances of the k-nearest neighbors to the entries from xyz in each row
        ind : ndarray of shape (xyz.shape[0], k), dtype=INT_DTYPE
            Index array that keeps the indices of the k-nearest neighbors to the entries from xyz in each row
        """

        if k < 1 or k > self._n_elements:
            raise AssertionError(
                f"The value of k must be greater than 1 and less than the number of elements used to construct "
                f"the tree ({self._n_elements})."
            )
        if self.coordinate_system == "cartesian":
            coords = _prepare_xyz_for_query(coords)
        elif self.coordinate_system == "spherical":
            coords = _prepare_xy_for_query(
                coords, in_radians, distance_metric=self.distance_metric
            )
        else:
            raise ValueError(
                f"Unknown coordinate_system, {self.coordinate_system}, use either 'cartesian' or "
                f"'spherical'"
            )

        # perform query with distance
        if return_distance:
            d, ind = self._current_tree().query(
                coords, k, return_distance, dualtree, breadth_first, sort_results
            )

            ind = np.asarray(ind, dtype=INT_DTYPE)

            if coords.shape[0] == 1:
                ind = ind.squeeze()

            # only one pair was queried
            if coords.shape[0] == 1:
                d = d.squeeze()

            if not in_radians and self.coordinate_system == "spherical":
                d = np.rad2deg(d)

            return d, ind

        # perform query without distance
        else:
            ind = self._current_tree().query(
                coords, k, return_distance, dualtree, breadth_first, sort_results
            )

            ind = np.asarray(ind, dtype=INT_DTYPE)

            if coords.shape[0] == 1:
                ind = ind.squeeze()
        return ind

    def query_radius(
        self,
        coords: np.ndarray | list | tuple,
        r: int | None = 1.0,
        return_distance: bool | None = False,
        in_radians: bool | None = False,
        count_only: bool | None = False,
        sort_results: bool | None = False,
    ):
        """Queries the tree for all neighbors within a radius ``r``.

        Parameters
        ----------
        coords : array_like
           coordinate pairs in cartesian (x, y, z) or spherical (lat, lon) to query
        r: distance within which neighbors are returned
            r is a single value for the radius of which to query
        return_distance : bool, default=False
            Indicates whether distances should be returned
        in_radians : bool, optional
            if True, queries assuming coords are inputted in radians, not degrees. Only applies to spherical coordinates
        count_only : bool, default=False
            Indicates whether only counts should be returned
        sort_results : bool, default=False
            Indicates whether distances should be sorted

        Returns
        -------
        d : ndarray of shape (xyz.shape[0], k), dtype=double
            Distance array that keeps the distances of all neighbors within some radius to the entries from xyz in each row
        ind : ndarray of shape (xyz.shape[0], k), dtype=INT_DTYPE
            Index array that keeps the indices of all neighbors within some radius to the entries from xyz in each row
        """

        if r < 0.0:
            raise AssertionError(
                "The value of r must be greater than or equal to zero."
            )

        # Use the correct function to prepare for query based on coordinate type
        if self.coordinate_system == "cartesian":
            coords = _prepare_xyz_for_query(coords)
        elif self.coordinate_system == "spherical":
            coords = _prepare_xy_for_query(
                coords, in_radians, distance_metric=self.distance_metric
            )
        else:
            raise ValueError(
                f"Unknown coordinate_system, {self.coordinate_system}, use either 'cartesian' or "
                f"'spherical'"
            )

        if count_only:
            count = self._current_tree().query_radius(
                coords, r, return_distance, count_only, sort_results
            )

            return count

        elif return_distance:
            ind, d = self._current_tree().query_radius(
                coords, r, return_distance, count_only, sort_results
            )

            ind = [np.asarray(cur_ind, dtype=INT_DTYPE) for cur_ind in ind]
            d = [np.asarray(cur_d) for cur_d in d]

            if coords.shape[0] == 1:
                ind = ind[0]
                d = d[0]

            if not in_radians and self.coordinate_system == "spherical":
                d = [np.rad2deg(cur_d) for cur_d in d]

            return d, ind
        else:
            ind = self._current_tree().query_radius(
                coords, r, return_distance, count_only, sort_results
            )

            ind = [np.asarray(cur_ind, dtype=INT_DTYPE) for cur_ind in ind]

            if coords.shape[0] == 1:
                ind = ind[0]

            return ind

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        self._coordinates = value

        # set up appropriate reference to tree
        if self._coordinates == "nodes":
            if self._tree_from_nodes is None or self.reconstruct:
                self._tree_from_nodes = self._build_from_nodes()
            self._n_elements = self._source_grid.n_node
        elif self._coordinates == "face centers":
            if self._tree_from_face_centers is None or self.reconstruct:
                self._tree_from_face_centers = self._build_from_face_centers()
            self._n_elements = self._source_grid.n_face
        elif self._coordinates == "edge centers":
            if self._tree_from_edge_centers is None or self.reconstruct:
                self._tree_from_edge_centers = self._build_from_edge_centers()
            self._n_elements = self._source_grid.n_edge
        else:
            raise ValueError(
                f"Unknown coordinates location, {self._coordinates}, use either 'nodes', 'face centers', "
                f"or 'edge centers'"
            )


class BallTree:
    """Custom BallTree data structure written around the
    ``sklearn.neighbors.BallTree`` implementation for use with either the
    (``node_x``, ``node_y``, ``node_z``) and (``node_lon``, ``node_lat``), edge
    (``edge_x``, ``edge_y``, ``edge_z``) and (``edge_lon``, ``edge_lat``), or
    center (``face_x``, ``face_y``, ``face_z``) and (``face_lon``,
    ``face_lat``) nodes of the inputted unstructured grid.

    Parameters
    ----------
    grid : ux.Grid
        Source grid used to construct the BallTree
    coordinates : str, default="nodes"
            Identifies which tree to construct or select, with "nodes" selecting the Corner Nodes, "face centers" selecting the Face
            Centers of each face, and "edge centers" selecting the edge centers of each face.
    distance_metric : str, default="haversine"
        Distance metric used to construct the BallTree, options include:
        'euclidean', 'l2', 'minkowski', 'p','manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity', 'seuclidean',
        'mahalanobis', 'hamming', 'canberra', 'braycurtis', 'jaccard', 'dice', 'rogerstanimoto', 'russellrao',
        'sokalmichener', 'sokalsneath', 'haversine'

    Notes
    -----
    See `sklearn.neighbors.BallTree <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html>`__
    for further information about the wrapped data structures.
    """

    def __init__(
        self,
        grid,
        coordinates: str | None = "face centers",
        coordinate_system: str | None = "spherical",
        distance_metric: str | None = "haversine",
        reconstruct: bool = False,
    ):
        # maintain a reference to the source grid
        self._source_grid = grid
        self.distance_metric = distance_metric
        self._coordinates = coordinates
        self.coordinate_system = coordinate_system
        self.reconstruct = reconstruct

        self._tree_from_nodes = None
        self._tree_from_face_centers = None
        self._tree_from_edge_centers = None

        # set up appropriate reference to tree
        if coordinates == "nodes":
            self._tree_from_nodes = self._build_from_nodes()
            self._n_elements = self._source_grid.n_node
        elif coordinates == "face centers":
            self._tree_from_face_centers = self._build_from_face_centers()
            self._n_elements = self._source_grid.n_face
        elif coordinates == "edge centers":
            self._tree_from_edge_centers = self._build_from_edge_centers()
            self._n_elements = self._source_grid.n_edge
        else:
            raise ValueError(
                f"Unknown coordinates location, {self._coordinates}, use either 'nodes', 'face centers', "
                f"or 'edge centers'"
            )

    def _build_from_face_centers(self):
        """Internal``sklearn.neighbors.BallTree`` constructed from face
        centers."""
        from sklearn.neighbors import BallTree as SKBallTree

        if self._tree_from_face_centers is None or self.reconstruct:
            # Sets which values to use for the tree based on the coordinate_system
            if self.coordinate_system == "spherical":
                coords = np.vstack(
                    (
                        deg2rad(self._source_grid.face_lat.values),
                        deg2rad(self._source_grid.face_lon.values),
                    )
                ).T

            elif self.coordinate_system == "cartesian":
                coords = np.stack(
                    (
                        self._source_grid.face_x.values,
                        self._source_grid.face_y.values,
                        self._source_grid.face_z.values,
                    ),
                    axis=-1,
                )
            else:
                raise ValueError(
                    f"Unknown coordinate_system, {self.coordinate_system}, use either 'cartesian' or "
                    f"'spherical'"
                )

            self._tree_from_face_centers = SKBallTree(
                coords, metric=self.distance_metric
            )

        return self._tree_from_face_centers

    def _build_from_nodes(self):
        """Internal``sklearn.neighbors.BallTree`` constructed from corner
        nodes."""
        from sklearn.neighbors import BallTree as SKBallTree

        if self._tree_from_nodes is None or self.reconstruct:
            # Sets which values to use for the tree based on the coordinate_system
            if self.coordinate_system == "spherical":
                coords = np.vstack(
                    (
                        deg2rad(self._source_grid.node_lat.values),
                        deg2rad(self._source_grid.node_lon.values),
                    )
                ).T

            if self.coordinate_system == "cartesian":
                coords = np.stack(
                    (
                        self._source_grid.node_x.values,
                        self._source_grid.node_y.values,
                        self._source_grid.node_z.values,
                    ),
                    axis=-1,
                )
            self._tree_from_nodes = SKBallTree(coords, metric=self.distance_metric)

        return self._tree_from_nodes

    def _build_from_edge_centers(self):
        """Internal``sklearn.neighbors.BallTree`` constructed from edge
        centers."""
        from sklearn.neighbors import BallTree as SKBallTree

        if self._tree_from_edge_centers is None or self.reconstruct:
            # Sets which values to use for the tree based on the coordinate_system
            if self.coordinate_system == "spherical":
                if self._source_grid.edge_lat is None:
                    raise ValueError("edge_lat isn't populated")

                coords = np.vstack(
                    (
                        deg2rad(self._source_grid.edge_lat.values),
                        deg2rad(self._source_grid.edge_lon.values),
                    )
                ).T

            elif self.coordinate_system == "cartesian":
                if self._source_grid.edge_x is None:
                    raise ValueError("edge_x isn't populated")

                coords = np.stack(
                    (
                        self._source_grid.edge_x.values,
                        self._source_grid.edge_y.values,
                        self._source_grid.edge_z.values,
                    ),
                    axis=-1,
                )
            else:
                raise ValueError(
                    f"Unknown coordinate_system, {self.coordinate_system}, use either 'cartesian' or "
                    f"'spherical'"
                )

            self._tree_from_edge_centers = SKBallTree(
                coords, metric=self.distance_metric
            )

        return self._tree_from_edge_centers

    def _current_tree(self):
        _tree = None

        if self._coordinates == "nodes":
            _tree = self._tree_from_nodes
        elif self._coordinates == "face centers":
            _tree = self._tree_from_face_centers
        elif self._coordinates == "edge centers":
            _tree = self._tree_from_edge_centers
        else:
            raise TypeError(
                f"Unknown coordinates location, {self._coordinates}, use either 'nodes', 'face centers', "
                f"or 'edge centers'"
            )

        return _tree

    def query(
        self,
        coords: np.ndarray | list | tuple,
        k: int | None = 1,
        in_radians: bool | None = False,
        return_distance: bool | None = True,
        dualtree: bool | None = False,
        breadth_first: bool | None = False,
        sort_results: bool | None = True,
    ):
        """Queries the tree for the ``k`` nearest neighbors.

        Parameters
        ----------
        coords : array_like
            coordinate pairs in degrees (lon, lat) or cartesian (x, y, z) to query
        k: int, default=1
            The number of nearest neighbors to return
        in_radians : bool, optional
            if True, queries assuming coords are inputted in radians, not degrees. Only applies to spherical coords
        return_distance : bool, optional
            Indicates whether distances should be returned
        dualtree : bool, default=False
            Indicates whether to use the dual-tree formalism for node queries
        breadth_first : bool, default=False
            Indicates whether to query nodes in a breadth-first manner
        sort_results : bool, default=True
            Indicates whether distances should be sorted

        Returns
        -------
        d : ndarray of shape (coords.shape[0], k), dtype=double
            Distance array that keeps the distances of the k-nearest neighbors to the entries from coords in each row
        ind : ndarray of shape (coords.shape[0], k), dtype=INT_DTYPE
            Index array that keeps the indices of the k-nearest neighbors to the entries from coords in each row
        """

        if k < 1 or k > self._n_elements:
            raise AssertionError(
                f"The value of k must be greater than 1 and less than the number of elements used to construct "
                f"the tree ({self._n_elements})."
            )

        # Use the correct function to prepare for query based on coordinate type
        if self.coordinate_system == "spherical":
            coords = _prepare_xy_for_query(
                coords, in_radians, distance_metric=self.distance_metric
            )

        elif self.coordinate_system == "cartesian":
            coords = _prepare_xyz_for_query(coords)

        # perform query with distance
        if return_distance:
            d, ind = self._current_tree().query(
                coords, k, return_distance, dualtree, breadth_first, sort_results
            )

            ind = np.asarray(ind, dtype=INT_DTYPE)

            if coords.shape[0] == 1:
                ind = ind.squeeze()

            # only one pair was queried
            if coords.shape[0] == 1:
                d = d.squeeze()

            if not in_radians and self.coordinate_system == "spherical":
                d = np.rad2deg(d)

            return d, ind

        # perform query without distance
        else:
            ind = self._current_tree().query(
                coords, k, return_distance, dualtree, breadth_first, sort_results
            )

            ind = np.asarray(ind, dtype=INT_DTYPE)

            if coords.shape[0] == 1:
                ind = ind.squeeze()

            return ind

    def query_radius(
        self,
        coords: np.ndarray | list | tuple,
        r: int | None = 1.0,
        in_radians: bool | None = False,
        return_distance: bool | None = False,
        count_only: bool | None = False,
        sort_results: bool | None = False,
    ):
        """Queries the tree for all neighbors within a radius ``r``.

        Parameters
        ----------
        coords : array_like
           coordinate pairs in degrees (lon, lat) to query
        r: distance in degrees within which neighbors are returned
            r is a single value for the radius of which to query
        in_radians : bool, optional
            if True, queries assuming coords are inputted in radians, not degrees. Only applies to spherical coordinates
        return_distance : bool, default=False
            Indicates whether distances should be returned
        count_only : bool, default=False
            Indicates whether only counts should be returned
        sort_results : bool, default=False
            Indicates whether distances should be sorted

        Returns
        -------
        d : ndarray of shape (coords.shape[0], k), dtype=double
            Distance array that keeps the distances of all neighbors within some radius to the entries from coords in each row
        ind : ndarray of shape (coords.shape[0], k), dtype=INT_DTYPE
            Index array that keeps the indices of all neighbors within some radius to the entries from coords in each row
        """

        if r < 0.0:
            raise AssertionError(
                "The value of r must be greater than or equal to zero."
            )

        # Use the correct function to prepare for query based on coordinate type
        if self.coordinate_system == "spherical":
            r = np.deg2rad(r)
            coords = _prepare_xy_for_query(
                coords, in_radians, distance_metric=self.distance_metric
            )

        if self.coordinate_system == "cartesian":
            coords = _prepare_xyz_for_query(coords)

        if count_only:
            count = self._current_tree().query_radius(
                coords, r, return_distance, count_only, sort_results
            )

            return count

        elif return_distance:
            ind, d = self._current_tree().query_radius(
                coords, r, return_distance, count_only, sort_results
            )

            ind = [np.asarray(cur_ind, dtype=INT_DTYPE) for cur_ind in ind]
            d = [np.asarray(cur_d) for cur_d in d]

            if coords.shape[0] == 1:
                ind = ind[0]
                d = d[0]

            if not in_radians and self.coordinate_system == "spherical":
                d = [np.rad2deg(cur_d) for cur_d in d]

            return d, ind
        else:
            ind = self._current_tree().query_radius(
                coords, r, return_distance, count_only, sort_results
            )

            ind = [np.asarray(cur_ind, dtype=INT_DTYPE) for cur_ind in ind]

            if coords.shape[0] == 1:
                ind = ind[0]

            return ind

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        self._coordinates = value

        # set up appropriate reference to tree
        if self._coordinates == "nodes":
            if self._tree_from_nodes is None or self.reconstruct:
                self._tree_from_nodes = self._build_from_nodes()
            self._n_elements = self._source_grid.n_node
        elif self._coordinates == "face centers":
            if self._tree_from_face_centers is None or self.reconstruct:
                self._tree_from_face_centers = self._build_from_face_centers()
            self._n_elements = self._source_grid.n_face
        elif self._coordinates == "edge centers":
            if self._tree_from_edge_centers is None or self.reconstruct:
                self._tree_from_edge_centers = self._build_from_edge_centers()
            self._n_elements = self._source_grid.n_edge
        else:
            raise ValueError(
                f"Unknown coordinates location, {self._coordinates}, use either 'nodes', 'face centers', "
                f"or 'edge centers'"
            )


class SpatialHash:
    """Custom data structure that is used for performing grid searches using Spatial Hashing. This class constructs an overlying
    uniformly spaced structured grid, called the "hash grid" on top an unstructured grid. Faces in the unstructured grid are related
    to the cells in the hash grid by determining the hash cells the bounding box of the unstructured face cells overlap with.

    Parameters
    ----------
    grid : ux.Grid
        Source grid used to construct the hash grid and hash table
    reconstruct : bool, default=False
        If true, reconstructs the spatial hash

    Note
    ----
    Does not currently support queries on periodic elements.
    """

    def __init__(
        self,
        grid,
        reconstruct: bool = False,
    ):
        self._source_grid = grid
        self._nelements = self._source_grid.n_face

        self.reconstruct = reconstruct

        # Hash grid size
        self._dh = self._hash_cell_size()

        # Lower left corner of the hash grid
        lon_min = np.deg2rad(self._source_grid.node_lon.min().to_numpy())
        lat_min = np.deg2rad(self._source_grid.node_lat.min().to_numpy())
        lon_max = np.deg2rad(self._source_grid.node_lon.max().to_numpy())
        lat_max = np.deg2rad(self._source_grid.node_lat.max().to_numpy())

        self._xmin = lon_min - self._dh
        self._ymin = lat_min - self._dh
        self._xmax = lon_max + self._dh
        self._ymax = lat_max + self._dh

        # Number of x points in the hash grid; used for
        # array flattening
        Lx = self._xmax - self._xmin
        Ly = self._ymax - self._ymin
        self._nx = int(np.ceil(Lx / self._dh))
        self._ny = int(np.ceil(Ly / self._dh))

        # Generate the mapping from the hash indices to unstructured grid elements
        self._face_hash_table = None
        self._face_hash_table = self._initialize_face_hash_table()

    def _hash_cell_size(self):
        """Computes the size of the hash cells from the source grid.
        The hash cell size is set to 1/2 of the median edge length in the grid (in radians)"""
        return self._source_grid.edge_node_distances.median().to_numpy() * 0.5

    def _hash_index2d(self, coords):
        """Computes the 2-d hash index (i,j) for the location (x,y), where x and y are given in spherical
        coordinates (in degrees)"""

        i = ((coords[:, 0] - self._xmin) / self._dh).astype(INT_DTYPE)
        j = ((coords[:, 1] - self._ymin) / self._dh).astype(INT_DTYPE)
        return i, j

    def _hash_index(self, coords):
        """Computes the flattened hash index for the location (x,y), where x and y are given in spherical
        coordinates (in degrees). The single dimensioned hash index orders the flat index with all of the
        i-points first and then all the j-points."""
        i, j = self._hash_index2d(coords)
        return i + self._nx * j

    def _initialize_face_hash_table(self):
        """Create a mapping that relates unstructured grid faces to hash indices by determining
        which faces overlap with which hash cells"""

        if self._face_hash_table is None or self.reconstruct:
            index_to_face = [[] for i in range(self._nx * self._ny)]
            lon_bounds = np.sort(self._source_grid.face_bounds_lon.to_numpy(), 1)
            lat_bounds = self._source_grid.face_bounds_lat.to_numpy()

            coords = np.column_stack(
                (
                    np.deg2rad(lon_bounds[:, 0].flatten()),
                    np.deg2rad(lat_bounds[:, 0].flatten()),
                )
            )
            i1, j1 = self._hash_index2d(coords)
            coords = np.column_stack(
                (
                    np.deg2rad(lon_bounds[:, 1].flatten()),
                    np.deg2rad(lat_bounds[:, 1].flatten()),
                )
            )
            i2, j2 = self._hash_index2d(coords)

            try:
                for eid in range(self._source_grid.n_face):
                    for j in range(j1[eid], j2[eid] + 1):
                        for i in range(i1[eid], i2[eid] + 1):
                            index_to_face[i + self._nx * j].append(eid)
            except IndexError:
                raise IndexError(
                    "list index out of range. This may indicate incorrect `edge_node_distances` values."
                )

            return index_to_face

    def query(
        self,
        coords: np.ndarray | list | tuple,
        in_radians: bool | None = False,
        tol: float | None = 1e-6,
    ):
        """Queries the hash table.

        Parameters
        ----------
        coords : array_like
            coordinate pairs in degrees (lon, lat) to query
        in_radians : bool, optional
            if True, queries assuming coords are inputted in radians, not degrees. Only applies to spherical coords


        Returns
        -------
        faces : ndarray of shape (coords.shape[0]), dtype=INT_DTYPE
            Face id's in the self._source_grid where each coords element is found. When a coords element is not found, the
            corresponding array entry in faces is set to -1.
        bcoords : ndarray of shape (coords.shape[0], self._source_grid.n_max_face_nodes), dtype=double
            Barycentric coordinates of each coords element
        """

        coords = _prepare_xy_for_query(coords, in_radians, distance_metric=None)
        num_coords = coords.shape[0]
        max_nodes = self._source_grid.n_max_face_nodes

        # Preallocate results
        bcoords = np.zeros((num_coords, max_nodes), dtype=np.double)
        faces = np.full(num_coords, -1, dtype=INT_DTYPE)

        # Get grid variables
        n_nodes_per_face = self._source_grid.n_nodes_per_face.to_numpy()
        face_node_connectivity = self._source_grid.face_node_connectivity.to_numpy()

        # Precompute radian values for node coordinates:
        node_lon = np.deg2rad(self._source_grid.node_lon.to_numpy())
        node_lat = np.deg2rad(self._source_grid.node_lat.to_numpy())

        # Get the list of candidate faces for each coordinate
        candidate_faces = [
            self._face_hash_table[pid] for pid in self._hash_index(coords)
        ]

        for i, (coord, candidates) in enumerate(zip(coords, candidate_faces)):
            for face_id in candidates:
                n_nodes = n_nodes_per_face[face_id]
                node_ids = face_node_connectivity[face_id, :n_nodes]
                nodes = np.column_stack((node_lon[node_ids], node_lat[node_ids]))
                bcoord = np.asarray(_barycentric_coordinates(nodes, coord))
                err = abs(np.dot(bcoord, nodes[:, 0]) - coord[0]) + abs(
                    np.dot(bcoord, nodes[:, 1]) - coord[1]
                )
                if (bcoord >= 0).all() and err < tol:
                    faces[i] = face_id
                    bcoords[i, :n_nodes] = bcoord[:n_nodes]
                    break

        return faces, bcoords


@njit(cache=True)
def _triangle_area(A, B, C):
    """
    Compute the area of a triangle given by three points.
    """
    return 0.5 * abs(A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))


@njit(cache=True)
def _barycentric_coordinates(nodes, point):
    """
    Compute the barycentric coordinates of a point P inside a convex polygon using area-based weights.
    So that this method generalizes to n-sided polygons, we use the Waschpress points as the generalized
    barycentric coordinates, which is only valid for convex polygons.

    Parameters
    ----------
        nodes : numpy.ndarray
            Spherical coordinates (lon,lat) of each corner node of a face
        point : numpy.ndarray
            Spherical coordinates (lon,lat) of the point
    Returns
    -------
    numpy.ndarray
        Barycentric coordinates corresponding to each vertex.

    """
    n = len(nodes)
    sum_wi = 0
    w = []

    for i in range(0, n):
        vim1 = nodes[i - 1]
        vi = nodes[i]
        vi1 = nodes[(i + 1) % n]
        a0 = _triangle_area(vim1, vi, vi1)
        a1 = max(_triangle_area(point, vim1, vi), ERROR_TOLERANCE)
        a2 = max(_triangle_area(point, vi, vi1), ERROR_TOLERANCE)
        sum_wi += a0 / (a1 * a2)
        w.append(a0 / (a1 * a2))
    barycentric_coords = [w_i / sum_wi for w_i in w]

    return barycentric_coords


def _prepare_xy_for_query(xy, use_radians, distance_metric):
    """Prepares xy coordinates for query with the sklearn BallTree or
    KDTree."""

    xy = np.asarray(xy)

    # expand if only a single node pair is provided
    if xy.ndim == 1:
        xy = np.expand_dims(xy, axis=0)

    # expected shape is [n_pairs, 2]
    if xy.shape[1] == 3:
        raise DimensionError(
            "The dimension of each coordinate pair must be two (lon, lat). Did you attempt to query using Cartesian "
            "(x, y, z) coordinates?"
        )

    if xy.shape[1] != 2:
        raise DimensionError(
            "The dimension of each coordinate pair must be two (lon, lat).)"
        )

    # swap x and y if the distance metric used is haversine
    if distance_metric == "haversine":
        # swap X and Y for query
        xy = np.flip(xy, axis=1)

    # balltree expects units in radians for query
    if not use_radians:
        xy = np.deg2rad(xy)

    return xy


def _prepare_xyz_for_query(xyz):
    """Prepares xyz coordinates for query with the sklearn BallTree and
    KDTree."""

    xyz = np.asarray(xyz)

    # expand if only a single node pair is provided
    if xyz.ndim == 1:
        xyz = np.expand_dims(xyz, axis=0)

    # expected shape is [n_pairs, 3]
    if xyz.shape[1] == 2:
        raise DimensionError(
            "The dimension of each coordinate pair must be three (x, y, z). Did you attempt to query using latlon "
            "(lat, lon) coordinates?"
        )

    if xyz.shape[1] != 3:
        raise DimensionError(
            "The dimension of each coordinate pair must be three (x, y, z).)"
        )

    return xyz


def _populate_edge_node_distances(grid):
    """Populates ``edge_node_distances``"""
    edge_node_distances = _construct_edge_node_distances(
        grid.node_lon.values, grid.node_lat.values, grid.edge_node_connectivity.values
    )

    grid._ds["edge_node_distances"] = xr.DataArray(
        data=edge_node_distances,
        dims=["n_edge"],
        attrs={
            "long_name": "arc distance between the nodes of each edge",
        },
    )


@njit(cache=True)
def _construct_edge_node_distances(node_lon, node_lat, edge_nodes):
    """Helper for computing the arc-distance between nodes compose each
    edge."""

    edge_lon_a = np.deg2rad((node_lon[edge_nodes[:, 0]]))
    edge_lon_b = np.deg2rad((node_lon[edge_nodes[:, 1]]))

    edge_lat_a = np.deg2rad((node_lat[edge_nodes[:, 0]]))
    edge_lat_b = np.deg2rad((node_lat[edge_nodes[:, 1]]))

    # arc length
    edge_node_distances = np.arccos(
        np.sin(edge_lat_a) * np.sin(edge_lat_b)
        + np.cos(edge_lat_a) * np.cos(edge_lat_b) * np.cos(edge_lon_a - edge_lon_b)
    )

    return edge_node_distances


def _populate_edge_face_distances(grid):
    """Populates ``edge_face_distances``"""
    edge_face_distances = _construct_edge_face_distances(
        grid.face_lon.values, grid.face_lat.values, grid.edge_face_connectivity.values
    )

    grid._ds["edge_face_distances"] = xr.DataArray(
        data=edge_face_distances,
        dims=["n_edge"],
        attrs={
            "long_name": "arc distance between the face centers that saddle each edge",
        },
    )


@njit(cache=True)
def _construct_edge_face_distances(face_lon, face_lat, edge_faces):
    """Helper for computing the arc-distance between faces that saddle a given
    edge."""

    saddle_mask = edge_faces[:, 1] != INT_FILL_VALUE

    edge_face_distances = np.zeros(edge_faces.shape[0])

    edge_lon_a = np.deg2rad((face_lon[edge_faces[saddle_mask, 0]]))
    edge_lon_b = np.deg2rad((face_lon[edge_faces[saddle_mask, 1]]))

    edge_lat_a = np.deg2rad((face_lat[edge_faces[saddle_mask, 0]]))
    edge_lat_b = np.deg2rad((face_lat[edge_faces[saddle_mask, 1]]))

    # arc length
    edge_face_distances[saddle_mask] = np.arccos(
        np.sin(edge_lat_a) * np.sin(edge_lat_b)
        + np.cos(edge_lat_a) * np.cos(edge_lat_b) * np.cos(edge_lon_a - edge_lon_b)
    )

    return edge_face_distances


def _get_element_coords(grid, data_mapping: str, coordinate_system: str):
    """Gathers the coordinate array used to query a ``BallTree`` for a given
    grid element location and coordinate system.

    Parameters
    ----------
    grid : Grid
        Source grid containing the coordinate arrays.
    data_mapping : str
        One of "nodes", "edge centers", or "face centers".
    coordinate_system : str
        Either "spherical" or "cartesian".

    Returns
    -------
    coords : np.ndarray
        Array of shape (n_elements, 2) for "spherical" (lon, lat) or
        (n_elements, 3) for "cartesian" (x, y, z).
    """
    prefix_map = {
        "nodes": "node",
        "edge centers": "edge",
        "face centers": "face",
    }

    if data_mapping not in prefix_map:
        raise ValueError(
            f"Invalid data_mapping. Expected 'nodes', 'edge centers', or 'face centers', "
            f"but received: {data_mapping}"
        )

    prefix = prefix_map[data_mapping]

    if coordinate_system == "spherical":
        lon = getattr(grid, f"{prefix}_lon").values
        lat = getattr(grid, f"{prefix}_lat").values
        return np.vstack((lon, lat)).T

    elif coordinate_system == "cartesian":
        x = getattr(grid, f"{prefix}_x").values
        y = getattr(grid, f"{prefix}_y").values
        z = getattr(grid, f"{prefix}_z").values
        return np.vstack((x, y, z)).T

    else:
        raise ValueError(
            f"Invalid coordinate_system. Expected either 'spherical' or 'cartesian', "
            f"but received {coordinate_system}"
        )


# A neighborhood reduction is a segmented reduction over a ragged (CSR-like)
# neighbor structure: elementwise in every dimension except the grid axis,
# which it reduces over. That is exactly a generalized ufunc signature, so the
# kernels below declare the grid axis as a core dimension. Two consequences
# fall out of stating it that way:
#
#   * dask can parallelize over the remaining (chunked) dimensions on its own,
#     so the filter stays lazy instead of materializing the whole array, and
#   * the grid axis is a *core* dimension, so dask refuses to split it rather
#     than silently handing a kernel a block the neighbor indices overrun.
#
# ``(n)`` is the source grid axis, ``(k)`` the flattened neighbor index array,
# and ``(m)`` the destination axis. Output is float64 regardless of input
# dtype, matching the behaviour of the generic path below.
_GUFUNC_SIGNATURES = [
    "void(float64[:], int64[:], int64[:], int64[:], float64, float64[:])",
    "void(float32[:], int64[:], int64[:], int64[:], float64, float64[:])",
]
_GUFUNC_LAYOUT = "(n),(k),(m),(m),()->(m)"
_GUFUNC_KWARGS = {"nopython": True, "cache": True, "target": "parallel"}


def _make_kernel(reduce_fn):
    """Returns a kernel, compiled on its first call, that gathers each
    neighborhood, then calls ``reduce_fn(window, param)`` on the 1-D result.

    ``reduce_fn`` must be numba-compilable, and must be defined in a real
    source file for ``cache=True`` to find it.
    """
    # A reducer shared between kernels arrives already compiled; numba rejects
    # jitting a dispatcher twice.
    if not hasattr(reduce_fn, "py_func"):
        reduce_fn = njit(cache=True)(reduce_fn)

    @functools.cache
    def build():
        @guvectorize(_GUFUNC_SIGNATURES, _GUFUNC_LAYOUT, **_GUFUNC_KWARGS)
        def kernel(data, flat, starts, counts, param, out):
            widest = 0
            for i in range(counts.shape[0]):
                if counts[i] > widest:
                    widest = counts[i]
            buffer = np.empty(widest, dtype=np.float64)

            for i in range(starts.shape[0]):
                count = counts[i]
                if count == 0:
                    out[i] = np.nan
                    continue
                start = starts[i]
                for j in range(count):
                    buffer[j] = data[flat[start + j]]
                out[i] = reduce_fn(buffer[:count], param)

        return kernel

    def kernel(*args):
        return build()(*args)

    return kernel


# Reducers take ``(window, param)``; those without a parameter ignore the
# second argument. Numba keys its cache by code object rather than qualified
# name, so the identically-named lambdas below do not collide.
@njit(cache=True)
def _variance(window, ddof):
    """Variance with a delta degrees of freedom. Numba's ``np.var`` takes no
    ``ddof``, so the two-pass form is spelled out."""
    denominator = window.size - ddof
    if denominator <= 0:
        return np.nan
    center = np.mean(window)
    total = 0.0
    for value in window:
        total += (value - center) ** 2
    return total / denominator


@njit(cache=True)
def _median(window, _):
    """numba's ``np.median`` selects by partitioning, and whether a NaN survives
    that depends on where it lands -- so unlike numpy's, it propagates NaN
    only sometimes. This spelling short-circuits and allocates nothing:
    ``np.any(np.isnan(window))`` costs ~14% more, and routing through
    ``np.quantile``, which does propagate, costs 2.5x.
    """
    for value in window:
        if np.isnan(value):
            return np.nan
    return np.median(window)


def _as_quantile(q, scale: float):
    """Validates ``q`` on a 0-``scale`` scale and returns it as a 0-1 fraction."""
    value = float(q)
    if not 0.0 <= value <= scale:
        raise ValueError(f"`q` must be between 0 and {scale:g}, but got {q!r}.")
    return value / scale


def _csr_neighbors(grid, data_mapping: str, r: float):
    """Queries the neighborhood of every element and returns it in CSR form.

    ``query_radius`` returns a ragged sequence of index arrays, one per
    element. Flattening it into ``(flat, starts, counts)`` gives the kernels a
    layout they can walk without allocating per-neighborhood temporaries.

    Returns
    -------
    flat : np.ndarray
        Concatenated neighbor indices for every element.
    starts : np.ndarray
        Offset into ``flat`` at which each element's neighbors begin.
    counts : np.ndarray
        Number of neighbors of each element.
    """
    # Request a spherical/haversine tree explicitly rather than relying on the
    # defaults. Without this, a cartesian tree cached by an earlier call would
    # be reused and ``r`` would be silently interpreted as a chord length
    # instead of the great-circle degrees documented by the callers.
    coordinate_system = "spherical"
    tree = grid.get_ball_tree(
        coordinates=data_mapping,
        coordinate_system=coordinate_system,
        distance_metric="haversine",
    )

    dest_coords = _get_element_coords(grid, data_mapping, coordinate_system)
    neighbor_indices = tree.query_radius(dest_coords, r=r)

    # ``query_radius`` unwraps its result for a single query point, which a
    # one-element grid would hit.
    if isinstance(neighbor_indices, np.ndarray):
        neighbor_indices = [neighbor_indices]

    counts = np.fromiter(
        map(len, neighbor_indices), dtype=np.int64, count=len(neighbor_indices)
    )
    starts = np.zeros(counts.size, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    flat = np.concatenate(neighbor_indices).astype(np.int64, copy=False)

    return flat, starts, counts


def _neighborhood_reduce(block, flat, starts, counts, func: Callable):
    """Generic fallback: applies ``func`` to each neighborhood in turn.

    Used when ``func`` has no compiled kernel. ``block`` is a NumPy array with
    the grid dimension last.
    """
    destination_data = np.full(block.shape, np.nan)

    # The `axis` check lives outside the loop: whether `func` accepts the
    # keyword cannot change between iterations, so validating it once is
    # equivalent to validating it every time and leaves the loop body bare.
    try:
        for i in range(starts.shape[0]):
            idx = flat[starts[i] : starts[i] + counts[i]]
            # Apply func along the last (grid) axis only, so any extra leading
            # dimensions (e.g. time) are preserved rather than being collapsed.
            destination_data[..., i] = func(block[..., idx], axis=-1)
    except TypeError as exc:
        if "axis" not in str(exc):
            raise
        raise TypeError(
            f"`func` must accept an `axis` keyword argument so that the "
            f"reduction is applied over the neighborhood only, but "
            f"{getattr(func, '__name__', func)!r} does not. Use a NumPy "
            f"reduction such as `np.mean` or `np.median`, or wrap your "
            f"function with `functools.partial` to supply `axis`."
        ) from exc

    return destination_data


def _rechunk_grid_dim(uxda, grid_dim: str):
    """Collapses the grid dimension to a single chunk, warning if that changes
    the user's chunking.

    A neighborhood is global — an element near a chunk boundary draws on
    elements in other chunks — so the grid dimension cannot be chunked. This is
    done explicitly rather than through ``allow_rechunk``, which would do it
    silently and also disable ``apply_gufunc``'s other consistency checks.
    """
    if uxda.chunks is None:
        return uxda

    grid_chunks = uxda.chunksizes.get(grid_dim, ())
    if len(grid_chunks) <= 1:
        return uxda

    warnings.warn(
        f"Rechunking {grid_dim!r} from {len(grid_chunks)} chunks into one, as a "
        f"neighborhood may span the whole grid. Each task will hold "
        f"{uxda.sizes[grid_dim]} elements along {grid_dim!r}; chunk the "
        f"non-grid dimensions instead to bound memory use.",
        UserWarning,
        stacklevel=3,
    )

    return uxda.chunk({grid_dim: -1})


ELEMENT_DIMS = {
    "nodes": "n_node",
    "edge centers": "n_edge",
    "face centers": "n_face",
}


class Neighborhood:
    """The set of grid elements within a radius ``r`` of every element of one
    grid location, ready to be reduced over.

    Building this queries a ``BallTree`` once, which is by far the dominant
    cost of a neighborhood reduction — typically far more than the reduction
    itself. Holding onto the result lets several reductions, or several
    variables, share that one query instead of repeating it.

    Each reduction is a method: :meth:`mean`, :meth:`std`, :meth:`percentile`
    and so on run a compiled kernel, and :meth:`reduce` takes an arbitrary
    callable for anything without one. Every one takes the data to reduce as
    its first argument, so several variables can share one query.

    Parameters
    ----------
    grid : Grid
        Grid whose elements define the neighborhood.
    r : float, default=1.
        Radius of the neighborhood, in degrees of great-circle distance.
    on : str, default="face centers"
        Grid location the neighborhood is built around: "nodes",
        "edge centers", or "face centers".

    Examples
    --------
    >>> import uxarray as ux
    >>> uxds = ux.tutorial.open_dataset("outCSne30-vortex")  # doctest: +SKIP
    >>> nb = uxds.uxgrid.neighborhood(r=5.0)  # doctest: +SKIP
    >>> smooth = nb.mean(uxds["psi"])  # doctest: +SKIP
    >>> spread = nb.std(uxds["psi"], ddof=1)  # doctest: +SKIP

    See Also
    --------
    UxDataArray.neighborhood : Neighborhood around a single variable's elements.
    UxDataset.neighborhood : Neighborhood around every variable in a dataset.
    """

    def __init__(self, grid, r: float = 1.0, on: str = "face centers"):
        if on not in ELEMENT_DIMS:
            raise ValueError(
                f"Invalid `on`. Expected one of {', '.join(sorted(ELEMENT_DIMS))}, "
                f"but received {on!r}."
            )

        self._grid = grid
        self._r = float(r)
        self._on = on
        self._flat, self._starts, self._counts = _csr_neighbors(grid, on, self._r)

    @property
    def grid(self):
        """Grid the neighborhood was built from."""
        return self._grid

    @property
    def r(self) -> float:
        """Neighborhood radius, in degrees."""
        return self._r

    @property
    def on(self) -> str:
        """Grid location the neighborhood is centered on."""
        return self._on

    @property
    def grid_dim(self) -> str:
        """Name of the grid dimension this reduces over."""
        return ELEMENT_DIMS[self._on]

    @property
    def n_neighbors(self) -> xr.DataArray:
        """Number of elements in each neighborhood, itself a grid-mapped field.

        Useful for seeing how a fixed radius samples a variable-resolution
        mesh, where the count varies by region.
        """
        return xr.DataArray(
            self._counts.copy(),
            dims=[self.grid_dim],
            name="n_neighbors",
            attrs={"long_name": f"elements within {self._r} degrees"},
        )

    def __repr__(self) -> str:
        return (
            f"<Neighborhood on={self._on!r} r={self._r} "
            f"n_elements={self._counts.size} "
            f"neighbors_per_element=[{self._counts.min()}, {self._counts.max()}]>"
        )

    # One compiled kernel per reduction. The methods below call into these
    # directly. Non-compiled functions are only provided hooks through
    # ``reduce``. If new compiled reductions are desired, they should follow
    # this pattern.
    #
    # ``_make_kernel`` defers each build to the kernel's first call. The
    # deferred compilation ensures that these kernels will only be compiled
    # individually and lazily. Further, the lazy compilation prevents gufuncs
    # from spawning threadpools eagerly and disrupting threading and forking in
    # other contexts. They are wrapped in ``staticmethod`` because a plain
    # function in a class body would bind ``self`` as the kernel's first
    # argument.

    _mean_kernel = staticmethod(_make_kernel(lambda window, _: np.mean(window)))
    _sum_kernel = staticmethod(_make_kernel(lambda window, _: np.sum(window)))
    _min_kernel = staticmethod(_make_kernel(lambda window, _: np.min(window)))
    _max_kernel = staticmethod(_make_kernel(lambda window, _: np.max(window)))
    _ptp_kernel = staticmethod(_make_kernel(lambda window, _: np.max(window) - np.min(window)))
    _median_kernel = staticmethod(_make_kernel(_median))
    _var_kernel = staticmethod(_make_kernel(_variance))
    _std_kernel = staticmethod(_make_kernel(lambda window, ddof: np.sqrt(_variance(window, ddof))))

    # ``percentile`` is ``quantile`` on a 0-100 scale, so both methods
    # rescale onto this one kernel rather than compiling a near-duplicate.
    _quantile_kernel = staticmethod(_make_kernel(lambda window, q: np.quantile(window, q)))

    def mean(self, uxda):
        """Mean of each neighborhood."""
        return self._apply_kernel(uxda, self._mean_kernel, 0.0)

    def sum(self, uxda):
        """Sum of each neighborhood."""
        return self._apply_kernel(uxda, self._sum_kernel, 0.0)

    def min(self, uxda):
        """Smallest value in each neighborhood."""
        return self._apply_kernel(uxda, self._min_kernel, 0.0)

    def max(self, uxda):
        """Largest value in each neighborhood."""
        return self._apply_kernel(uxda, self._max_kernel, 0.0)

    def ptp(self, uxda):
        """Peak-to-peak spread (``max - min``) of each neighborhood."""
        return self._apply_kernel(uxda, self._ptp_kernel, 0.0)

    def median(self, uxda):
        """Median of each neighborhood."""
        return self._apply_kernel(uxda, self._median_kernel, 0.0)

    def var(self, uxda, ddof: int = 0):
        """Variance of each neighborhood, with ``ddof`` delta degrees of
        freedom."""
        return self._apply_kernel(uxda, self._var_kernel, float(ddof))

    def std(self, uxda, ddof: int = 0):
        """Standard deviation of each neighborhood, with ``ddof`` delta degrees
        of freedom."""
        return self._apply_kernel(uxda, self._std_kernel, float(ddof))

    def quantile(self, uxda, q: float):
        """Quantile ``q`` (between 0 and 1) of each neighborhood."""
        return self._apply_kernel(uxda, self._quantile_kernel, _as_quantile(q, 1.0))

    def percentile(self, uxda, q: float):
        """Percentile ``q`` (between 0 and 100) of each neighborhood."""
        return self._apply_kernel(uxda, self._quantile_kernel, _as_quantile(q, 100.0))

    def reduce(self, uxda, func: Callable):
        """Reduces each neighborhood with an arbitrary callable.

        This is the escape hatch for reductions with no method of their own.
        ``func`` is applied as ``func(values, axis=-1)`` over a block whose last
        axis is the neighborhood, once per grid element, in Python — which is
        considerably slower than the compiled methods above. Prefer a method
        where one exists.

        Parameters
        ----------
        uxda : UxDataArray
            Data to reduce.
        func : Callable
            Reduction to apply. Must accept an ``axis`` keyword argument. Use
            ``functools.partial`` to bind any further arguments.

        Returns
        -------
        UxDataArray
            Reduced data as float64, with the input's dimension order. Lazy if
            the input was lazy.

        Examples
        --------
        >>> from scipy.stats import skew  # doctest: +SKIP
        >>> nb.reduce(uxds["psi"], skew)  # doctest: +SKIP
        """

        def run(block, arrays):
            return _neighborhood_reduce(block, *arrays, func)

        return self._apply(uxda, run)

    def _apply_kernel(self, uxda, kernel, param: float):
        """Runs a compiled ``kernel`` over every neighborhood."""

        def run(block, arrays):
            # The kernels are compiled for float32/float64 only; anything
            # else (integer fields, say) is promoted, which the generic
            # path does too by writing into a float64 output.
            if block.dtype not in (np.float64, np.float32):
                block = block.astype(np.float64)
            return kernel(block, *arrays, param)

        return self._apply(uxda, run)

    def _apply(self, uxda, run):
        """Validates ``uxda`` against this neighborhood and maps ``run`` over
        it, one NumPy block at a time with the grid dimension last."""
        # Local import: uxarray.core.dataarray imports this module.
        from uxarray.core.dataarray import UxDataArray
        from uxarray.errors import DataCenteringError

        grid_dim = self.grid_dim
        if grid_dim not in uxda.dims:
            raise DataCenteringError(
                f"This neighborhood is built on {self._on!r} and reduces over "
                f"{grid_dim!r}, but the data has dimensions {tuple(uxda.dims)!r}."
            )
        if uxda.sizes[grid_dim] != self._counts.size:
            raise DataCenteringError(
                f"Data has {uxda.sizes[grid_dim]} elements along {grid_dim!r}, but "
                f"this neighborhood describes {self._counts.size}. The data is "
                f"probably mapped to a different grid."
            )

        arrays = (self._flat, self._starts, self._counts)

        def _apply(block):
            return run(block, arrays)

        work = _rechunk_grid_dim(uxda, grid_dim)

        # ``apply_ufunc`` moves the grid dimension last before calling
        # ``_apply`` and, for dask-backed input, hands each chunk over as a
        # materialized NumPy block. Indexing the array one destination element
        # at a time would instead trigger one graph execution per grid element.
        filtered = xr.apply_ufunc(
            _apply,
            work,
            input_core_dims=[[grid_dim]],
            output_core_dims=[[grid_dim]],
            dask="parallelized",
            output_dtypes=[np.float64],
            keep_attrs=True,
        )

        # Core dimensions come back appended last, so restore the input order.
        if filtered.dims != uxda.dims:
            filtered = filtered.transpose(*uxda.dims)

        # ``apply_ufunc`` returns a plain xr.DataArray, dropping the subclass
        # and its grid. Name, coords and attrs are carried through already.
        return UxDataArray(filtered, uxgrid=getattr(uxda, "uxgrid", self._grid))


class _BoundNeighborhoodReductions:
    """The reduction vocabulary of a :class:`Neighborhood` whose data is already
    supplied, spelled once for every class that carries it.

    A reduction has to be an attribute of ``Neighborhood``, which takes its data
    as an argument, and of the classes below, which already hold theirs. Those
    two signatures cannot share a definition, so the vocabulary is spelled twice
    in all -- but only twice, and only one of the two chooses a kernel.

    This is the half that does not. Each method here hands ``_map`` the
    ``Neighborhood`` method it stands for, as the function object itself rather
    than a name to look up later, and subclasses implement ``_map`` to say
    *which data* to call it on. That is the only thing that differs between a
    neighborhood bound to one variable and one bound to a whole dataset, so
    everything else is written once.

    Passing the method rather than a name is what keeps the two halves honest:
    the reference is resolved when this class is created, so a reduction that
    ``Neighborhood`` does not define cannot be spelled here at all, and a bound
    method cannot reach a different kernel, or a different ``ddof``, than the
    unbound one it names.
    """

    def _map(self, reduction: Callable, *args, **kwargs):
        """Calls ``reduction(neighborhood, uxda, *args, **kwargs)`` on the data
        this is bound to, returning the result in the same container.

        Subclasses must implement this; it is the only thing they need to.
        """
        raise NotImplementedError

    def mean(self):
        """Mean of each neighborhood."""
        return self._map(Neighborhood.mean)

    def sum(self):
        """Sum of each neighborhood."""
        return self._map(Neighborhood.sum)

    def min(self):
        """Smallest value in each neighborhood."""
        return self._map(Neighborhood.min)

    def max(self):
        """Largest value in each neighborhood."""
        return self._map(Neighborhood.max)

    def ptp(self):
        """Peak-to-peak spread (``max - min``) of each neighborhood."""
        return self._map(Neighborhood.ptp)

    def median(self):
        """Median of each neighborhood."""
        return self._map(Neighborhood.median)

    def var(self, ddof: int = 0):
        """Variance of each neighborhood, with ``ddof`` delta degrees of
        freedom."""
        return self._map(Neighborhood.var, ddof=ddof)

    def std(self, ddof: int = 0):
        """Standard deviation of each neighborhood, with ``ddof`` delta degrees
        of freedom."""
        return self._map(Neighborhood.std, ddof=ddof)

    def quantile(self, q: float):
        """Quantile ``q`` (between 0 and 1) of each neighborhood."""
        return self._map(Neighborhood.quantile, q)

    def percentile(self, q: float):
        """Percentile ``q`` (between 0 and 100) of each neighborhood."""
        return self._map(Neighborhood.percentile, q)

    def reduce(self, func: Callable):
        """Reduces each neighborhood with an arbitrary callable.

        See :meth:`Neighborhood.reduce`, which this supplies the data to.
        """
        return self._map(Neighborhood.reduce, func)


class DataArrayNeighborhood(_BoundNeighborhoodReductions):
    """Neighborhood of radius ``r`` around the elements of one data variable,
    ready to be reduced over.

    Carries the same reduction methods as :class:`Neighborhood`, with the data
    already supplied, so they take only the parameters of the reduction itself.
    The underlying neighbor query is built once and reused by every reduction
    called on this object.

    Returned by :meth:`UxDataArray.neighborhood`; not constructed directly.

    Examples
    --------
    >>> import uxarray as ux
    >>> uxds = ux.tutorial.open_dataset("outCSne30-vortex")  # doctest: +SKIP
    >>> nb = uxds["psi"].neighborhood(r=5.0)  # doctest: +SKIP
    >>> smooth, spread = nb.mean(), nb.std(ddof=1)  # doctest: +SKIP

    See Also
    --------
    Neighborhood : The unbound neighborhood this wraps.
    """

    def __init__(self, neighborhood: Neighborhood, uxda):
        self._neighborhood = neighborhood
        self._uxda = uxda

    @property
    def neighborhood(self) -> Neighborhood:
        """The unbound :class:`Neighborhood` these reductions run on.

        Reusable across other variables at the same grid location, without
        repeating the neighbor query.
        """
        return self._neighborhood

    @property
    def grid(self):
        """Grid the neighborhood was built from."""
        return self._neighborhood.grid

    @property
    def r(self) -> float:
        """Neighborhood radius, in degrees."""
        return self._neighborhood.r

    @property
    def on(self) -> str:
        """Grid location the neighborhood is centered on."""
        return self._neighborhood.on

    @property
    def grid_dim(self) -> str:
        """Name of the grid dimension this reduces over."""
        return self._neighborhood.grid_dim

    @property
    def n_neighbors(self) -> xr.DataArray:
        """Number of elements in each neighborhood."""
        return self._neighborhood.n_neighbors

    def __repr__(self) -> str:
        return (
            f"<DataArrayNeighborhood of {self._uxda.name!r} on {self._neighborhood!r}>"
        )

    def _map(self, reduction: Callable, *args, **kwargs):
        """Applies ``reduction`` to the one variable this is bound to."""
        return reduction(self._neighborhood, self._uxda, *args, **kwargs)


class DatasetNeighborhood(_BoundNeighborhoodReductions):
    """Neighborhood of radius ``r`` around every grid-mapped variable of a
    dataset, ready to be reduced over.

    Carries the same reduction methods as :class:`Neighborhood`, applying each
    to every data variable at once. Variables without a grid dimension pass
    through unchanged, and variables mapped to the same grid location share one
    neighbor query, so a reduction costs one query per location present rather
    than one per variable.

    Returned by :meth:`UxDataset.neighborhood`; not constructed directly.

    Examples
    --------
    >>> import uxarray as ux
    >>> uxds = ux.tutorial.open_dataset("outCSne30-vortex")  # doctest: +SKIP
    >>> smooth = uxds.neighborhood(r=5.0).mean()  # doctest: +SKIP

    See Also
    --------
    Neighborhood : The single-location neighborhood this builds on.
    """

    def __init__(self, uxds, r: float = 1.0):
        self._uxds = uxds
        self._r = float(r)
        # Built on first use, keyed by grid location, since the query depends
        # only on (grid, location, radius) -- not on the data.
        self._by_location: dict[str, Neighborhood] = {}

    @property
    def r(self) -> float:
        """Neighborhood radius, in degrees."""
        return self._r

    def __repr__(self) -> str:
        return (
            f"<DatasetNeighborhood r={self._r} data_vars={len(self._uxds.data_vars)}>"
        )

    def _map(self, reduction: Callable, *args, **kwargs):
        """Applies ``reduction`` to every grid-mapped variable, sharing a
        neighbor query between variables at the same grid location."""
        destination_uxds = self._uxds._copy()

        for var_name in self._uxds.data_vars:
            uxda = self._uxds[var_name]

            # Skip variables that are not mapped to a grid element.
            if not any(dim in GRID_DIMS for dim in uxda.dims):
                continue

            location = uxda._neighborhood_location("neighborhood")
            if location not in self._by_location:
                self._by_location[location] = Neighborhood(
                    self._uxds.uxgrid, r=self._r, on=location
                )

            # The Neighborhood methods restore the input dimension order, so
            # it is always preserved.
            destination_uxds[var_name] = reduction(
                self._by_location[location], uxda, *args, **kwargs
            )

        return destination_uxds
