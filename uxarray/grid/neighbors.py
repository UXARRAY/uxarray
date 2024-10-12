import numpy as np
from numpy import deg2rad

import xarray as xr

from numba import njit

from sklearn.neighbors import BallTree as SKBallTree
from sklearn.neighbors import KDTree as SKKDTree

from typing import Optional, Union

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


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
        coordinates: Optional[str] = "nodes",
        coordinate_system: Optional[str] = "cartesian",
        distance_metric: Optional[str] = "minkowski",
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
                raise TypeError(
                    f"Unknown coordinate_system, {self.coordinate_system}, use either 'cartesian' or "
                    f"'spherical'"
                )

            self._tree_from_nodes = SKKDTree(coords, metric=self.distance_metric)

        return self._tree_from_nodes

    def _build_from_face_centers(self):
        """Internal``sklearn.neighbors.KDTree`` constructed from face
        centers."""

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
            raise TypeError(
                f"Unknown coordinates location, {self._coordinates}, use either 'nodes', 'face centers', "
                f"or 'edge centers'"
            )

        return _tree

    def query(
        self,
        coords: Union[np.ndarray, list, tuple],
        k: Optional[int] = 1,
        return_distance: Optional[bool] = True,
        in_radians: Optional[bool] = False,
        dualtree: Optional[bool] = False,
        breadth_first: Optional[bool] = False,
        sort_results: Optional[bool] = True,
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
        coords: Union[np.ndarray, list, tuple],
        r: Optional[int] = 1.0,
        return_distance: Optional[bool] = False,
        in_radians: Optional[bool] = False,
        count_only: Optional[bool] = False,
        sort_results: Optional[bool] = False,
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
        coordinates: Optional[str] = "nodes",
        coordinate_system: Optional[str] = "spherical",
        distance_metric: Optional[str] = "haversine",
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
        coords: Union[np.ndarray, list, tuple],
        k: Optional[int] = 1,
        in_radians: Optional[bool] = False,
        return_distance: Optional[bool] = True,
        dualtree: Optional[bool] = False,
        breadth_first: Optional[bool] = False,
        sort_results: Optional[bool] = True,
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
        coords: Union[np.ndarray, list, tuple],
        r: Optional[int] = 1.0,
        in_radians: Optional[bool] = False,
        return_distance: Optional[bool] = False,
        count_only: Optional[bool] = False,
        sort_results: Optional[bool] = False,
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


def _prepare_xy_for_query(xy, use_radians, distance_metric):
    """Prepares xy coordinates for query with the sklearn BallTree or
    KDTree."""

    xy = np.asarray(xy)

    # expand if only a single node pair is provided
    if xy.ndim == 1:
        xy = np.expand_dims(xy, axis=0)

    # expected shape is [n_pairs, 2]
    if xy.shape[1] == 3:
        raise AssertionError(
            "The dimension of each coordinate pair must be two (lon, lat). Did you attempt to query using Cartesian "
            "(x, y, z) coordinates?"
        )

    if xy.shape[1] != 2:
        raise AssertionError(
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
        raise AssertionError(
            "The dimension of each coordinate pair must be three (x, y, z). Did you attempt to query using latlon "
            "(lat, lon) coordinates?"
        )

    if xyz.shape[1] != 3:
        raise AssertionError(
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
        grid.node_lon.values, grid.node_lat.values, grid.edge_face_connectivity.values
    )

    grid._ds["edge_face_distances"] = xr.DataArray(
        data=edge_face_distances,
        dims=["n_edge"],
        attrs={
            "long_name": "arc distance between the face centers that saddle each edge",
        },
    )


@njit(cache=True)
def _construct_edge_face_distances(node_lon, node_lat, edge_faces):
    """Helper for computing the arc-distance between faces that saddle a given
    edge."""

    saddle_mask = edge_faces[:, 1] != INT_FILL_VALUE

    edge_face_distances = np.zeros(edge_faces.shape[0])

    edge_lon_a = np.deg2rad((node_lon[edge_faces[saddle_mask, 0]]))
    edge_lon_b = np.deg2rad((node_lon[edge_faces[saddle_mask, 1]]))

    edge_lat_a = np.deg2rad((node_lat[edge_faces[saddle_mask, 0]]))
    edge_lat_b = np.deg2rad((node_lat[edge_faces[saddle_mask, 1]]))

    # arc length
    edge_face_distances[saddle_mask] = np.arccos(
        np.sin(edge_lat_a) * np.sin(edge_lat_b)
        + np.cos(edge_lat_a) * np.cos(edge_lat_b) * np.cos(edge_lon_a - edge_lon_b)
    )

    return edge_face_distances
