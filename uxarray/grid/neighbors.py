import numpy as np
from numpy import deg2rad
import warnings

from sklearn.neighbors import BallTree as SKBallTree
from typing import Optional, Union

from uxarray.constants import INT_DTYPE


class BallTree:
    """Custom BallTree data structure written around the
    ``sklearn.neighbors.BallTree`` implementation for use with either the
    corner (``Mesh2_node_x``, ``Mesh2_node_y``) or center (``Mesh2_face_x``,
    ``Mesh2_face_y``) nodes of the inputted unstructured grid.

    Parameters
    ----------
    grid : ux.Grid
        Source grid used to construct the BallTree
    tree_type : str, default="nodes"
            Identifies which tree to construct or select, with "nodes" selecting the Corner Nodes and "face centers" selecting the Face
            Centers of each face
    distance_metric : str, default="haversine"
        Distance metric used to construct the BallTree

    Notes
    -----
    See `sklearn.neighbors.BallTree <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html>`__
    for further information about the wrapped data structures.
    """

    def __init__(self,
                 grid,
                 tree_type: Optional[str] = "nodes",
                 distance_metric='haversine'):

        # maintain a reference to the source grid
        self._source_grid = grid
        self.distance_metric = distance_metric
        self._tree_type = tree_type

        self._tree_from_nodes = None
        self._tree_from_face_centers = None

        # set up appropriate reference to tree
        if tree_type == "nodes":
            self._tree_from_nodes = self._build_from_nodes()
            self._n_elements = self._source_grid.nMesh2_node
        elif tree_type == "face centers":
            self._tree_from_face_centers = self._build_from_face_centers()
            self._n_elements = self._source_grid.nMesh2_face
        else:
            raise ValueError

    def _build_from_face_centers(self):
        """Internal``sklearn.neighbors.BallTree`` constructed from face
        centers."""
        if self._tree_from_face_centers is None:
            if self._source_grid.Mesh2_face_x is None:
                raise ValueError

            latlon = np.vstack(
                (deg2rad(self._source_grid.Mesh2_face_y.values),
                 deg2rad(self._source_grid.Mesh2_face_x.values))).T

            self._tree_from_face_centers = SKBallTree(
                latlon, metric=self.distance_metric)

        return self._tree_from_face_centers

    def _build_from_nodes(self):
        """Internal``sklearn.neighbors.BallTree`` constructed from corner
        nodes."""
        if self._tree_from_nodes is None:
            latlon = np.vstack(
                (deg2rad(self._source_grid.Mesh2_node_y.values),
                 deg2rad(self._source_grid.Mesh2_node_x.values))).T
            self._tree_from_nodes = SKBallTree(latlon,
                                               metric=self.distance_metric)

        return self._tree_from_nodes

    def _current_tree(self):

        _tree = None

        if self._tree_type == "nodes":
            _tree = self._tree_from_nodes
        elif self._tree_type == "face centers":
            _tree = self._tree_from_face_centers
        else:
            raise TypeError

        return _tree

    def query(self,
              xy: Union[np.ndarray, list, tuple],
              k: Optional[int] = 1,
              xy_in_radians: Optional[bool] = False,
              return_distance: Optional[bool] = True,
              dualtree: Optional[bool] = False,
              breadth_first: Optional[bool] = False,
              sort_results: Optional[bool] = True):
        """Queries the tree for the ``k`` nearest neighbors.

        Parameters
        ----------
        xy : array_like
            coordinate pairs in degrees (lon, lat) to query
        k: int, default=1
            The number of nearest neighbors to return
        xy_in_radians : bool, optional
            if True, queries assuming xy are inputted in radians, not degrees
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
        d : ndarray of shape (xy.shape[0], k), dtype=double
            Distance array that keeps the distances of the k-nearest neighbors to the entries from xy in each row
        ind : ndarray of shape (xy.shape[0], k), dtype=INT_DTYPE
            Index array that keeps the indices of the k-nearest neighbors to the entries from xy in each row
        """

        if k < 1 or k > self._n_elements:
            raise AssertionError(
                f"The value of k must be greater than 1 and less than the number of elements used to construct "
                f"the tree ({self._n_elements}).")

        xy = _prepare_xy_for_query(xy, xy_in_radians)

        d, ind = self._current_tree().query(xy, k, return_distance, dualtree,
                                            breadth_first, sort_results)

        ind = np.asarray(ind, dtype=INT_DTYPE)

        if xy.shape[0] == 1:
            ind = ind.squeeze()

        # perform query with distance
        if return_distance:
            # only one pair was queried
            if xy.shape[0] == 1:
                d = d.squeeze()

            if not xy_in_radians:
                d = np.rad2deg(d)

            return d, ind

        return ind

    def query_radius(self,
                     xy: Union[np.ndarray, list, tuple],
                     r: Optional[int] = 1.0,
                     xy_in_radians: Optional[bool] = False,
                     return_distance: Optional[bool] = True,
                     count_only: Optional[bool] = False,
                     sort_results: Optional[bool] = False):
        """Queries the tree for all neighbors within a radius ``r``.

        Parameters
        ----------
        xy : array_like
           coordinate pairs in degrees (lon, lat) to query
        r: distance in degrees within which neighbors are returned
            r can be a single value , or an array of values of shape x.shape[:-1] if different radii are desired for each point.
        xy_in_radians : bool, optional
            if True, queries assuming xy are inputted in radians, not degrees
        return_distance : bool, default=False
            Indicates whether distances should be returned
        count_only : bool, default=False
            Indicates whether only counts should be returned
        sort_results : bool, default=False
            Indicates whether distances should be sorted

        Returns
        -------
        d : ndarray of shape (xy.shape[0], k), dtype=double
            Distance array that keeps the distances of all neighbors within some radius to the entries from xy in each row
        ind : ndarray of shape (xy.shape[0], k), dtype=INT_DTYPE
            Index array that keeps the indices of all neighbors within some radius to the entries from xy in each row
        """

        if r < 0.0:
            raise AssertionError(
                f"The value of r must be greater than or equal to zero.")

        r = np.deg2rad(r)
        xy = _prepare_xy_for_query(xy, xy_in_radians)

        if count_only:
            count = self._current_tree().query_radius(xy, r, return_distance,
                                                      count_only, sort_results)

            return count

        else:

            ind, d = self._current_tree().query_radius(xy, r, return_distance,
                                                       count_only, sort_results)

            ind = np.asarray(ind[0], dtype=INT_DTYPE)

            if xy.shape[0] == 1:
                ind = ind.squeeze()

            if return_distance:
                if not xy_in_radians:
                    d = np.rad2deg(d[0])

                return d, ind

            return ind

    @property
    def tree_type(self):
        return self._tree_type

    @tree_type.setter
    def tree_type(self, value):
        self._tree_type = value

        # set up appropriate reference to tree
        if self._tree_type == "nodes":
            if self._tree_from_nodes is None:
                self._tree_from_nodes = self._build_from_nodes()
            self._n_elements = self._source_grid.nMesh2_node
        elif self._tree_type == "face centers":
            if self._tree_from_face_centers is None:
                self._tree_from_face_centers = self._build_from_face_centers()
            self._n_elements = self._source_grid.nMesh2_face
        else:
            raise ValueError


def _prepare_xy_for_query(xy, use_radians):
    """Prepares xy coordinates for query with the sklearn BallTree."""

    xy = np.asarray(xy)

    # expand if only a single node pair is provided
    if xy.ndim == 1:
        xy = np.expand_dims(xy, axis=0)

    # expected shape is [n_pairs, 2]
    if xy.shape[1] == 3:
        raise AssertionError(
            f"The dimension of each coordinate pair must be two (lon, lat). Did you attempt to query using Cartesian "
            f"(x, y, z) coordinates?")

    if xy.shape[1] != 2:
        raise AssertionError(
            f"The dimension of each coordinate pair must be two (lon, lat).)")

    # swap X and Y for query
    xy[:, [0, 1]] = xy[:, [1, 0]]

    # balltree expects units in radians for query
    if not use_radians:
        xy = np.deg2rad(xy)

    return xy
