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

    Notes
    -----
    See `sklearn.neighbors.BallTree <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html>`__
    for further information about the wrapped data structures.
    """

    def __init__(self, grid, distance_metric='haversine'):

        # maintain a reference to the source grid
        self._source_grid = grid

        self._corner_node_ball_tree = None
        self._face_center_ball_tree = None

        self.n_node = grid.nMesh2_node
        self.n_face = grid.nMesh2_face

        self.distance_metric = distance_metric

    @property
    def corner_node_ball_tree(self):
        """Internal``sklearn.neighbors.BallTree`` constructed from corner
        nodes."""
        if self._corner_node_ball_tree is None:
            latlon = np.vstack(
                (deg2rad(self._source_grid.Mesh2_node_y.values),
                 deg2rad(self._source_grid.Mesh2_node_x.values))).T
            self._corner_node_ball_tree = SKBallTree(
                latlon, metric=self.distance_metric)
        return self._corner_node_ball_tree

    @property
    def face_center_ball_tree(self):
        """Internal``sklearn.neighbors.BallTree`` constructed from face
        centers."""
        if self._face_center_ball_tree is None:
            if self._source_grid.Mesh2_face_x is None:
                raise ValueError
            latlon = np.vstack(
                (deg2rad(self._source_grid.Mesh2_face_y.values),
                 deg2rad(self._source_grid.Mesh2_face_x.values))).T

            self._face_center_ball_tree = SKBallTree(
                latlon, metric=self.distance_metric)
        return self._face_center_ball_tree

    def query(self,
              xy: Union[np.ndarray, list, tuple],
              k: Optional[int] = 1,
              tree_type: Optional[str] = "corner nodes",
              use_radians: Optional[bool] = False,
              return_distance: Optional[bool] = True,
              dualtree: Optional[bool] = False,
              breadth_first: Optional[bool] = False,
              sort_results: Optional[bool] = True):
        """Queries the tree defined by ``tree_type`` ("corner nodes" or "face
        centers") for the ``k`` nearest neighbors.

        Parameters
        ----------
        xy : array_like
            coordinate pairs in degrees (lon, lat) to query
        k: int, default=1
            The number of nearest neighbors to return
        tree_type: str, default="corner nodes"
            Specifies which tree to use, "corner nodes" or "face centers"
        use_radians : bool, optional
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

        # set up appropriate reference to tree
        if tree_type == "corner nodes":
            _tree = self.corner_node_ball_tree
            _n_elements = self.n_node
        elif tree_type == "face centers":
            _tree = self.face_center_ball_tree
            _n_elements = self.n_face
        else:
            raise ValueError

        if k < 1 or k > _n_elements:
            raise AssertionError(
                f"The value of k must be greater than 1 and less than the number of elements used to construct "
                f"the tree ({_n_elements}).")

        xy = _prepare_xy_for_query(xy, use_radians)

        # perform query with distance
        if return_distance:
            d, ind = _tree.query(xy, k, return_distance, dualtree,
                                 breadth_first, sort_results)

            ind = np.asarray(ind, dtype=INT_DTYPE)

            # only one pair was queried
            if xy.shape[0] == 1:
                ind = ind.squeeze()
                d = d.squeeze()

            d = np.rad2deg(d)

            return d, ind

        # perform query without distance
        else:
            ind = _tree.query(xy, k, return_distance, dualtree, breadth_first,
                              sort_results)

            ind = np.asarray(ind, dtype=INT_DTYPE)

            if xy.shape[0] == 1:
                ind = ind.squeeze()

            return ind

    def query_radius(self,
                     xy: Union[np.ndarray, list, tuple],
                     r: Optional[int] = 1.0,
                     tree_type: Optional[str] = "corner nodes",
                     use_radians: Optional[bool] = False,
                     return_distance: Optional[bool] = True,
                     count_only: Optional[bool] = False,
                     sort_results: Optional[bool] = False):
        """Queries the tree defined by ``tree_type`` ("corner nodes" or "face
        centers") for all neighbors within a radius ``r``.

        Parameters
        ----------
        XY : array_like
           coordinate pairs in degrees (lon, lat) to query
        r: distance in degrees within which neighbors are returned
            r can be a single value , or an array of values of shape x.shape[:-1] if different radii are desired for each point.
        tree_type: str, default="corner nodes"
            Specifies which tree to use, "corner nodes" or "face centers"
        use_radians : bool, default=True
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

        # set up appropriate reference to tree
        if tree_type == "corner nodes":
            _tree = self._corner_node_ball_tree
        elif tree_type == "face centers":
            _n_elements = self.n_face
        else:
            raise ValueError

        if r < 0.0:
            raise AssertionError(
                f"The value of r must be greater than or equal to zero.")

        r = np.deg2rad(r)
        xy = _prepare_xy_for_query(xy, use_radians)

        if count_only:
            count = _tree.query_radius(xy, r, return_distance, count_only,
                                       sort_results)

            return count

        if not count_only and not return_distance:
            ind = self.tree.query_radius(xy, r, return_distance, count_only,
                                         sort_results)

            ind = np.asarray(ind, dtype=INT_DTYPE)

            if xy.shape[0] == 1:
                ind = ind.squeeze()

            return ind

        else:
            ind, d = _tree.query_radius(xy, r, return_distance, count_only,
                                        sort_results)

            ind = np.asarray(ind[0], dtype=INT_DTYPE)

            if xy.shape[0] == 1:
                ind = ind.squeeze()

            d = np.rad2deg(d[0])

            return d, ind


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
