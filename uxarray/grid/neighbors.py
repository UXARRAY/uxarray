import numpy as np
from numpy import deg2rad
import warnings

from sklearn.neighbors import BallTree as SKBallTree
from sklearn.neighbors import KDTree as SKKDTree
from typing import Optional, Union

from uxarray.constants import INT_DTYPE


class KDTree:
    """Custom KDTree data structure written around the
    ``sklearn.neighbors.KDTree`` implementation for use with either the corner
    (``node_x``, ``node_y``, ``node_z``) or center (``face_x``, ``face_y``,
    ``face_z``) nodes of the inputted unstructured grid.

    Parameters
    ----------
    grid : ux.Grid
        Source grid used to construct the KDTree
    tree_type : str, default="nodes"
            Identifies which tree to construct or select, with "nodes" selecting the corner nodes and "face centers" selecting the face
            centers of each face
    distance_metric : str, default="minkowski"
        Distance metric used to construct the KDTree

    Notes
    -----
    See `sklearn.neighbors.KDTree <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html>`__
    for further information about the wrapped data structures.
    """

    def __init__(self,
                 grid,
                 tree_type: Optional[str] = "nodes",
                 distance_metric="minkowski"):

        # Set up references
        self._source_grid = grid
        self._tree_type = tree_type
        self.distance_metric = distance_metric

        self._tree_from_nodes = None
        self._tree_from_face_centers = None

        # Build the tree based on face centers or nodes
        if tree_type == "nodes":
            self._tree_from_nodes = self._build_from_nodes()
            self._n_elements = self._source_grid.n_node
        elif tree_type == "face centers":
            self._tree_from_face_centers = self._build_from_face_centers()
            self._n_elements = self._source_grid.n_face
        else:
            raise ValueError

    def _build_from_nodes(self):
        """Internal``sklearn.neighbors.KDTree`` constructed from corner
        nodes."""
        if self._tree_from_nodes is None:
            cart_coords = np.stack((self._source_grid.node_x.values,
                                    self._source_grid.node_y.values,
                                    self._source_grid.node_z.values),
                                   axis=-1)
            self._tree_from_nodes = SKKDTree(cart_coords,
                                             metric=self.distance_metric)

        return self._tree_from_nodes

    def _build_from_face_centers(self):
        """Internal``sklearn.neighbors.KDTree`` constructed from face
        centers."""
        if self._tree_from_face_centers is None:
            if self._source_grid.face_x is None:
                raise ValueError

            cart_coords = np.stack((self._source_grid.face_x.values,
                                    self._source_grid.face_y.values,
                                    self._source_grid.face_z.values),
                                   axis=-1)
            self._tree_from_face_centers = SKKDTree(cart_coords,
                                                    metric=self.distance_metric)

        return self._tree_from_face_centers

    def _current_tree(self):
        """Creates and returns the current tree."""
        _tree = None

        if self._tree_type == "nodes":
            _tree = self._tree_from_nodes
        elif self._tree_type == "face centers":
            _tree = self._tree_from_face_centers
        else:
            raise TypeError

        return _tree

    def query(self,
              xyz: Union[np.ndarray, list, tuple],
              k: Optional[int] = 1,
              return_distance: Optional[bool] = True,
              dualtree: Optional[bool] = False,
              breadth_first: Optional[bool] = False,
              sort_results: Optional[bool] = True):
        """Queries the tree for the ``k`` nearest neighbors.

        Parameters
        ----------
        xyz : array_like
            coordinate pairs in cartesian (x, y, z) to query
        k: int, default=1
            The number of nearest neighbors to return
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
        d : ndarray of shape (xyz.shape[0], k), dtype=double
            Distance array that keeps the distances of the k-nearest neighbors to the entries from xyz in each row
        ind : ndarray of shape (xyz.shape[0], k), dtype=INT_DTYPE
            Index array that keeps the indices of the k-nearest neighbors to the entries from xyz in each row
        """

        if k < 1 or k > self._n_elements:
            raise AssertionError(
                f"The value of k must be greater than 1 and less than the number of elements used to construct "
                f"the tree ({self._n_elements}).")

        xyz = _prepare_xyz_for_query(xyz)

        d, ind = self._current_tree().query(xyz, k, return_distance, dualtree,
                                            breadth_first, sort_results)

        ind = np.asarray(ind, dtype=INT_DTYPE)

        if xyz.shape[0] == 1:
            ind = ind.squeeze()

        # perform query with distance
        if return_distance:
            # only one pair was queried
            if xyz.shape[0] == 1:
                d = d.squeeze()

            return d, ind

        return ind

    def query_radius(self,
                     xyz: Union[np.ndarray, list, tuple],
                     r: Optional[int] = 1.0,
                     return_distance: Optional[bool] = True,
                     count_only: Optional[bool] = False,
                     sort_results: Optional[bool] = False):
        """Queries the tree for all neighbors within a radius ``r``.

        Parameters
        ----------
        xyz : array_like
           coordinate pairs in cartesian (x, y, z) to query
        r: distance within which neighbors are returned
            r is a single value for the radius of which to query
        return_distance : bool, default=False
            Indicates whether distances should be returned
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
                f"The value of r must be greater than or equal to zero.")

        xyz = _prepare_xyz_for_query(xyz)

        if count_only:
            count = self._current_tree().query_radius(xyz, r, return_distance,
                                                      count_only, sort_results)

            return count

        else:

            ind, d = self._current_tree().query_radius(xyz, r, return_distance,
                                                       count_only, sort_results)

            ind = np.asarray(ind[0], dtype=INT_DTYPE)

            if xyz.shape[0] == 1:
                ind = ind.squeeze()

            if return_distance:

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
            self._n_elements = self._source_grid.n_node
        elif self._tree_type == "face centers":
            if self._tree_from_face_centers is None:
                self._tree_from_face_centers = self._build_from_face_centers()
            self._n_elements = self._source_grid.n_face
        else:
            raise ValueError


class BallTree:
    """Custom BallTree data structure written around the
    ``sklearn.neighbors.BallTree`` implementation for use with either the
    corner (``node_lon``, ``node_lat``) or center (``face_lon``, ``face_lat``)
    nodes of the inputted unstructured grid.

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
            self._n_elements = self._source_grid.n_node
        elif tree_type == "face centers":
            self._tree_from_face_centers = self._build_from_face_centers()
            self._n_elements = self._source_grid.n_face
        else:
            raise ValueError

    def _build_from_face_centers(self):
        """Internal``sklearn.neighbors.BallTree`` constructed from face
        centers."""
        if self._tree_from_face_centers is None:
            if self._source_grid.node_lon is None:
                raise ValueError

            latlon = np.vstack((deg2rad(self._source_grid.node_lat.values),
                                deg2rad(self._source_grid.node_lon.values))).T

            self._tree_from_face_centers = SKBallTree(
                latlon, metric=self.distance_metric)

        return self._tree_from_face_centers

    def _build_from_nodes(self):
        """Internal``sklearn.neighbors.BallTree`` constructed from corner
        nodes."""
        if self._tree_from_nodes is None:
            latlon = np.vstack((deg2rad(self._source_grid.node_lat.values),
                                deg2rad(self._source_grid.node_lon.values))).T
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
            r is a single value for the radius of which to query
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
            self._n_elements = self._source_grid.n_node
        elif self._tree_type == "face centers":
            if self._tree_from_face_centers is None:
                self._tree_from_face_centers = self._build_from_face_centers()
            self._n_elements = self._source_grid.n_face
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


def _prepare_xyz_for_query(xyz):
    """Prepares xyz coordinates for query with the sklearn KDTree."""

    xyz = np.asarray(xyz)

    # expand if only a single node pair is provided
    if xyz.ndim == 1:
        xyz = np.expand_dims(xyz, axis=0)

    # expected shape is [n_pairs, 3]
    if xyz.shape[1] == 2:
        raise AssertionError(
            f"The dimension of each coordinate pair must be three (x, y, z). Did you attempt to query using latlon "
            f"(lat, lon) coordinates?")

    if xyz.shape[1] != 3:
        raise AssertionError(
            f"The dimension of each coordinate pair must be three (x, y, z).)")

    return xyz
