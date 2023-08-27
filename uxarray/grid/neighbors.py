import numpy as np
from numpy import deg2rad
import warnings

from sklearn.neighbors import BallTree as SKBallTree
from typing import Optional, Union

from uxarray.constants import INT_DTYPE


class BallTree:
    """Custom BallTree datastructure written around the
    ``sklearn.neighbors.BallTree`` implementation for use with either the
    corner (``Mesh2_node_x``, ``Mesh2_node_y``) or center (``Mesh2_face_x``,
    ``Mesh2_face_y``) nodes of the inputted unstructured grid.

    Notes
    -----
    See `sklearn.neighbors.BallTree <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html>`__
    for further information about the wrapped data structure.
    """

    def __init__(self, grid, node_type='corner'):

        # construct tree from corner nodes
        if node_type == "corner":

            XY = np.vstack((deg2rad(grid.Mesh2_node_y.values),
                            deg2rad(grid.Mesh2_node_x.values))).T

            self.n_elements = grid.nMesh2_node
            self.tree = SKBallTree(XY, metric='haversine')

        # construct tree from center nodes
        elif node_type == "center":

            # warning until we work on how face centers are represented and constructed
            warnings.warn(
                "Internal representation of face center coordinates has not been tested or verified. Results"
                "may not be as expected until this has been patched in the Grid class."
            )

            # center nodes must be present, remove once construction methods for them is added
            if grid.Mesh2_face_x is None or grid.Mesh2_face_y is None:
                raise ValueError

            XY = np.vstack((deg2rad(grid.Mesh2_face_y.values),
                            deg2rad(grid.Mesh2_face_x.values))).T

            self.n_elements = grid.nMesh2_face
            self.tree = SKBallTree(XY, metric='haversine')

    def query(self,
              xy: Union[np.ndarray, list, tuple],
              k: Optional[int] = 1,
              use_radians: Optional[bool] = False,
              return_distance: Optional[bool] = True,
              dualtree: Optional[bool] = False,
              breadth_first: Optional[bool] = False,
              sort_results: Optional[bool] = True):
        """Query the tree for the ``k`` nearest neighbors.

        Parameters
        ----------
        xy : array_like
            coordinate pairs in degrees (lon, lat) to query
        k: int, default=1
            The number of nearest neighbors to return
        use_radians : bool, optional
            if True, queries assuming xy are inputted in radians, not degrees
        return_distance : bool, optional
            if True, return a tuple ``(d, i)`` of distances and indices if False, return array i
        dualtree : bool, default=False
            if True, use the dual tree formalism for the query: a tree is built for the query points, and the pair of trees is used to efficiently search this space. This can lead to better performance as the number of points grows large.
        breadth_first : bool, default=False
            if True, then query the nodes in a breadth-first manner. Otherwise, query the nodes in a depth-first manner.
        sort_results : bool, default=True
            if True, then distances and indices of each point are sorted on return, so that the first column contains the closest points. Otherwise, neighbors are returned in an arbitrary order.

        Returns
        -------
        d : ndarray of shape (xy.shape[0], k), dtype=double
            Each entry gives the list of distances to the neighbors of the corresponding point.
        ind : ndarray of shape (xy.shape[0], k), dtype=INT_DTYPE
            Each entry gives the list of indices of neighbors of the corresponding point.
        """

        if k < 1 or k > self.n_elements:
            raise AssertionError(
                f"The value of k must be greater than 1 and less than the number of elements used to construct "
                f"the tree ({self.n_elemetns}).")

        xy = _prepare_xy_for_query(xy, use_radians)

        # perform query with distance
        if return_distance:
            d, ind = self.tree.query(xy, k, return_distance, dualtree,
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
            ind = self.tree.query(xy, k, return_distance, dualtree,
                                  breadth_first, sort_results)

            ind = np.asarray(ind, dtype=INT_DTYPE)

            if xy.shape[0] == 1:
                ind = ind.squeeze()

            return ind

    def query_radius(self,
                     xy: Union[np.ndarray, list, tuple],
                     r: Optional[int] = 1.0,
                     use_radians: Optional[bool] = False,
                     return_distance: Optional[bool] = True,
                     count_only: Optional[bool] = False,
                     sort_results: Optional[bool] = False):
        """Queries the tree for neighbors within a radius ``r``.

        Parameters
        ----------
        XY : array_like
           coordinate pairs in degrees (lon, lat) to query
        r: distance in degrees within which neighbors are returned
            r can be a single value , or an array of values of shape x.shape[:-1] if different radii are desired for each point.
        use_radians : bool, default=True
            if True, queries assuming xy are inputted in radians, not degrees
        return_distance : bool, default=False
            if True, return distances to neighbors of each point if False, return only neighbors Note that unlike the query() method, setting return_distance=True here adds to the computation time. Not all distances need to be calculated explicitly for return_distance=False. Results are not sorted by default: see sort_results keyword.
        count_only : bool, default=False
            if True, return only the count of points within distance r if False, return the indices of all points within distance r If return_distance==True, setting count_only=True will result in an error.
        sort_results : bool, default=False
           if True, the distances and indices will be sorted before being returned. If False, the results will not be sorted. If return_distance == False, setting sort_results = True will result in an error.


        Returns
        -------
        d : ndarray of shape (xy.shape[0], k), dtype=double
            Each entry gives the list of distances to the neighbors of the corresponding point.
        ind : ndarray of shape (xy.shape[0], k), dtype=INT_DTYPE
            Each entry gives the list of indices of neighbors of the corresponding point.
        """

        if r < 0.0:
            raise AssertionError(
                f"The value of r must be greater than or equal to zero.")

        r = np.deg2rad(r)
        xy = _prepare_xy_for_query(xy, use_radians)

        if count_only:
            count = self.tree.query_radius(xy, r, return_distance, count_only,
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
            ind, d = self.tree.query_radius(xy, r, return_distance, count_only,
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
