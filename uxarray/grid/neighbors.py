import numpy as np
from numpy import deg2rad

from sklearn.neighbors import BallTree
from typing import Any, Dict, Optional, Union

from uxarray.constants import INT_DTYPE


class CornerNodeBallTree:

    def __init__(self, grid):

        XY = np.vstack((deg2rad(grid.Mesh2_node_y.values),
                        deg2rad(grid.Mesh2_node_x.values))).T

        self.n_node = grid.nMesh2_node
        self.tree = BallTree(XY, metric='haversine')

    def query(self,
              xy: Union[np.ndarray, list, tuple],
              k: Optional[int] = 1,
              use_radians: Optional[bool] = False,
              return_distance: Optional[bool] = True,
              dualtree: Optional[bool] = False,
              breadth_first: Optional[bool] = False,
              sort_results: Optional[bool] = True):
        """TODO: Docstring

        Parameters
        ----------
        xy : array_like
            TODO
        k: int, optional
            TODO
        return_distance : bool, optional
            TODO
        dualtree : bool, optional
            TODO
        breadth_first : bool, optional
            TODO
        sort_results : bool, optional
            TODO
        use_radians : bool, optional
            TODO

        Returns
        -------
        d : np.ndarray
            TODO
        ind : np.ndarray
            TODO
        """

        if k < 1 or k > self.n_node:
            raise ValueError  # TODO

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
                     xy,
                     r,
                     return_distance=False,
                     count_only=False,
                     sort_results=False,
                     use_radians=False):
        """TODO: Docstring

         Parameters
         ----------
         XY :
             TODO
         r: scalar, float
             TODO
         return_distance :
             TODO
         count_only :
             TODO
         sort_results :
             TODO
         use_radians :
             TODO

         Returns
         -------
         d :
             TODO
         ind :
             TODO
         """

        if r < 0.0:
            raise ValueError  # TODO

        xy = _prepare_xy_for_query(xy, use_radians)

        # TODO: query with radius

        pass


class CenterNodeBallTree:

    def __init__(self, grid):

        # requires center nodes
        if grid.Mesh2_face_x is None or grid.Mesh2_face_y is None:
            raise ValueError

        XY = np.vstack((deg2rad(grid.Mesh2_face_y.values),
                        deg2rad(grid.Mesh2_face_x.values))).T

        self.n_face = grid.nMesh2_face
        self.tree = BallTree(XY, metric='haversine')

    def query(self,
              xy: Union[np.ndarray, list, tuple],
              k: Optional[int] = 1,
              use_radians: Optional[bool] = False,
              return_distance: Optional[bool] = True,
              dualtree: Optional[bool] = False,
              breadth_first: Optional[bool] = False,
              sort_results: Optional[bool] = True):
        """TODO: Docstring

        Parameters
        ----------
        xy : array_like
            TODO
        k: int, optional
            TODO
        return_distance : bool, optional
            TODO
        dualtree : bool, optional
            TODO
        breadth_first : bool, optional
            TODO
        sort_results : bool, optional
            TODO
        use_radians : bool, optional
            TODO

        Returns
        -------
        d : np.ndarray
            TODO
        ind : np.ndarray
            TODO
        """
        pass

    def query_radius(self,
                     xy,
                     r,
                     return_distance=False,
                     count_only=False,
                     sort_results=False,
                     use_radians=False):
        """TODO: Docstring

         Parameters
         ----------
         XY :
             TODO
         r: scalar, float
             TODO
         return_distance :
             TODO
         count_only :
             TODO
         sort_results :
             TODO
         use_radians :
             TODO

         Returns
         -------
         d :
             TODO
         ind :
             TODO
         """

        if r < 0.0:
            raise ValueError  # TODO

        xy = _prepare_xy_for_query(xy, use_radians)

        # TODO: query with radius

        pass


def _prepare_xy_for_query(xy, use_radians):

    xy = np.asarray(xy)

    # expand if only a single node pair is provided
    if xy.ndim == 1:
        xy = np.expand_dims(xy, axis=0)

    # expected shape is [n_pairs, 2]
    if xy.shape[1] == 3:
        raise ValueError  # TODO, caterisian case

    if xy.shape[1] != 2:
        raise ValueError  # TODO, all others

    # swap X and Y for query
    xy[:, [0, 1]] = xy[:, [1, 0]]

    # balltree expects units in radians for query
    if not use_radians:
        xy = np.deg2rad(xy)

    return xy
