from sklearn.neighbors import BallTree

import numpy as np
from numpy import deg2rad


def _corner_nodes_to_balltree(grid):
    """Constructs a ``sklearn.neighbors.BallTree`` data structure from the
    corner nodes (``Mesh2_node_x`` & ``Mesh2_node_y``) using the Haversine
    formula as a distance metric.

    Parameters
    ----------
    grid: ux.Grid
        TODO

    Returns
    -------
    tree: sklearn.neighbors.BallTree
        TODO
    """

    # pairs of lat/lon coordinates in radians with shape [n_nodes, 2]
    XY = np.vstack((deg2rad(grid.Mesh2_node_y.values),
                    deg2rad(grid.Mesh2_node_x.values))).T

    # construct Ball Tree using haversine distance calculations
    tree = BallTree(XY, metric='haversine')

    return tree
