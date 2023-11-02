# (cellVar[cellsOnEdge[edgeInd,1],k]-cellVar[cellsOnEdge[edgeInd,0],k]) / dcEdge[edgeInd]
import numpy as np

from numba import njit

from uxarray.constants import INT_FILL_VALUE


# @njit
def _calculate_abs_edge_grad(d_var, edge_faces, edge_node_distances, n_edge):
    """docstring TODO.

    Parameters
    ----------
    d_var
        todo
    edge_faces
        todo
    edge_node_distances
        todo
    n_edge
        todo
    Returns
    -------
    grad
        tdo
    """

    # obtain all edges that saddle two faces
    saddle_mask = edge_faces[:, 1] != INT_FILL_VALUE

    # gradient initialized to zero
    grad = np.zeros(n_edge)

    # compute gradient
    grad[saddle_mask] = (d_var[..., edge_faces[saddle_mask, 0]] -
                         d_var[..., edge_faces[saddle_mask, 1]]
                        ) / edge_node_distances[saddle_mask]

    # no sense of direction, take absolute value
    grad = np.abs(grad)

    return grad


def _gradient_uxda(d_var, grid, method):
    pass
