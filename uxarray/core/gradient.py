# (cellVar[cellsOnEdge[edgeInd,1],k]-cellVar[cellsOnEdge[edgeInd,0],k]) / dcEdge[edgeInd]
import numpy as np

from numba import njit


@njit
def _calculate_edge_grad(d_var, edge_faces, edge_node_distances, n_edge):

    grad = (d_var[..., edge_faces[:, 0]] -
            d_var[..., edge_faces[:, 1]]) / edge_node_distances

    return grad
