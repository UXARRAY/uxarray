# (cellVar[cellsOnEdge[edgeInd,1],k]-cellVar[cellsOnEdge[edgeInd,0],k]) / dcEdge[edgeInd]
import numpy as np

from typing import Optional
from uxarray.constants import INT_FILL_VALUE


# @njit
def _calculate_grad_on_edge(
    d_var,
    edge_faces,
    edge_face_distances,
    n_edge,
    use_magnitude: Optional[bool] = True,
    normalize: Optional[bool] = True,
):
    """Helper function for computing the gradient on each edge.

    TODO: add algorithmic outline

    Parameters
    ----------
    d_var
        todo
    edge_faces
        todo
    edge_face_distances
        todo
    n_edge
        todo
    use_magnitude: bool, optional=True
        todo
    normalize
        todo
    Returns
    -------
    grad
        todo
    """

    # obtain all edges that saddle two faces
    saddle_mask = edge_faces[:, 1] != INT_FILL_VALUE

    # gradient initialized to zero
    grad = np.zeros(n_edge)

    # compute gradient (difference between cell-center value divided by arc length)
    grad[saddle_mask] = (
        d_var[..., edge_faces[saddle_mask, 0]] - d_var[..., edge_faces[saddle_mask, 1]]
    ) / edge_face_distances[saddle_mask]
    if use_magnitude:
        # obtain magnitude if desired
        grad = np.abs(grad)

    if normalize:
        # normalize to [0, 1] if desired
        grad = grad / np.linalg.norm(grad)

    return grad
