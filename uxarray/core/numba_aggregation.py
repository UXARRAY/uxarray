import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _node_to_face_mean_numba(data, face_node_connectivity, n_nodes_per_face, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.mean(
            data[..., face_node_connectivity[i, : n_nodes_per_face[i]]]
        )
    return result


@njit(parallel=True)
def _node_to_face_max_numba(data, face_node_connectivity, n_nodes_per_face, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.max(
            data[..., face_node_connectivity[i, : n_nodes_per_face[i]]]
        )
    return result


@njit(parallel=True)
def _node_to_face_min_numba(data, face_node_connectivity, n_nodes_per_face, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.min(
            data[..., face_node_connectivity[i, : n_nodes_per_face[i]]]
        )
    return result


@njit(parallel=True)
def _node_to_face_prod_numba(data, face_node_connectivity, n_nodes_per_face, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.prod(
            data[..., face_node_connectivity[i, : n_nodes_per_face[i]]]
        )
    return result


@njit(parallel=True)
def _node_to_face_sum_numba(data, face_node_connectivity, n_nodes_per_face, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.sum(
            data[..., face_node_connectivity[i, : n_nodes_per_face[i]]]
        )
    return result


@njit(parallel=True)
def _node_to_face_std_numba(data, face_node_connectivity, n_nodes_per_face, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.std(
            data[..., face_node_connectivity[i, : n_nodes_per_face[i]]]
        )
    return result


@njit(parallel=True)
def _node_to_face_var_numba(data, face_node_connectivity, n_nodes_per_face, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.var(
            data[..., face_node_connectivity[i, : n_nodes_per_face[i]]]
        )
    return result


@njit(parallel=True)
def _node_to_face_median_numba(data, face_node_connectivity, n_nodes_per_face, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.median(
            data[..., face_node_connectivity[i, : n_nodes_per_face[i]]]
        )
    return result


@njit(parallel=True)
def _node_to_face_all_numba(data, face_node_connectivity, n_nodes_per_face, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.all(
            data[..., face_node_connectivity[i, : n_nodes_per_face[i]]]
        )
    return result


@njit(parallel=True)
def _node_to_face_any_numba(data, face_node_connectivity, n_nodes_per_face, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.any(
            data[..., face_node_connectivity[i, : n_nodes_per_face[i]]]
        )
    return result


NUMBA_NODE_FACE_AGGS = {
    "mean": _node_to_face_mean_numba,
    "max": _node_to_face_max_numba,
    "min": _node_to_face_min_numba,
    "prod": _node_to_face_prod_numba,
    "sum": _node_to_face_sum_numba,
    "std": _node_to_face_std_numba,
    "var": _node_to_face_var_numba,
    "median": _node_to_face_median_numba,
    "all": _node_to_face_all_numba,
    "any": _node_to_face_any_numba,
}


@njit(parallel=True)
def _node_to_edge_mean_numba(data, edge_node_connectivity, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.mean(data[..., edge_node_connectivity[i]])
    return result


@njit(parallel=True)
def _node_to_edge_max_numba(data, edge_node_connectivity, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.max(data[..., edge_node_connectivity[i]])
    return result


@njit(parallel=True)
def _node_to_edge_min_numba(data, edge_node_connectivity, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.min(data[..., edge_node_connectivity[i]])
    return result


@njit(parallel=True)
def _node_to_edge_prod_numba(data, edge_node_connectivity, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.prod(data[..., edge_node_connectivity[i]])
    return result


@njit(parallel=True)
def _node_to_edge_sum_numba(data, edge_node_connectivity, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.sum(data[..., edge_node_connectivity[i]])
    return result


@njit(parallel=True)
def _node_to_edge_std_numba(data, edge_node_connectivity, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.std(data[..., edge_node_connectivity[i]])
    return result


@njit(parallel=True)
def _node_to_edge_var_numba(data, edge_node_connectivity, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.var(data[..., edge_node_connectivity[i]])
    return result


@njit(parallel=True)
def _node_to_edge_median_numba(data, edge_node_connectivity, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.median(data[..., edge_node_connectivity[i]])
    return result


@njit(parallel=True)
def _node_to_edge_all_numba(data, edge_node_connectivity, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.all(data[..., edge_node_connectivity[i]])
    return result


@njit(parallel=True)
def _node_to_edge_any_numba(data, edge_node_connectivity, n_face):
    result = np.empty(shape=(data.shape[:-1]) + (n_face,))
    for i in prange(n_face):
        result[..., i] = np.any(data[..., edge_node_connectivity[i]])
    return result


NUMBA_NODE_EDGE_AGGS = {
    "mean": _node_to_edge_mean_numba,
    "max": _node_to_edge_max_numba,
    "min": _node_to_edge_min_numba,
    "prod": _node_to_edge_prod_numba,
    "sum": _node_to_edge_sum_numba,
    "std": _node_to_edge_std_numba,
    "var": _node_to_edge_var_numba,
    "median": _node_to_edge_median_numba,
    "all": _node_to_edge_all_numba,
    "any": _node_to_edge_any_numba,
}
