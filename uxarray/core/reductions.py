import numpy as np
import dask.array as da

from uxarray.grid.connectivity import get_face_node_partitions

import uxarray.core.dataarray


NUMPY_REDUCTIONS = {
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
    "prod": np.prod,
    "sum": np.sum,
    "std": np.std,
    "var": np.var,
    "median": np.median,
    "all": np.all,
    "any": np.any,
}


def _uxda_grid_reduce(uxda, keep_attrs, destination, reduction, **kwargs):
    if destination is None:
        raise ValueError(
            "Attempting to perform a local reduction, but no destination was provided."
        )

    if uxda._node_centered():
        # reduction of a node-centered data variable
        if destination == "face":
            return _node_to_face_reduction(uxda, reduction, kwargs)
        elif destination == "edge":
            return _node_to_edge_reduction(uxda, reduction, kwargs)
        else:
            raise ValueError("TODO: Invalid dimension for node reduction")

    elif uxda._edge_centered():
        # reduction of an edge-centered data variable
        raise NotImplementedError(
            "Reductions of edge-centered data variables not yet supported."
        )
        # if destination == "node":
        #     pass
        # elif destination == "face":
        #     pass
        # else:
        #     raise ValueError("TODO: )

    elif uxda._face_centered():
        # reduction of a face-centered data variable
        raise NotImplementedError(
            "Reductions of face-centered data variables not yet supported."
        )
        # if destination == "node":
        #     pass
        # elif destination == "edge":
        #     pass
        # else:
        #     raise ValueError("TODO: ")

    else:
        raise ValueError(
            f"Invalid destination provided. Got: {destination}, but expected one of: "
            f"['node', 'edge', 'face']"
        )


def _node_to_face_reduction(uxda, reduction, reduction_func_kwargs):
    """TODO:"""
    if not uxda._node_centered():
        raise ValueError(
            f"Data Variable must be mapped to the corner nodes of each face, with dimension "
            f"{uxda.uxgrid.n_face}."
        )

    if isinstance(uxda.data, np.ndarray):
        # apply reduction using numpy
        reduced_var = _apply_node_to_face_reduction_numpy(
            uxda, NUMPY_REDUCTIONS[reduction], reduction_func_kwargs
        )
    elif isinstance(uxda.data, da.array):
        # apply reduction on dask array, TODO:
        reduced_var = _apply_node_to_face_reduction_numpy(
            uxda, NUMPY_REDUCTIONS[reduction], reduction_func_kwargs
        )
    else:
        raise ValueError

    return uxarray.core.dataarray.UxDataArray(
        uxgrid=uxda.uxgrid,
        data=reduced_var,
        dims=uxda.dims,
        name=uxda.name,
    ).rename({"n_node": "n_face"})


def _apply_node_to_face_reduction_numpy(uxda, reduction_func, reduction_func_kwargs):
    data = uxda.values
    face_node_conn = uxda.uxgrid.face_node_connectivity.values
    n_nodes_per_face = uxda.uxgrid.n_nodes_per_face.values

    (
        change_ind,
        n_nodes_per_face_sorted_ind,
        element_sizes,
        size_counts,
    ) = get_face_node_partitions(n_nodes_per_face)

    result = np.empty(shape=(data.shape[:-1]) + (uxda.uxgrid.n_face,))

    for e, start, end in zip(element_sizes, change_ind[:-1], change_ind[1:]):
        face_inds = n_nodes_per_face_sorted_ind[start:end]
        face_nodes_par = face_node_conn[face_inds, 0:e]

        # apply reduction function to current face node partition
        reduction_par = reduction_func(
            data[..., face_nodes_par], axis=-1, **reduction_func_kwargs
        )

        # store current reduction
        result[..., face_inds] = reduction_par

    return result


def _apply_node_to_face_reduction_dask():
    pass


def _node_to_edge_reduction(uxda, reduction, reduction_func_kwargs):
    if not uxda._node_centered():
        raise ValueError(
            f"Data Variable must be mapped to the corner nodes of each face, with dimension "
            f"{uxda.uxgrid.n_face}."
        )

    if isinstance(uxda.data, np.ndarray):
        # apply reduction using numpy
        reduced_var = _apply_node_to_edge_reduction_numpy(
            uxda, NUMPY_REDUCTIONS[reduction], reduction_func_kwargs
        )
    elif isinstance(uxda.data, da.array):
        # apply reduction on dask array, TODO:
        reduced_var = _apply_node_to_edge_reduction_numpy(
            uxda, NUMPY_REDUCTIONS[reduction], reduction_func_kwargs
        )
    else:
        raise ValueError

    return uxarray.core.dataarray.UxDataArray(
        uxgrid=uxda.uxgrid,
        data=reduced_var,
        dims=uxda.dims,
        name=uxda.name,
    ).rename({"n_node": "n_edge"})


def _apply_node_to_edge_reduction_numpy(uxda, reduction_func, reduction_func_kwargs):
    data = uxda.values
    edge_node_conn = uxda.uxgrid.edge_node_connectivity.values

    result = reduction_func(data[..., edge_node_conn], axis=-1, **reduction_func_kwargs)

    return result


def _apply_node_to_edge_reduction_dask(uxda, reduction_func, reduction_func_kwargs):
    pass
