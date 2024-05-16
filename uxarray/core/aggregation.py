import numpy as np
import dask.array as da

from uxarray.grid.connectivity import get_face_node_partitions

import uxarray.core.dataarray


NUMPY_AGGREGATIONS = {
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


def _uxda_grid_aggregate(uxda, destination, aggregation, **kwargs):
    """Applies a desired aggregation on the data stored in the provided
    UxDataArray."""
    if destination is None:
        raise ValueError(
            "Attempting to perform a topological aggregation, but no destination was provided."
        )

    if uxda._node_centered():
        # aggregation of a node-centered data variable
        if destination == "face":
            return _node_to_face_aggregation(uxda, aggregation, kwargs)
        elif destination == "edge":
            return _node_to_edge_aggregation(uxda, aggregation, kwargs)
        else:
            raise ValueError(
                f"Invalid destination for a node-centered data variable. Expected"
                f"one of ['face', 'edge' but received {destination}"
            )

    elif uxda._edge_centered():
        # aggregation of an edge-centered data variable
        raise NotImplementedError(
            "Aggregation of edge-centered data variables is not yet supported."
        )
        # if destination == "node":
        #     pass
        # elif destination == "face":
        #     pass
        # else:
        #     raise ValueError("TODO: )

    elif uxda._face_centered():
        # aggregation of a face-centered data variable
        raise NotImplementedError(
            "Aggregation of face-centered data variables is not yet supported."
        )
        # if destination == "node":
        #     pass
        # elif destination == "edge":
        #     pass
        # else:
        #     raise ValueError("TODO: ")

    else:
        raise ValueError(
            "Invalid data mapping. Data variable is expected to be mapped to either the "
            "nodes, faces, or edges of the source grid."
        )


def _node_to_face_aggregation(uxda, aggregation, aggregation_func_kwargs):
    """Applies a Node to Face Topological aggregation."""
    if not uxda._node_centered():
        raise ValueError(
            f"Data Variable must be mapped to the corner nodes of each face, with dimension "
            f"{uxda.uxgrid.n_face}."
        )

    if isinstance(uxda.data, np.ndarray):
        # apply aggregation using numpy
        aggregated_var = _apply_node_to_face_aggregation_numpy(
            uxda, NUMPY_AGGREGATIONS[aggregation], aggregation_func_kwargs
        )
    elif isinstance(uxda.data, da.Array):
        # apply aggregation on dask array, TODO:
        aggregated_var = _apply_node_to_face_aggregation_numpy(
            uxda, NUMPY_AGGREGATIONS[aggregation], aggregation_func_kwargs
        )
    else:
        raise ValueError

    return uxarray.core.dataarray.UxDataArray(
        uxgrid=uxda.uxgrid,
        data=aggregated_var,
        dims=uxda.dims,
        name=uxda.name,
    ).rename({"n_node": "n_face"})


def _apply_node_to_face_aggregation_numpy(
    uxda, aggregation_func, aggregation_func_kwargs
):
    """Applies a Node to Face Topological aggregation on a Numpy array."""
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

        # apply aggregation function to current face node partition
        aggregation_par = aggregation_func(
            data[..., face_nodes_par], axis=-1, **aggregation_func_kwargs
        )

        # store current aggregation
        result[..., face_inds] = aggregation_par

    return result


def _apply_node_to_face_aggregation_dask(*args, **kwargs):
    """Applies a Node to Face Topological aggregation on a Dask array."""
    pass


def _node_to_edge_aggregation(uxda, aggregation, aggregation_func_kwargs):
    """Applies a Node to Edge Topological aggregation."""
    if not uxda._node_centered():
        raise ValueError(
            f"Data Variable must be mapped to the corner nodes of each face, with dimension "
            f"{uxda.uxgrid.n_face}."
        )

    if isinstance(uxda.data, np.ndarray):
        # apply aggregation using numpy
        aggregation_var = _apply_node_to_edge_aggregation_numpy(
            uxda, NUMPY_AGGREGATIONS[aggregation], aggregation_func_kwargs
        )
    elif isinstance(uxda.data, da.Array):
        # apply aggregation on dask array, TODO:
        aggregation_var = _apply_node_to_edge_aggregation_numpy(
            uxda, NUMPY_AGGREGATIONS[aggregation], aggregation_func_kwargs
        )
    else:
        raise ValueError

    return uxarray.core.dataarray.UxDataArray(
        uxgrid=uxda.uxgrid,
        data=aggregation_var,
        dims=uxda.dims,
        name=uxda.name,
    ).rename({"n_node": "n_edge"})


def _apply_node_to_edge_aggregation_numpy(
    uxda, aggregation_func, aggregation_func_kwargs
):
    """Applies a Node to Edge topological aggregation on a numpy array."""
    data = uxda.values
    edge_node_conn = uxda.uxgrid.edge_node_connectivity.values
    result = aggregation_func(
        data[..., edge_node_conn], axis=-1, **aggregation_func_kwargs
    )
    return result


def _apply_node_to_edge_aggregation_dask(*args, **kwargs):
    """Applies a Node to Edge topological aggregation on a dask array."""
    pass
