import numpy as np
import dask
import dask.array as da


import uxarray.core.dataarray


def result_array(arr):
    if isinstance(arr, np.ndarray):
        return np.empty
    if isinstance(arr, dask.array.core.Array):
        return da.empty


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

    # TODO:
    aggregated_var = _apply_node_to_face_aggregation(
        uxda, aggregation, aggregation_func_kwargs
    )

    return uxarray.core.dataarray.UxDataArray(
        uxgrid=uxda.uxgrid,
        data=aggregated_var,
        dims=uxda.dims,
        name=uxda.name,
    ).rename({"n_node": "n_face"})


def _apply_node_to_face_aggregation(
    uxda, aggregation_func, aggregation_func_kwargs, result_array_kwargs=None
):
    """TODO:"""

    # shape [..., n_face] since data is being aggregated onto the faces
    result = result_array(uxda.data)(
        shape=(uxda.data.shape[:-1]) + (uxda.uxgrid.n_face,)
    )

    for (
        cur_face_node_partition,
        cur_original_face_indices,
    ) in uxda.uxgrid.partitioned_face_node_connectivity:
        # index array using flattened connectivity (to avoid Dask errors)
        data_flat = uxda.data[..., cur_face_node_partition.flatten()]

        # reshape index data back to desired shape [..., n_face_geom, geom_size]
        data_reshaped = data_flat.reshape(
            (uxda.data.shape[:-1]) + cur_face_node_partition.shape
        )

        # apply aggregation on current partition of elements
        aggregation_par = getattr(data_reshaped, aggregation_func)(
            axis=-1, **aggregation_func_kwargs
        )

        # store computed aggregation using original face indices
        result[..., cur_original_face_indices] = aggregation_par

    return result


def _node_to_edge_aggregation(uxda, aggregation, aggregation_func_kwargs):
    """Applies a Node to Edge Topological aggregation."""
    if not uxda._node_centered():
        raise ValueError(
            f"Data Variable must be mapped to the corner nodes of each face, with dimension "
            f"{uxda.uxgrid.n_face}."
        )

    # TODO:
    aggregated_var = _apply_node_to_edge_aggregation_(
        uxda, aggregation, aggregation_func_kwargs
    )

    return uxarray.core.dataarray.UxDataArray(
        uxgrid=uxda.uxgrid,
        data=aggregated_var,
        dims=uxda.dims,
        name=uxda.name,
    ).rename({"n_node": "n_edge"})


def _apply_node_to_edge_aggregation_(
    uxda, aggregation_func, aggregation_func_kwargs, result_array_kwargs=None
):
    """TODO:"""

    data_flat = uxda.data[..., uxda.uxgrid.edge_node_connectivity.data.flatten()]
    data_reshaped = data_flat.reshape((uxda.data.shape[:-1]) + (uxda.uxgrid.n_edge, 2))
    result = getattr(data_reshaped, aggregation_func)(
        axis=-1, **aggregation_func_kwargs
    )
    return result
