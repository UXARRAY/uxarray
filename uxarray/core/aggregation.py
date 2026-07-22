import numpy as np

import uxarray.core.dataarray
from uxarray.grid.connectivity import get_face_node_partitions

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
    import dask.array as da

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
        return uxarray.core.dataarray.UxDataArray(
            uxgrid=uxda.uxgrid,
            data=aggregated_var,
            dims=uxda.dims,
            name=uxda.name,
        ).rename({"n_node": "n_face"})
    elif isinstance(uxda.data, da.Array):
        # apply aggregation lazily on a dask array
        return _apply_node_to_face_aggregation_dask(
            uxda, NUMPY_AGGREGATIONS[aggregation], aggregation_func_kwargs
        )
    else:
        raise ValueError


def _node_to_face_kernel(
    data,
    face_node_conn,
    n_nodes_per_face,
    n_face,
    aggregation_func,
    aggregation_func_kwargs,
):
    """Node-to-face topological aggregation on a single numpy block.

    ``data`` has the node dimension as its last axis, shape ``(..., n_node)``;
    the result has the face dimension as its last axis, shape ``(..., n_face)``.
    Shared verbatim by the numpy path and the (blockwise) dask path.
    """
    (
        change_ind,
        n_nodes_per_face_sorted_ind,
        element_sizes,
        size_counts,
    ) = get_face_node_partitions(n_nodes_per_face)

    result = np.empty(shape=(data.shape[:-1]) + (n_face,))

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


def _apply_node_to_face_aggregation_numpy(
    uxda, aggregation_func, aggregation_func_kwargs
):
    """Applies a Node to Face Topological aggregation on a Numpy array."""
    return _node_to_face_kernel(
        uxda.values,
        uxda.uxgrid.face_node_connectivity.values,
        uxda.uxgrid.n_nodes_per_face.values,
        uxda.uxgrid.n_face,
        aggregation_func,
        aggregation_func_kwargs,
    )


def _apply_node_to_face_aggregation_dask(
    uxda, aggregation_func, aggregation_func_kwargs
):
    """Applies a Node to Face Topological aggregation on a Dask array, lazily"""
    import xarray as xr

    uxgrid = uxda.uxgrid
    n_face = uxgrid.n_face

    # n_node must live in a single chunk: any node can feed any face. The grid
    # metadata are core-dim inputs, so their core dims must be single-chunk too.
    da_in = uxda.chunk({"n_node": -1})
    face_node_conn = uxgrid.face_node_connectivity.chunk(
        {"n_face": -1, "n_max_face_nodes": -1}
    )
    n_nodes_per_face = uxgrid.n_nodes_per_face.chunk({"n_face": -1})

    result = xr.apply_ufunc(
        _node_to_face_kernel,
        da_in,
        face_node_conn,
        n_nodes_per_face,
        input_core_dims=[["n_node"], ["n_face", "n_max_face_nodes"], ["n_face"]],
        output_core_dims=[["n_face"]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"n_face": n_face}},
        kwargs={
            "n_face": n_face,
            "aggregation_func": aggregation_func,
            "aggregation_func_kwargs": aggregation_func_kwargs,
        },
    )

    return uxarray.core.dataarray.UxDataArray(result, uxgrid=uxgrid, name=uxda.name)


def _node_to_edge_aggregation(uxda, aggregation, aggregation_func_kwargs):
    """Applies a Node to Edge Topological aggregation."""
    import dask.array as da

    if not uxda._node_centered():
        raise ValueError(
            f"Data Variable must be mapped to the corner nodes of each face, with dimension "
            f"{uxda.uxgrid.n_face}."
        )

    aggregation_func = NUMPY_AGGREGATIONS[aggregation]

    if isinstance(uxda.data, np.ndarray):
        # apply aggregation using numpy
        aggregation_var = _apply_node_to_edge_aggregation_numpy(
            uxda, aggregation_func, aggregation_func_kwargs
        )
        return uxarray.core.dataarray.UxDataArray(
            uxgrid=uxda.uxgrid,
            data=aggregation_var,
            dims=uxda.dims,
            name=uxda.name,
        ).rename({"n_node": "n_edge"})
    elif isinstance(uxda.data, da.Array):
        # apply aggregation lazily on a dask array
        return _apply_node_to_edge_aggregation_dask(
            uxda, aggregation_func, aggregation_func_kwargs
        )
    else:
        raise ValueError


def _node_to_edge_kernel(
    data, edge_node_conn, aggregation_func, aggregation_func_kwargs
):
    """Node-to-edge topological aggregation on a single numpy block.

    ``data`` has the node dimension as its last axis, shape ``(..., n_node)``;
    the result has the edge dimension as its last axis, shape ``(..., n_edge)``.
    Each edge reduces over its two endpoint nodes. Shared verbatim by the numpy
    path and the (blockwise) dask path.
    """
    return aggregation_func(
        data[..., edge_node_conn], axis=-1, **aggregation_func_kwargs
    )


def _apply_node_to_edge_aggregation_numpy(
    uxda, aggregation_func, aggregation_func_kwargs
):
    """Applies a Node to Edge topological aggregation on a numpy array."""
    return _node_to_edge_kernel(
        uxda.values,
        uxda.uxgrid.edge_node_connectivity.values,
        aggregation_func,
        aggregation_func_kwargs,
    )


def _apply_node_to_edge_aggregation_dask(
    uxda, aggregation_func, aggregation_func_kwargs
):
    """Applies a Node to Edge topological aggregation on a Dask array, lazily.

    Mirrors the node-to-face dask path: the data field and the edge-node
    connectivity enter ``apply_ufunc`` as dask inputs (nothing computed
    eagerly), ``n_node`` is a single (uncrunked) core dimension, and the leading
    dimensions are processed blockwise. Each edge reduces over its two endpoint
    nodes.
    """
    import xarray as xr

    uxgrid = uxda.uxgrid
    n_edge = uxgrid.n_edge

    # n_node must live in a single chunk: any node can feed any edge. The grid
    # metadata is a core-dim input, so its core dims must be single-chunk too.
    da_in = uxda.chunk({"n_node": -1})
    edge_node_conn = uxgrid.edge_node_connectivity.chunk({"n_edge": -1, "two": -1})

    # match the numpy path's dtype (float for numeric aggs, bool for all/any)
    out_dtype = aggregation_func(
        np.empty((1, edge_node_conn.shape[-1]), dtype=uxda.dtype),
        axis=-1,
        **aggregation_func_kwargs,
    ).dtype

    result = xr.apply_ufunc(
        _node_to_edge_kernel,
        da_in,
        edge_node_conn,
        input_core_dims=[["n_node"], ["n_edge", "two"]],
        output_core_dims=[["n_edge"]],
        dask="parallelized",
        output_dtypes=[out_dtype],
        dask_gufunc_kwargs={"output_sizes": {"n_edge": n_edge}},
        kwargs={
            "aggregation_func": aggregation_func,
            "aggregation_func_kwargs": aggregation_func_kwargs,
        },
    )

    return uxarray.core.dataarray.UxDataArray(result, uxgrid=uxgrid, name=uxda.name)
