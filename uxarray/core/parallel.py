"""Grid-aware ``map_blocks`` for ``UxDataArray`` and ``UxDataset``."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import dask.array as da
import xarray as xr
from dask import is_dask_collection

from uxarray.constants import GRID_DIMS
from uxarray.errors import DimensionError

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray
    from uxarray.core.dataset import UxDataset
    from uxarray.grid import Grid


def map_blocks(
    func: Callable[..., UxDataArray | UxDataset],
    obj: UxDataArray | UxDataset,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    template: xr.DataArray | xr.Dataset | None = None,
) -> UxDataArray | UxDataset:
    """Apply a function to each block of a ``UxDataArray`` or ``UxDataset``,
    keeping the grid attached.

    Behaves like :py:func:`xarray.map_blocks`, except that ``func`` receives
    each block as a ``UxDataArray``/``UxDataset`` with a grid attached, and the
    result is returned with ``obj.uxgrid`` attached.

    - If no grid dimension is split across chunks (e.g. chunked only along
      ``time``), every block carries ``obj.uxgrid`` itself.
    - If ``n_face`` is split, each block carries
      ``obj.uxgrid.isel(n_face=...)`` for exactly the faces it holds. The
      sub-grid is built inside the task, once per block. Node and edge data
      passed in ``args`` is gathered onto the sub-grid's nodes and edges, i.e.
      every node and edge of the block's faces, so it matches the block's grid.
    - If ``n_node`` or ``n_edge`` is split, every block carries ``obj.uxgrid``
      itself, whose sizes do not match the block. Pass the grid variables
      ``func`` needs in ``args`` instead (e.g. ``args=[uxgrid.node_lat]``):
      each block then receives the values for its own nodes or edges.

    Splitting along a grid dimension is only correct for some functions; see
    Notes.

    Parameters
    ----------
    func : callable
        Called as ``func(block, *args, **kwargs)``. Must return a
        ``DataArray`` or ``Dataset`` (a ``UxDataArray``/``UxDataset`` is fine).
        It must not change the length of a grid dimension.
    obj : UxDataArray or UxDataset
        Object whose blocks are passed to ``func``. If it is not backed by
        dask, ``func(obj, *args, **kwargs)`` is returned directly.
    args : sequence, optional
        Passed to ``func`` after ``obj``. ``DataArray``/``Dataset`` arguments,
        including grid variables such as ``uxgrid.edge_node_distances``, are
        rechunked along ``obj``'s grid dimensions to match ``obj``, split into
        blocks alongside it, and carry the same grid as its block. With
        ``n_face`` split, node and edge data is instead passed whole to every
        task and gathered onto each block's sub-grid.
    kwargs : mapping, optional
        Passed to ``func`` unchanged. Must not contain dask collections.
    template : DataArray or Dataset, optional
        Same as in :py:func:`xarray.map_blocks`. When omitted, the output is
        inferred by calling ``func`` on a zero-size block that carries a
        zero-face grid.

    Returns
    -------
    UxDataArray or UxDataset
        The assembled result, with ``obj.uxgrid`` attached.

    Raises
    ------
    DimensionError
        If more than one grid dimension is split across chunks, or if one is
        split while ``obj`` also has another grid dimension. Blocks along
        different grid dimensions cover unrelated parts of the grid. Rechunk the
        split dimensions to a single chunk first. Also raised if ``n_face`` is
        split and one ``args`` entry has both ``n_face`` and ``n_node`` or
        ``n_edge`` dimensions; pass face and node/edge data separately.

    See Also
    --------
    xarray.map_blocks
    UxDataArray.map_blocks
    UxDataset.map_blocks
    Grid.chunks

    Notes
    -----
    Splitting along a grid dimension (``n_face``, ``n_edge`` or ``n_node``) is
    correct for functions that use each element's own values, such as area or
    edge-length weighting, Coriolis terms, masks from coordinates, or per-face
    geometry. With ``n_face`` split, it is also correct for functions that
    compute each face from its own nodes and edges, such as node-to-face
    topological aggregations or a flux divergence from edge velocities, when
    the node or edge data is passed in ``args``.

    It is not correct for functions that read neighbouring elements of the same
    kind, such as ``gradient``, ``curl``, ``divergence`` or neighbourhood
    reductions, or that compute nodes or edges from faces:

    - With ``n_face`` split, each block's edge looks like a grid boundary, so
      faces near it come out NaN or truncated, without an error. Face indices
      are often not spatially compact, so this can affect most faces in a block.
    - With ``n_node`` or ``n_edge`` split, the block's grid is the whole grid,
      which does not match the block's nodes or edges.

    For those functions, keep the grid dimension in a single chunk and split
    along other dimensions, e.g. ``time`` or ``lev``.
    """
    from uxarray.core.dataarray import UxDataArray
    from uxarray.core.dataset import UxDataset

    if not isinstance(obj, UxDataArray | UxDataset):
        raise TypeError(
            "ux.map_blocks expects a UxDataArray or UxDataset; "
            f"got type(obj)={type(obj)}. Use xarray.map_blocks for plain xarray objects."
        )
    uxgrid = obj.uxgrid

    if not is_dask_collection(obj):
        return func(obj, *args, **(kwargs or {}))

    chunksizes = obj.chunksizes
    split_dims = [dim for dim in GRID_DIMS if len(chunksizes.get(dim, ())) > 1]
    grid_dims = [dim for dim in GRID_DIMS if dim in obj.dims]
    if split_dims and grid_dims != split_dims[:1]:
        single_chunk = {dim: -1 for dim in split_dims}
        raise DimensionError(
            "ux.map_blocks can split only one grid dimension, and only for objects with "
            f"no other grid dimension. Got grid dimensions {grid_dims}, with {split_dims} "
            "split across chunks. Blocks along different grid dimensions do not line up "
            "(a block of faces and a block of nodes cover unrelated parts of the grid); "
            f"rechunk with obj.chunk({single_chunk}) first."
        )

    # grid variables passed in args are usually in memory as a single chunk; match
    # obj's chunks along its grid dimensions so xarray can split them alongside obj
    grid_chunks = {dim: chunksizes[dim] for dim in grid_dims if dim in chunksizes}
    if split_dims == ["n_face"]:
        # node and edge data is gathered onto each block's sub-grid, so every task
        # needs it whole; left chunked, xarray would split it along its own grid dim
        grid_chunks.update({"n_node": -1, "n_edge": -1})
    args = [
        arg.chunk({dim: grid_chunks[dim] for dim in grid_chunks.keys() & arg.dims})
        if isinstance(arg, xr.DataArray | xr.Dataset)
        else arg
        for arg in args
    ]

    face_positions = None
    if split_dims == ["n_face"]:
        arg_dims = [
            set(arg.dims) for arg in args if isinstance(arg, xr.DataArray | xr.Dataset)
        ]
        if any("n_face" in dims and dims & {"n_node", "n_edge"} for dims in arg_dims):
            raise DimensionError(
                "With 'n_face' split, each DataArray/Dataset in args must have either "
                "'n_face' or 'n_node'/'n_edge' dimensions, not both: face data is split "
                "into blocks while node and edge data is gathered onto each block's "
                "sub-grid. Pass them as separate arguments."
            )
        if any("n_edge" in dims for dims in arg_dims):
            # face sub-grids keep their edges only when this connectivity exists; build
            # it once here rather than racing to build it from inside the tasks
            _ = uxgrid.face_edge_connectivity

        # Each face's position in the parent grid, passed as a hidden first argument:
        # xarray splits it into blocks alongside obj, so a block can recover which
        # faces (and so which sub-grid) it holds.
        face_positions = xr.DataArray(
            da.arange(obj.sizes["n_face"], chunks=(chunksizes["n_face"],)),
            dims="n_face",
        )
        args = [face_positions, *args]

    @functools.wraps(func)
    def _block_func(block, *block_args, **block_kwargs):
        block_positions = None
        if face_positions is not None:
            block_positions, *block_args = block_args
        block_grid = _grid_for_block(block, block_positions, uxgrid)
        return func(
            _with_grid(block, block_grid),
            *(
                _arg_for_block(arg, block_grid, uxgrid, block_positions is not None)
                for arg in block_args
            ),
            **block_kwargs,
        )

    result = xr.map_blocks(_block_func, obj, args, kwargs, template)

    if isinstance(result, xr.Dataset):
        return UxDataset(
            result, uxgrid=uxgrid, source_datasets=getattr(obj, "source_datasets", None)
        )
    return UxDataArray(result, uxgrid=uxgrid)


def _grid_for_block(
    block: xr.DataArray | xr.Dataset,
    face_positions: xr.DataArray | None,
    uxgrid: Grid,
) -> Grid:
    """The grid matching one block: the sub-grid of the faces at ``face_positions``
    when ``n_face`` was split, otherwise the whole grid."""
    if face_positions is not None:
        positions = face_positions.values
        start = int(positions[0]) if positions.size else 0
        return uxgrid.isel(n_face=slice(start, start + positions.size))
    if any(block.sizes[dim] == 0 for dim in GRID_DIMS if dim in block.dims):
        # xarray infers the output template by calling func on zero-size blocks
        return uxgrid.isel(n_face=slice(0, 0))
    return uxgrid


def _arg_for_block(value: Any, block_grid: Grid, uxgrid: Grid, face_split: bool) -> Any:
    """One ``args`` entry for one block. When ``n_face`` was split, node and edge
    data arrive whole and are gathered onto the nodes and edges of the block's
    face sub-grid (its halo); everything else just gets the block's grid."""
    if (
        face_split
        and isinstance(value, xr.DataArray | xr.Dataset)
        and {"n_node", "n_edge"} & set(value.dims)
    ):
        return _with_grid(value, uxgrid)._slice_from_grid(block_grid)
    return _with_grid(value, block_grid)


def _with_grid(value: Any, uxgrid: Grid) -> Any:
    """Wrap a plain xarray block as a ``UxDataArray``/``UxDataset`` on ``uxgrid``;
    pass anything else through."""
    from uxarray.core.dataarray import UxDataArray
    from uxarray.core.dataset import UxDataset

    if isinstance(value, xr.DataArray):
        return UxDataArray(value, uxgrid=uxgrid)
    if isinstance(value, xr.Dataset):
        return UxDataset(value, uxgrid=uxgrid)
    return value
