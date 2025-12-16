from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, TypeAlias
from warnings import warn

import numpy as np

from uxarray.core.dataset import UxDataset
from uxarray.core.utils import (
    _map_dims_to_ugrid,
    _open_dataset_with_fallback,
    match_chunks_to_ugrid,
)
from uxarray.grid import Grid
from uxarray.io._scrip import (
    _detect_multigrid,
    _extract_single_grid,
    _read_scrip,
    _resolve_cell_dims,
    _stack_cell_dims,
)

if TYPE_CHECKING:
    from xarray import Dataset

__all__ = [
    "open_grid",
    "open_multigrid",
    "list_grid_names",
    "open_dataset",
    "open_mfdataset",
]


def open_grid(
    grid_filename_or_obj: str | os.PathLike[Any] | dict | Dataset,
    chunks=None,
    use_dual: bool | None = False,
    **kwargs: dict[str, Any],
):
    """Constructs and returns a ``Grid`` from a grid file.

    Parameters
    ----------
    grid_filename_or_obj : str | os.PathLike[Any] | dict | xr.dataset
        Strings and Path objects are interpreted as a path to a grid file. Xarray Datasets assume that
        each member variable is in the UGRID conventions and will be used to create a ``ux.Grid``. Similarly, a dictionary
        containing UGRID variables can be used to create a ``ux.Grid``
    chunks : int, dict, 'auto' or None, default: None
        If provided, used to load the grid into dask arrays.

        - ``chunks="auto"`` will use dask ``auto`` chunking taking into account the
          engine preferred chunks.
        - ``chunks=None`` skips using dask, which is generally faster for
          small arrays.
        - ``chunks=-1`` loads the data with dask using a single chunk for all arrays.
        - ``chunks={}`` loads the data with dask using the engine's preferred chunk
          size, generally identical to the format's chunk size. If not available, a
          single chunk for all arrays.

        See dask chunking for more details.
    use_dual : bool, optional
        Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is MPAS
    **kwargs
        Additional arguments passed on to ``xarray.open_dataset``.
        Refer to the :func:`xarray docs <xarray.open_dataset>`
        for accepted keyword arguments.

    Returns
    -------
    uxgrid : uxarray.Grid
        Initialized ``ux.Grid``  object from the provided ``grid_filename_or_obj``

    Examples
    --------
    >>> import uxarray as ux

    Open a grid from a file path

    >>> uxgrid = ux.open_grid("grid_filename.nc")

    Lazily load grid variables using Dask

    >>> uxgrid = ux.open_grid("grid_filename.nc", chunks=-1)
    """
    import xarray as xr

    # Handle chunk-related kwargs first.
    data_chunks = kwargs.pop("data_chunks", None)
    if data_chunks is not None:
        chunks = match_chunks_to_ugrid(grid_filename_or_obj, data_chunks)
    elif chunks is not None:
        chunks = match_chunks_to_ugrid(grid_filename_or_obj, chunks)

    return_chunks = kwargs.pop("return_chunks", False)
    chunk_grid = kwargs.pop("chunk_grid", chunks is not None)
    grid_chunks = chunks if chunk_grid else None

    grid = None

    # Special case for FESOM2 ASCII Dataset (stored as a directory)
    if isinstance(grid_filename_or_obj, (str, os.PathLike)) and os.path.isdir(
        grid_filename_or_obj
    ):
        nod2d_path = os.path.join(grid_filename_or_obj, "nod2d.out")
        elem2d_path = os.path.join(grid_filename_or_obj, "elem2d.out")
        if os.path.isfile(nod2d_path) and os.path.isfile(elem2d_path):
            grid = Grid.from_dataset(grid_filename_or_obj)
        else:
            raise FileNotFoundError(
                f"The directory '{grid_filename_or_obj}' must contain both 'nod2d.out' and 'elem2d.out'."
            )

    elif isinstance(grid_filename_or_obj, dict):
        # Unpack the dictionary and construct a grid from topology
        grid = Grid.from_topology(**grid_filename_or_obj)

    elif isinstance(grid_filename_or_obj, (list, tuple, np.ndarray, xr.DataArray)):
        # Construct grid from face vertices
        grid = Grid.from_face_vertices(grid_filename_or_obj, **kwargs)

    elif isinstance(grid_filename_or_obj, xr.Dataset):
        # Construct a grid from a dataset file
        grid = Grid.from_dataset(grid_filename_or_obj, use_dual=use_dual)

    else:
        # Attempt to use Xarray directly for remaining input types
        grid_ds = _open_dataset_with_fallback(
            grid_filename_or_obj, chunks=grid_chunks, **kwargs
        )
        grid = Grid.from_dataset(grid_ds, use_dual=use_dual)

    # Return the grid (and chunks, if requested) in a consistent manner.
    if return_chunks:
        return grid, chunks
    else:
        return grid


MaskValue: TypeAlias = Any | Sequence[Any]
MaskActiveValue: TypeAlias = MaskValue | Mapping[str, MaskValue] | None


def open_multigrid(
    grid_filename_or_obj: str | Path | "Dataset",
    gridnames: list[str] | None = None,
    mask_filename: str | Path | "Dataset" | None = None,
    mask_active_value: MaskActiveValue = 1,
    **kwargs: dict[str, Any],
) -> dict[str, Grid]:
    """Open a multi-grid SCRIP file and construct ``Grid`` objects.

    Parameters
    ----------
    grid_filename_or_obj : str, Path or xr.Dataset
        Path to the multi-grid SCRIP file or an already opened dataset.
    gridnames : list of str, optional
        Specific grid names to load. If ``None``, all grids are loaded.
    mask_filename : str, Path or xr.Dataset, optional
        Optional path to a mask file containing ``<grid>.msk`` variables.
        Defaults to retaining cells where mask value equals 1.
    mask_active_value : scalar, sequence or mapping[str, scalar/sequence], optional
        Mask value(s) treated as active. Provide a scalar or sequence to apply to
        all grids, or a dict keyed by grid name for per-grid overrides. When a
        mapping is provided and a grid name is not found, the fallback is the
        mapping's ``"_default"``/``"default"`` entry if present, otherwise 1.
    **kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`xarray.open_dataset`
        when opening ``grid_filename_or_obj``.

    Returns
    -------
    dict[str, Grid]
        Dictionary mapping grid names to ``Grid`` objects.
    """
    import xarray as xr

    grid_ds_opened = False
    if isinstance(grid_filename_or_obj, xr.Dataset):
        grid_ds = grid_filename_or_obj
    else:
        grid_ds = xr.open_dataset(grid_filename_or_obj, **kwargs)
        grid_ds_opened = True

    mask_ds = None
    mask_ds_opened = False
    if mask_filename is not None:
        if isinstance(mask_filename, xr.Dataset):
            mask_ds = mask_filename
        else:
            mask_ds = xr.open_dataset(mask_filename)
            mask_ds_opened = True

    try:
        active_value_map: Mapping[str, MaskValue] | None = (
            mask_active_value if isinstance(mask_active_value, Mapping) else None
        )
        default_active_value: MaskValue = (
            active_value_map.get("_default", active_value_map.get("default", 1))
            if active_value_map is not None
            else (mask_active_value if mask_active_value is not None else 1)
        )
        if default_active_value is None:
            default_active_value = 1

        active_value_cache: dict[str, np.ndarray] = {}

        def _normalize_active_values(value: MaskValue | None) -> np.ndarray:
            """Normalize mask active values to a 1D numpy array."""
            if value is None:
                value = default_active_value

            if isinstance(value, (str, bytes)) or not np.iterable(value):
                return np.asarray([value])

            return np.asarray(list(value)).ravel()

        def _active_mask_values_for_grid(grid_name: str) -> np.ndarray:
            """Return the active mask value(s) for a grid as a 1D array."""
            cached = active_value_cache.get(grid_name)
            if cached is not None:
                return cached

            if active_value_map is not None:
                value = active_value_map.get(grid_name, default_active_value)
            else:
                value = (
                    mask_active_value
                    if mask_active_value is not None
                    else default_active_value
                )

            active_values = _normalize_active_values(value)
            active_value_cache[grid_name] = active_values
            return active_values

        format_type, grids_dict = _detect_multigrid(grid_ds)

        if format_type == "single_scrip":
            if gridnames is not None and "grid" not in gridnames:
                raise ValueError(
                    f"Requested grids {gridnames} not found. "
                    "This file contains a single grid named 'grid'."
                )
            grid_ds_ugrid, source_dims_dict = _read_scrip(grid_ds)
            return {
                "grid": Grid(
                    grid_ds_ugrid,
                    source_grid_spec="Scrip",
                    source_dims_dict=source_dims_dict,
                )
            }

        if not grids_dict:
            raise ValueError(f"No grids detected in file: {grid_filename_or_obj}")

        available_grids = list(grids_dict.keys())

        if gridnames is None:
            grids_to_load = available_grids
        else:
            if isinstance(gridnames, str):
                requested = [gridnames]
            else:
                requested = list(gridnames)

            grids_to_load = []
            for name in requested:
                if name not in grids_dict:
                    raise ValueError(
                        f"Grid '{name}' not found. Available grids: {available_grids}"
                    )
                grids_to_load.append(name)

        loaded_grids: dict[str, Grid] = {}
        for grid_name in grids_to_load:
            metadata = grids_dict[grid_name]
            scrip_ds = _extract_single_grid(grid_ds, grid_name, metadata)
            grid_ds_ugrid, source_dims_dict = _read_scrip(scrip_ds)

            grid = Grid(
                grid_ds_ugrid,
                source_grid_spec="Scrip",
                source_dims_dict=source_dims_dict,
            )

            if mask_ds is not None:
                mask_var = f"{grid_name}.msk"
                if mask_var in mask_ds:
                    mask_da = mask_ds[mask_var]
                    mask_cell_dims = _resolve_cell_dims(metadata, mask_da.dims)
                    mask_flat = _stack_cell_dims(mask_da, mask_cell_dims, "grid_size")
                    mask_values = np.asarray(mask_flat.values)
                    active_values = _active_mask_values_for_grid(grid_name)
                    active_mask = np.isin(mask_values, active_values)
                    active_indices = np.flatnonzero(active_mask)
                    grid = grid.isel(n_face=active_indices)
                else:
                    warn(
                        f"Mask variable '{mask_var}' not found in mask file; "
                        f"grid '{grid_name}' will be returned without masking."
                    )

            loaded_grids[grid_name] = grid

        return loaded_grids
    finally:
        if grid_ds_opened:
            grid_ds.close()
        if mask_ds is not None and mask_ds_opened:
            mask_ds.close()


def list_grid_names(
    grid_filename_or_obj: str | Path | "Dataset", **kwargs: dict[str, Any]
) -> list[str]:
    """List all grid names available within a grid file.

    Parameters
    ----------
    grid_filename_or_obj : str, Path or xr.Dataset
        Path to the grid file or an already opened dataset.
    **kwargs : dict, optional
        Additional keyword arguments forwarded to :func:`xarray.open_dataset`.

    Returns
    -------
    list[str]
        ``['grid']`` for single-grid files or the detected grid names for
        multi-grid files.
    """
    import xarray as xr

    grid_ds_opened = False
    if isinstance(grid_filename_or_obj, xr.Dataset):
        grid_ds = grid_filename_or_obj
    else:
        grid_ds = xr.open_dataset(grid_filename_or_obj, **kwargs)
        grid_ds_opened = True

    try:
        format_type, grids_dict = _detect_multigrid(grid_ds)
        names = list(grids_dict.keys())
        if format_type == "single_scrip":
            return names or ["grid"]
        return names
    finally:
        if grid_ds_opened:
            grid_ds.close()


def open_dataset(
    grid_filename_or_obj: str | os.PathLike[Any] | dict | Dataset,
    filename_or_obj: str | os.PathLike[Any],
    chunks=None,
    chunk_grid: bool = True,
    use_dual: bool | None = False,
    grid_kwargs: dict[str, Any] | None = None,
    **kwargs: dict[str, Any],
) -> UxDataset:
    """Wraps ``xarray.open_dataset()`` for loading in a dataset paired with a grid file.

    Parameters
    ----------
    grid_filename_or_obj : str | os.PathLike[Any] | dict | xr.dataset
        Strings and Path objects are interpreted as a path to a grid file. Xarray Datasets assume that
        each member variable is in the UGRID conventions and will be used to create a ``ux.Grid``. Similarly, a dictionary
        containing UGRID variables can be used to create a ``ux.Grid``
    filename_or_obj : str | os.PathLike[Any]
        String or Path object as a path to a netCDF file or an OpenDAP URL that
        stores the actual data set. It is the same ``filename_or_obj`` in
        ``xarray.open_dataset``.
    chunks : int, dict, 'auto' or None, default: None
        If provided, used to load the grid into dask arrays.

        - ``chunks="auto"`` will use dask ``auto`` chunking taking into account the
          engine preferred chunks.
        - ``chunks=None`` skips using dask, which is generally faster for
          small arrays.
        - ``chunks=-1`` loads the data with dask using a single chunk for all arrays.
        - ``chunks={}`` loads the data with dask using the engine's preferred chunk
          size, generally identical to the format's chunk size. If not available, a
          single chunk for all arrays.
    chunk_grid : bool, default: True
        If valid chunks are passed in, determines whether to also apply the same chunks to the attached ``Grid``
    use_dual : bool, optional
        Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is MPAS
    grid_kwargs : dict, optional
        Additional arguments passed on to ``ux.open_grid`` when opening up a Grid File.
    **kwargs
        Additional arguments passed on to ``xarray.open_dataset``.
        Refer to the :func:`xarray docs <xarray.open_dataset>`
        for accepted keyword arguments.

    Returns
    -------

    uxds : uxarray.UxDataset
        Dataset with linked `uxgrid` property of type `Grid`.

    Examples
    --------
    Open a dataset with a grid file and data file

    >>> import uxarray as ux
    >>> ux_ds = ux.open_dataset("grid_file.nc", "data_file.nc")
    """
    if grid_kwargs is None:
        grid_kwargs = {}

    # Construct a Grid, validate parameters, and correct chunks
    uxgrid, corrected_chunks = _get_grid(
        grid_filename_or_obj, chunks, chunk_grid, use_dual, grid_kwargs, **kwargs
    )

    # Load the data as a Xarray Dataset
    ds = _open_dataset_with_fallback(filename_or_obj, chunks=corrected_chunks, **kwargs)

    # Map original dimensions to the UGRID conventions
    ds = _map_dims_to_ugrid(ds, uxgrid._source_dims_dict, uxgrid)

    # Create a UXarray Dataset by linking the Xarray Dataset with a UXarray Grid
    return UxDataset(ds, uxgrid=uxgrid, source_datasets=str(filename_or_obj))


def open_mfdataset(
    grid_filename_or_obj: str | os.PathLike[Any] | dict | Dataset,
    paths: str | os.PathLike,
    chunks=None,
    chunk_grid: bool = True,
    use_dual: bool | None = False,
    grid_kwargs: dict[str, Any] | None = None,
    **kwargs: dict[str, Any],
) -> UxDataset:
    """Wraps ``xarray.open_dataset()`` to support reading in a grid and
    multiple data files together.

    Parameters
    ----------
    grid_filename_or_obj : str | os.PathLike[Any] | dict | xr.dataset
        Strings and Path objects are interpreted as a path to a grid file. Xarray Datasets assume that
        each member variable is in the UGRID conventions and will be used to create a ``ux.Grid``. Similarly, a dictionary
        containing UGRID variables can be used to create a ``ux.Grid``
    paths : string, required
        Either a string glob in the form ``"path/to/my/files/*.nc"`` or an explicit
        list of files to open. It is the same ``paths`` in ``xarray.open_mfdataset``.
    chunks : int, dict, 'auto' or None, default: None
        If provided, used to load the grid into dask arrays.

        - ``chunks="auto"`` will use dask ``auto`` chunking taking into account the
          engine preferred chunks.
        - ``chunks=None`` skips using dask, which is generally faster for
          small arrays.
        - ``chunks=-1`` loads the data with dask using a single chunk for all arrays.
        - ``chunks={}`` loads the data with dask using the engine's preferred chunk
          size, generally identical to the format's chunk size. If not available, a
          single chunk for all arrays.
    chunk_grid : bool, default: True
        If valid chunks are passed in, determines whether to also apply the same chunks to the attached ``Grid``
    use_dual : bool, optional
        Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is mpas
    grid_kwargs : dict, optional
        Additional arguments passed on to ``ux.open_grid`` when opening up a Grid File.
    **kwargs
        Additional arguments passed on to ``xarray.open_mfdataset``.
        Refer to the :func:`xarray docs <xarray.open_mfdataset>`
        for accepted keyword arguments.

    Returns
    -------

    object : uxarray.UxDataset
        Dataset with the unstructured grid.

    Examples
    --------

    Open grid file along with multiple data files (two or more)

    >>> import uxarray as ux

    1. Open from an explicit list of dataset files

    >>> ux_ds = ux.open_mfdataset(
    ...     "grid_filename.g", "grid_filename_vortex_1.nc", "grid_filename_vortex_2.nc"
    ... )

    2. Open from a string glob

    >>> ux_ds = ux.open_mfdataset("grid_filename.g", "grid_filename_vortex_*.nc")
    """
    import xarray as xr

    if grid_kwargs is None:
        grid_kwargs = {}

    # Construct a Grid, validate parameters, and correct chunks
    uxgrid, corrected_chunks = _get_grid(
        grid_filename_or_obj, chunks, chunk_grid, use_dual, grid_kwargs, **kwargs
    )

    # Load the data as a Xarray Dataset
    ds = xr.open_mfdataset(paths, chunks=corrected_chunks, **kwargs)

    # Map original dimensions to the UGRID conventions
    ds = _map_dims_to_ugrid(ds, uxgrid._source_dims_dict, uxgrid)

    # Create a UXarray Dataset by linking the Xarray Dataset with a UXarray Grid
    return UxDataset(ds, uxgrid=uxgrid, source_datasets=str(paths))


def _get_grid(
    grid_filename_or_obj, chunks, chunk_grid, use_dual, grid_kwargs, **kwargs
):
    """Utility function to validate the input parameters and return a Grid."""
    if "latlon" in kwargs:
        warn(
            "'latlon is no longer a supported parameter",
            DeprecationWarning,
            stacklevel=2,
        )
        grid_kwargs["latlon"] = kwargs["latlon"]

    grid_kwargs["data_chunks"] = chunks
    grid_kwargs["chunk_grid"] = chunk_grid
    grid_kwargs["return_chunks"] = True

    # Create a Grid
    return open_grid(grid_filename_or_obj, use_dual=use_dual, **grid_kwargs)


def concat(objs, *args, **kwargs):
    # Ensure there is at least one object to concat.
    import xarray as xr

    if not objs:
        raise ValueError("No objects provided for concatenation.")

    ref_uxgrid = getattr(objs[0], "uxgrid", None)
    if ref_uxgrid is None:
        raise AttributeError("The first object does not have a 'uxgrid' attribute.")

    ref_id = id(ref_uxgrid)

    for i, obj in enumerate(objs):
        uxgrid = getattr(obj, "uxgrid", None)
        if uxgrid is None:
            raise AttributeError(
                f"Object at index {i} does not have a 'uxgrid' attribute."
            )
        if id(uxgrid) != ref_id:
            raise ValueError(f"Object at index {i} has a different 'uxgrid' attribute.")

    res = xr.concat(objs, *args, **kwargs)
    return UxDataset(res, uxgrid=uxgrid)
