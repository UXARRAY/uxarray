import os
import numpy as np
import xarray as xr

from collections.abc import Hashable, Iterable
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

from uxarray.grid import Grid
from uxarray.core.dataset import UxDataset
from uxarray.core.utils import _map_dims_to_ugrid, match_chunks_to_ugrid

from xarray.core.types import T_Chunks

from warnings import warn

if TYPE_CHECKING:
    from xarray.core.types import (
        ConcatOptions,
    )

    T_DataVars = Union[ConcatOptions, Iterable[Hashable]]


def open_grid(
    grid_filename_or_obj: str | os.PathLike[Any] | dict | xr.Dataset,
    chunks: T_Chunks = None,
    use_dual: Optional[bool] = False,
    **kwargs: Dict[str, Any],
):
    """Constructs and returns a ``Grid`` from a grid file.

    Parameters
    ----------
    grid_filename_or_obj : str | os.PathLike[Any] | dict | xr.dataset
        Strings and Path objects are interpreted as a path to a grid file. Xarray Datasets assume that
        each member variable is in the UGRID conventions and will be used to create a ``ux.Grid``. Simiarly, a dictionary
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
    use_dual:
        Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is MPAS
    **kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_dataset``. Refer to the
        [xarray
        docs](https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html)
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
        grid_ds = xr.open_dataset(grid_filename_or_obj, chunks=grid_chunks, **kwargs)
        grid = Grid.from_dataset(grid_ds, use_dual=use_dual)

    # Return the grid (and chunks, if requested) in a consistent manner.
    if return_chunks:
        return grid, chunks
    else:
        return grid


def open_dataset(
    grid_filename_or_obj: str | os.PathLike[Any] | dict | xr.Dataset,
    filename_or_obj: str | os.PathLike[Any],
    chunks: T_Chunks = None,
    chunk_grid: bool = True,
    use_dual: Optional[bool] = False,
    grid_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> UxDataset:
    """Wraps ``xarray.open_dataset()`` for loading in a dataset paired with a grid file.

    Parameters
    ----------
    grid_filename_or_obj: str | os.PathLike[Any] | dict | xr.dataset
        Strings and Path objects are interpreted as a path to a grid file. Xarray Datasets assume that
        each member variable is in the UGRID conventions and will be used to create a ``ux.Grid``. Simiarly, a dictionary
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
    chunk_grid: bool, default: True
        If valid chunks are passed in, determines whether to also apply the same chunks to the attached ``Grid``
    use_dual: bool, optional
        Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is MPAS
    grid_kwargs : dict, optional
        Additional arguments passed on to ``ux.open_grid`` when opening up a Grid File.
    **kwargs:
        Additional arguments passed on to ``xarray.open_dataset``. Refer to the
        [xarray
        docs](https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html)
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
    ds = xr.open_dataset(filename_or_obj, chunks=corrected_chunks, **kwargs)

    # Map original dimensions to the UGRID conventions
    ds = _map_dims_to_ugrid(ds, uxgrid._source_dims_dict, uxgrid)

    # Create a UXarray Dataset by linking the Xarray Dataset with a UXarray Grid
    return UxDataset(ds, uxgrid=uxgrid, source_datasets=str(filename_or_obj))


def open_mfdataset(
    grid_filename_or_obj: str | os.PathLike[Any] | dict | xr.Dataset,
    paths: Union[str, os.PathLike],
    chunks: T_Chunks = None,
    chunk_grid: bool = True,
    use_dual: Optional[bool] = False,
    grid_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> UxDataset:
    """Wraps ``xarray.open_dataset()`` to support reading in a grid and
    multiple data files together.

    Parameters
    ----------
    grid_filename_or_obj : str | os.PathLike[Any] | dict | xr.dataset
        Strings and Path objects are interpreted as a path to a grid file. Xarray Datasets assume that
        each member variable is in the UGRID conventions and will be used to create a ``ux.Grid``. Simiarly, a dictionary
        containing UGRID variables can be used to create a ``ux.Grid``
    paths : string, required
        Either a string glob in the form "path/to/my/files/*.nc" or an explicit
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
    chunk_grid: bool, default: True
        If valid chunks are passed in, determines whether to also apply the same chunks to the attached ``Grid``
    use_dual: bool, optional
        Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is mpas
    grid_kwargs : dict, optional
        Additional arguments passed on to ``ux.open_grid`` when opening up a Grid File.
    **kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_mfdataset``. Refer to the
        [xarray
        docs](https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html)
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


concat.__doc__ = xr.concat.__doc__
