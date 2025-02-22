import os
import numpy as np
import xarray as xr

from typing import Any, Dict, Optional, Union

from uxarray.grid import Grid
from uxarray.core.dataset import UxDataset
from uxarray.core.utils import _map_dims_to_ugrid, match_chunks_to_ugrid

from xarray.core.types import T_Chunks

from warnings import warn


def open_grid(
    grid_filename_or_obj: str | os.PathLike[Any] | dict | xr.Dataset,
    chunks: T_Chunks = None,
    use_dual: Optional[bool] = False,
    **kwargs: Dict[str, Any],
) -> Grid:
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
        Selects whether the Dual grid should be used for supported grids: MPAS

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

    # Special case for FESOM2 ASCII Dataset (stored as a directory)
    if isinstance(grid_filename_or_obj, (str, os.PathLike)) and os.path.isdir(
        grid_filename_or_obj
    ):
        nod2d_path = os.path.join(grid_filename_or_obj, "nod2d.out")
        elem2d_path = os.path.join(grid_filename_or_obj, "elem2d.out")

        if os.path.isfile(nod2d_path) and os.path.isfile(elem2d_path):
            return Grid.from_dataset(grid_filename_or_obj)

        else:
            raise FileNotFoundError(
                f"The directory '{grid_filename_or_obj}' must contain both 'nod2d.out' and 'elem2d.out'."
            )

    elif isinstance(grid_filename_or_obj, dict):
        # unpack the dictionary and construct a grid from topology
        return Grid.from_topology(**grid_filename_or_obj)

    elif isinstance(grid_filename_or_obj, (list, tuple, np.ndarray, xr.DataArray)):
        # construct Grid from face vertices
        return Grid.from_face_vertices(grid_filename_or_obj, **kwargs)

    # TODO:
    if "data_chunks" in kwargs:
        # Special case for when chunks are passed in from open_dataset()
        chunks = match_chunks_to_ugrid(grid_filename_or_obj, kwargs["data_chunks"])
        del kwargs["data_chunks"]
    elif chunks is not None:
        chunks = match_chunks_to_ugrid(grid_filename_or_obj, chunks)

    if isinstance(grid_filename_or_obj, xr.Dataset):
        # construct a grid from a dataset file
        # TODO: insert/rechunk here?
        uxgrid = Grid.from_dataset(grid_filename_or_obj, use_dual=use_dual)

    # attempt to use Xarray directly for remaining input types
    else:
        grid_ds = xr.open_dataset(grid_filename_or_obj, chunks=chunks, **kwargs)
        uxgrid = Grid.from_dataset(grid_ds, use_dual=use_dual)

    return uxgrid


def open_dataset(
    grid_filename_or_obj: Union[
        str, os.PathLike, xr.DataArray, np.ndarray, list, tuple, dict
    ],
    filename_or_obj: str,
    use_dual: Optional[bool] = False,
    grid_kwargs: Optional[Dict[str, Any]] = {},
    **kwargs: Dict[str, Any],
) -> UxDataset:
    """Wraps ``xarray.open_dataset()`` to support reading in a grid and data
    file together.

    Parameters
    ----------

    grid_filename_or_obj : string, required
        String or Path object as a path to a netCDF file or an OpenDAP URL that
        stores the unstructured grid definition that the dataset belongs to. It
        is read similar to ``filename_or_obj`` in ``xarray.open_dataset``.

    filename_or_obj : string, required
        String or Path object as a path to a netCDF file or an OpenDAP URL that
        stores the actual data set. It is the same ``filename_or_obj`` in
        ``xarray.open_dataset``.


    use_dual: bool, optional
        Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is MPAS

    grid_kwargs : Dict[str, Any], optional
        Additional arguments passed on to ``xarray.open_dataset`` when opening up a Grid File. Refer to the
        [xarray
        docs](https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html)
        for accepted keyword arguments.

    **kwargs : Dict[str, Any]
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

    Open dataset with a grid topology file

    >>> import uxarray as ux
    >>> ux_ds = ux.open_dataset("grid_filename.g", "grid_filename_vortex.nc")
    :param grid_kwargs:
    """

    if "latlon" in kwargs:
        # TODO: Raise deprecation warning
        grid_kwargs["latlon"] = kwargs["latlon"]

    # TODO:
    if "chunks" in kwargs and "chunks" not in grid_kwargs:
        chunks = kwargs["chunks"]
        grid_kwargs["data_chunks"] = chunks

    # Grid definition
    uxgrid = open_grid(grid_filename_or_obj, use_dual=use_dual, **grid_kwargs)

    # if "chunks" in kwargs:
    #     # correctly chunk standardized ugrid dimension names
    #     source_dims_dict = uxgrid._source_dims_dict
    #     for original_grid_dim, ugrid_grid_dim in source_dims_dict.items():
    #         if ugrid_grid_dim in kwargs["chunks"]:
    #             kwargs["chunks"][original_grid_dim] = kwargs["chunks"][ugrid_grid_dim]

    # UxDataset
    ds = xr.open_dataset(filename_or_obj, **kwargs)  # type: ignore

    # map each dimension to its UGRID equivalent
    # TODO: maybe issues here?
    ds = _map_dims_to_ugrid(ds, uxgrid._source_dims_dict, uxgrid)

    uxds = UxDataset(ds, uxgrid=uxgrid, source_datasets=str(filename_or_obj))
    # UxDataset.from_xarray(ds, uxgrid=uxgrid, source_d

    return uxds


def open_mfdataset(
    grid_filename_or_obj: Union[
        str, os.PathLike, xr.DataArray, np.ndarray, list, tuple, dict
    ],
    paths: Union[str, os.PathLike],
    use_dual: Optional[bool] = False,
    grid_kwargs: Optional[Dict[str, Any]] = {},
    **kwargs: Dict[str, Any],
) -> UxDataset:
    """Wraps ``xarray.open_dataset()`` to support reading in a grid and
    multiple data files together.

    Parameters
    ----------

    grid_filename_or_obj : string, required
        String or Path object as a path to a netCDF file or an OpenDAP URL that
        stores the unstructured grid definition that the dataset belongs to. It
        is read similar to ``filename_or_obj`` in ``xarray.open_dataset``.

    paths : string, required
        Either a string glob in the form "path/to/my/files/*.nc" or an explicit
        list of files to open. It is the same ``paths`` in ``xarray.open_mfdataset``.


    use_dual: bool, optional
        Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is mpas

    grid_kwargs : Dict[str, Any], optional
        Additional arguments passed on to ``xarray.open_dataset`` when opening up a Grid File. Refer to the
        [xarray
        docs](https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html)
        for accepted keyword arguments.

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

    if "source_grid" in kwargs.keys():
        warn(
            "source_grid is no longer a supported kwarg",
            DeprecationWarning,
            stacklevel=2,
        )

    # Grid definition
    uxgrid = open_grid(grid_filename_or_obj, use_dual=use_dual, **grid_kwargs)

    # if "chunks" in kwargs:
    #     # correctly chunk standardized ugrid dimension names
    #     source_dims_dict = uxgrid._source_dims_dict
    #     for original_grid_dim, ugrid_grid_dim in source_dims_dict.items():
    #         if ugrid_grid_dim in kwargs["chunks"]:
    #             kwargs["chunks"][original_grid_dim] = kwargs["chunks"][ugrid_grid_dim]

    # UxDataset
    ds = xr.open_mfdataset(paths, **kwargs)  # type: ignore

    # map each dimension to its UGRID equivalent
    ds = _map_dims_to_ugrid(ds, uxgrid._source_dims_dict, uxgrid)

    uxds = UxDataset(ds, uxgrid=uxgrid, source_datasets=str(paths))

    return uxds
