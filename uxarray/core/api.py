"""UXarray dataset module."""

import os
import numpy as np
import xarray as xr

from typing import Any, Dict, Optional, Union

from uxarray.grid import Grid
from uxarray.core.dataset import UxDataset
from uxarray.core.utils import _map_dims_to_ugrid

from warnings import warn


def open_grid(
    grid_filename_or_obj: Union[
        str, os.PathLike, xr.DataArray, np.ndarray, list, tuple, dict
    ],
    latlon: Optional[bool] = False,
    use_dual: Optional[bool] = False,
    **kwargs: Dict[str, Any],
) -> Grid:
    """Constructs and returns a ``Grid`` from a grid file.

    Parameters
    ----------

    grid_filename_or_obj : string, xarray.Dataset, ndarray, list, tuple, dict, required
        String or Path object as a path to a netCDF file or an OpenDAP URL that
        stores the unstructured grid topology/definition. It is read similar to
        ``filename_or_obj`` in ``xarray.open_dataset``. Otherwise, either
        ``xr.DataArray``, ``np.ndarray``, ``list``, or ``tuple`` as a vertices
        object to define the grid.

    latlon : bool, optional
            Specify if the grid is lat/lon based

    use_dual: bool, optional
        Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is mpas

    **kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_dataset``. Refer to the
        [xarray
        docs](https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html)
        for accepted keyword arguments.

    Returns
    -------

    uxgrid : uxarray.Grid
        Initialized Grid Object from Input Grid File

    Examples
    --------

    Open dataset with a grid topology file

    >>> import uxarray as ux
    >>> uxgrid = ux.open_grid("grid_filename.g")
    """

    if "source_grid" in kwargs.keys():
        warn(
            "source_grid is no longer a supported kwarg",
            DeprecationWarning,
            stacklevel=2,
        )

    if isinstance(grid_filename_or_obj, xr.Dataset):
        # construct a grid from a dataset file
        uxgrid = Grid.from_dataset(grid_filename_or_obj, use_dual=use_dual)

    elif isinstance(grid_filename_or_obj, dict):
        # unpack the dictionary and construct a grid from topology
        uxgrid = Grid.from_topology(**grid_filename_or_obj)

    elif isinstance(grid_filename_or_obj, (list, tuple, np.ndarray, xr.DataArray)):
        # construct Grid from face vertices
        uxgrid = Grid.from_face_vertices(grid_filename_or_obj, latlon=latlon)

    # attempt to use Xarray directly for remaining input types
    else:
        try:
            grid_ds = xr.open_dataset(grid_filename_or_obj, **kwargs)

            uxgrid = Grid.from_dataset(grid_ds, use_dual=use_dual)
        except ValueError:
            raise ValueError("Inputted grid_filename_or_obj not supported.")

    return uxgrid


def open_dataset(
    grid_filename_or_obj: Union[
        str, os.PathLike, xr.DataArray, np.ndarray, list, tuple, dict
    ],
    filename_or_obj: str,
    latlon: Optional[bool] = False,
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

    latlon : bool, optional
            Specify if the grid is lat/lon based

    use_dual: bool, optional
        Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is mpas

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

    if "source_grid" in kwargs.keys():
        warn(
            "source_grid is no longer a supported kwarg",
            DeprecationWarning,
            stacklevel=2,
        )

    # Grid definition
    uxgrid = open_grid(
        grid_filename_or_obj, latlon=latlon, use_dual=use_dual, **grid_kwargs
    )

    if "chunks" in kwargs:
        # correctly chunk standardized ugrid dimension names
        source_dims_dict = uxgrid._source_dims_dict
        for original_grid_dim, ugrid_grid_dim in source_dims_dict.items():
            if ugrid_grid_dim in kwargs["chunks"]:
                kwargs["chunks"][original_grid_dim] = kwargs["chunks"][ugrid_grid_dim]

    # UxDataset
    ds = xr.open_dataset(filename_or_obj, **kwargs)  # type: ignore

    # map each dimension to its UGRID equivalent
    ds = _map_dims_to_ugrid(ds, uxgrid._source_dims_dict, uxgrid)

    uxds = UxDataset(ds, uxgrid=uxgrid, source_datasets=str(filename_or_obj))

    return uxds


def open_mfdataset(
    grid_filename_or_obj: Union[
        str, os.PathLike, xr.DataArray, np.ndarray, list, tuple, dict
    ],
    paths: Union[str, os.PathLike],
    latlon: Optional[bool] = False,
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

    latlon : bool, optional
            Specify if the grid is lat/lon based

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
    uxgrid = open_grid(
        grid_filename_or_obj, latlon=latlon, use_dual=use_dual, **grid_kwargs
    )

    if "chunks" in kwargs:
        # correctly chunk standardized ugrid dimension names
        source_dims_dict = uxgrid._source_dims_dict
        for original_grid_dim, ugrid_grid_dim in source_dims_dict.items():
            if ugrid_grid_dim in kwargs["chunks"]:
                kwargs["chunks"][original_grid_dim] = kwargs["chunks"][ugrid_grid_dim]

    # UxDataset
    ds = xr.open_mfdataset(paths, **kwargs)  # type: ignore

    # map each dimension to its UGRID equivalent
    ds = _map_dims_to_ugrid(ds, uxgrid._source_dims_dict, uxgrid)

    uxds = UxDataset(ds, uxgrid=uxgrid, source_datasets=str(paths))

    return uxds
