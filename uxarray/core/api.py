"""UXarray dataset module."""

import os

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import xarray as xr

from uxarray.core.grid import Grid

from uxarray.core.dataset import UxDataset


def open_grid(grid_filename_or_obj: Union[str, Path, xr.DataArray, np.ndarray,
                                          list, tuple],
              gridspec: Optional[str] = None,
              islatlon: Optional[bool] = False,
              isconcave: Optional[bool] = False,
              use_dual: Optional[bool] = False,
              **kwargs: Dict[str, Any]) -> xr.Dataset:
    """Creates a ``uxarray.core.Grid`` object from a grid topology definition.

    Parameters
    ----------

    grid_filename_or_obj : string, xarray.Dataset, ndarray, list, tuple, required
        String or Path object as a path to a netCDF file or an OpenDAP URL that
        stores the unstructured grid topology/definition. It is read similar to
        ``filename_or_obj`` in ``xarray.open_dataset``. Otherwise, either
        ``xr.DataArray``, ``np.ndarray``, ``list``, or ``tuple`` as a vertices
        object to define the grid.

    islatlon : bool, optional
            Specify if the grid is lat/lon based

    isconcave: bool, optional
        Specify if this grid has concave elements (internal checks for this are possible)

    gridspec: str, optional
        Specifies gridspec

    source_grid: str, optional
        Path or URL to the source grid file. For diagnostic/reporting purposes only.

    use_dual: bool, optional
            Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is mpas

    **kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_dataset``. Refer to the
        [xarray
        docs](https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html)
        for accepted keyword arguments.

    Returns
    -------

    object : uxarray.core.Grid
        Dataset with the unstructured grid.

    Examples
    --------

    Open dataset with a grid topology file

    >>> import uxarray as ux
    >>> uxgrid = ux.open_grid("grid_filename.g")
    """

    ## Grid definition
    if isinstance(grid_filename_or_obj,
                  (list, tuple, np.ndarray, xr.DataArray)):
        uxgrid = Grid(grid_filename_or_obj,
                      gridspec=gridspec,
                      islatlon=islatlon,
                      isconcave=isconcave,
                      use_dual=use_dual)
    else:
        grid_ds = xr.open_dataset(grid_filename_or_obj,
                                  decode_times=False,
                                  **kwargs)  # type: ignore

        uxgrid = Grid(grid_ds,
                      gridspec=gridspec,
                      islatlon=islatlon,
                      isconcave=isconcave,
                      source_grid=grid_filename_or_obj,
                      use_dual=use_dual)

    return uxgrid


def open_dataset(grid_filename_or_obj: str,
                 filename_or_obj: str,
                 gridspec: Optional[str] = None,
                 vertices: Optional[list] = None,
                 islatlon: Optional[bool] = False,
                 isconcave: Optional[bool] = False,
                 use_dual: Optional[bool] = False,
                 **kwargs: Dict[str, Any]) -> xr.Dataset:
    """Wraps ``xarray.open_dataset()``, given a grid topology definition with a
    single dataset file or object with corresponding data.

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

    islatlon : bool, optional
            Specify if the grid is lat/lon based

    isconcave: bool, optional
        Specify if this grid has concave elements (internal checks for this are possible)

    gridspec: str, optional
        Specifies gridspec

    source_grid: str, optional
        Path or URL to the source grid file. For diagnostic/reporting purposes only.

    use_dual: bool, optional
            Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is mpas

    **kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_dataset``. Refer to the
        [xarray
        docs](https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html)
        for accepted keyword arguments.

    Returns
    -------

    object : uxarray.core.UxDataset
        Dataset with the unstructured grid.

    Examples
    --------

    Open dataset with a grid topology file

    >>> import uxarray as ux
    >>> ux_ds = ux.open_dataset("grid_filename.g", "grid_filename_vortex.nc")
    """

    grid_ds = xr.open_dataset(grid_filename_or_obj,
                              decode_times=False,
                              **kwargs)  # type: ignore
    ds = xr.open_dataset(filename_or_obj, decode_times=False,
                         **kwargs)  # type: ignore

    ## Grid definition
    uxgrid = Grid(grid_ds,
                  gridspec=gridspec,
                  vertices=vertices,
                  islatlon=islatlon,
                  isconcave=isconcave,
                  source_grid=grid_filename_or_obj,
                  use_dual=use_dual)

    ## UxDataset
    uxds = UxDataset(ds, uxgrid=uxgrid, source_datasets=str(filename_or_obj))

    return uxds


def open_mfdataset(grid_filename_or_obj: str,
                   paths: Union[str, os.PathLike],
                   gridspec: Optional[str] = None,
                   vertices: Optional[list] = None,
                   islatlon: Optional[bool] = False,
                   isconcave: Optional[bool] = False,
                   use_dual: Optional[bool] = False,
                   **kwargs: Dict[str, Any]) -> xr.Dataset:
    """Wraps ``xarray.open_mfdataset()``, given a single grid topology file
    with multiple dataset paths with corresponding data.

    Parameters
    ----------

    grid_filename_or_obj : string, required
        String or Path object as a path to a netCDF file or an OpenDAP URL that
        stores the unstructured grid definition that the dataset belongs to. It
        is read similar to ``filename_or_obj`` in ``xarray.open_dataset``.

    paths : string, required
        Either a string glob in the form "path/to/my/files/*.nc" or an explicit
        list of files to open. It is the same ``paths`` in ``xarray.open_mfdataset``.

    islatlon : bool, optional
            Specify if the grid is lat/lon based

    isconcave: bool, optional
        Specify if this grid has concave elements (internal checks for this are possible)

    gridspec: str, optional
        Specifies gridspec

    source_grid: str, optional
        Path or URL to the source grid file. For diagnostic/reporting purposes only.

    use_dual: bool, optional
            Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is mpas

    **kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_mfdataset``. Refer to the
        [xarray
        docs](https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html)
        for accepted keyword arguments.

    Returns
    -------

    object : uxarray.core.UxDataset
        Dataset with the unstructured grid.

    Examples
    --------

    Open grid file along with multiple data files (two or more)

    >>> import uxarray as ux
    >>> ux_ds = ux.open_mfdataset("grid_filename.g", "grid_filename_vortex_1.nc", "grid_filename_vortex_2.nc")
    """

    grid_ds = xr.open_dataset(grid_filename_or_obj,
                              decode_times=False,
                              **kwargs)  # type: ignore
    ds = xr.open_mfdataset(paths, decode_times=False, **kwargs)  # type: ignore

    ## Grid definition
    uxgrid = Grid(grid_ds,
                  gridspec=gridspec,
                  vertices=vertices,
                  islatlon=islatlon,
                  isconcave=isconcave,
                  source_grid=grid_filename_or_obj,
                  use_dual=use_dual)

    ## UxDataset
    uxds = UxDataset(ds, uxgrid=uxgrid, source_datasets=str(paths))

    return uxds
