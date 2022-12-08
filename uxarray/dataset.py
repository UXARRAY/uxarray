"""UXarray dataset module."""

# from .grid import *
from .helpers import parse_grid_type

from typing import Any, Dict, Optional

import numpy as np
import xarray as xr


def open_dataset(grid_filename_or_obj: str,
                 filename_or_obj: str,
                 gridspec: Optional[str] = None,
                 vertices: Optional[list] = None,
                 islatlon: Optional[bool] = False,
                 isconcave: Optional[bool] = False,
                 source_grid: Optional[str] = None,
                 **kwargs: Dict[str, Any]) -> xr.Dataset:
    """Wraps ``xarray.open_dataset()``, given a single grid topology file with
    a single dataset file with corresponding data.

    Parameters
    ----------

    grid_filename_or_obj : string, required
        Grid file is the first argument, which should be the file that
        houses the unstructured grid definition. It should be compatible to
        be opened with xarray.open_dataset (e.g. path to a file in the local
        storage, OpenDAP URL, etc).

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

    **kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_dataset``. Refer to the
        [xarray
        docs](https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html)
        for accepted keyword arguments.

    Returns
    -------

    object : xr.Dataset
        Dataset with the unstructured grid.

    Examples
    --------
    Open grid file only

    >>> mesh = ux.open_dataset("grid_filename.g")

    Open grid file along with data

    >>> mesh_and_data = ux.open_dataset("grid_filename.g", "grid_filename_vortex.nc")

    Open grid file along with multiple data files (two or more)

    >>> mesh_and_data_2 = ux.open_dataset("grid_filename.g", "grid_filename_vortex_1.nc", "grid_filename_vortex_2.nc")

    Open grid file along with a list of data files

    >>> data_files = ["grid_filename_vortex_1.nc", "grid_filename_vortex_2.nc", "grid_filename_vortex_3.nc"]
    >>> mesh_and_data = ux.open_dataset("grid_filename.g", *data_files)
    """

    # todo
    # unpack kwargs
    # sets default values for all kwargs to None
    # kwargs_list = [
    #     'gridspec', 'vertices', 'islatlon', 'isconcave', 'source_grid'
    # ]
    # for key in kwargs_list:
    #     setattr(self, key, kwargs.get(key, None))

    grid_ds = xr.open_dataset(grid_filename_or_obj, decode_times=False, **kwargs)  # type: ignore
    ds = xr.open_dataset(filename_or_obj, decode_times=False, **kwargs)  # type: ignore

    mesh_filetype, dataset = parse_grid_type(grid_filename_or_obj, **kwargs)

    # Determine the source data sets regarding args
    source_datasets = np.array(args) if len(args) > 0 else None

    # Construct the GridAccessor object
    ux_grid = GridAccessor(dataset=dataset,
                   mesh_filetype=mesh_filetype,
                   source_grid=grid_filename_or_obj,
                   source_datasets=source_datasets)

    # If there are additional data file(s) corresponding to this grid, merge them
    # into the dataset
    if len(args) > 0:
        # load all the datafiles using mfdataset
        all_data = xr.open_mfdataset(args)
        # merge data with grid ds
        ux_grid.ds = xr.merge([ux_grid.ds, all_data])

    return ux_grid