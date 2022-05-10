import os
import xarray as xr
from pathlib import PurePath


def determine_file_type(filepath):
    """Checks file path and contents to determine file type. Supports detection
    of UGrid, SCRIP, Exodus and shape file.

    Parameters
    ----------

    filepath : :class:`string`: required
       Filepath of the file for which the filetype is to be determined.

    Returns
    -------
    mesh_filetype : :class:`string`:
        String describing the file type (ug, exo, scrip or shp)
        to be used by Grid.py

    Raises:
       RuntimeError: Invalid file type
    """
    msg = ""
    # Extract file name and extension, read into xarray.Dataset to prep
    path = PurePath(filepath)
    file_extension = path.suffix
    # ext_ds = xr.open_dataset(filepath, mask_and_scale=False)

    # Shapefiles (not supported, placeholder)
    # if file_extension == ".shp":
    #     mesh_filetype = "shp"

    # NetCDF style files
    try:
        ext_ds = xr.open_dataset(filepath, mask_and_scale=False)
        if "coord" in ext_ds.data_vars:
            mesh_filetype = "exo"
            print("exo")

        elif "coordx" in ext_ds.data_vars:
            mesh_filetype = "exo"

        # SCRIP files
        elif 'grid_area' in ext_ds.data_vars:
            mesh_filetype = "scrip"
            print("scrip")

        # UGRID files
        elif "Mesh2" in ext_ds.data_vars:
            mesh_filetype = "ugrid"

        # Shape files
        # elif file_extension == ".shp":
        #     mesh_filetype = "shp"

        return mesh_filetype

    except:
        # Update with the correct exceptions, not sure which will be the best from below
        print("Unrecognized file type")


def determine_shapefile(filepath):
    # Extract file name and extension, read into xarray.Dataset to prep
    path = PurePath(filepath)
    file_extension = path.suffix
    # ext_ds = xr.open_dataset(filepath, mask_and_scale=False)

    # Shapefiles (not supported, placeholder)
    try:
        if file_extension == ".shp":
            mesh_filetype = "shp"
            return mesh_filetype
    except:
        print("File suffix is not shapefile")
