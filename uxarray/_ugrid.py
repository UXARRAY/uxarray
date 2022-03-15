import xarray as xr


def read_ugrid(filepath):
    """UGRID file reader.

    Parameters: string, required
        Name of file to be read

    Returns: the xarray Dataset loaded during init.
    """

    ext_ds = xr.open_dataset(filepath, mask_and_scale=False)
    # simply return the xarray object loaded
    ds = ext_ds
    return ds


# Write a uxgrid to a file with specified format.
def write_ugrid(ds, outfile):
    """UGRID file writer.
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be written to file
    outfile : string, required
        Name of output file

    Uses to_netcdf from xarray object.
    """
    print("Writing ugrid file: ", outfile)
    ds.to_netcdf(outfile)
