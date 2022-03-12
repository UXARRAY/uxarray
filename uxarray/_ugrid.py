import xarray as xr


def read_ugrid(filepath):
    """UGRID file reader.

    Parameters: string, required
        Name of file to be read

    Returns: the xarray Dataset loaded during init.
    """

    ext_ds = xr.open_dataset(filepath, mask_and_scale=False)
    # simply return the xarray object loaded
    in_ds = ext_ds
    ext_ds.close()
    return in_ds


# Write a uxgrid to a file with specified format.
def write_ugrid(in_ds, outfile):
    """UGRID file writer.
    Parameters
    ----------
    in_ds : xarray.Dataset
        Dataset to be written to file
    outfile : string, required
        Name of output file

    Uses to_netcdf from xarray object.
    """
    print("Writing ugrid file: ", outfile)
    in_ds.to_netcdf(outfile)
