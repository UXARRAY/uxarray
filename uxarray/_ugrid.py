import xarray as xr


def _read_ugrid(filepath):
    """UGRID file reader.

    Parameters: string, required
        Name of file to be read

    Returns: the xarray Dataset loaded during init.
    """

    # TODO: obtain and change to Mesh2 construct, see Issue #27
    # simply return the xarray object loaded
    return xr.open_dataset(filepath, mask_and_scale=False)


# Write a uxgrid to a file with specified format.
def _write_ugrid(ds, outfile):
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
