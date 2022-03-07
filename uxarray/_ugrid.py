import xarray as xr

def read_ugrid(filepath):
   """Returns the xarray Dataset loaded during init."""

   ext_ds = xr.open_dataset(filepath, mask_and_scale=False)
   in_ds = xr.Dataset()
   # simply return the xarray object loaded
   in_ds = ext_ds
   ext_ds.close()
   return in_ds

# Write a uxgrid to a file with specified format.
def write_ugrid(in_ds, outfile):
   """Function to write ugrid, uses to_netcdf from xarray object."""
   print("Writing ugrid file: ", outfile)
   in_ds.to_netcdf(outfile)