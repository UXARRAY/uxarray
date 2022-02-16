def read_ugrid(self, filename):
    print("Reading ugrid file: ", filename)
    # simply return the xarray object loaded
    self.in_ds = self.ext_ds
    return self.ext_ds


# Write a uxgrid to a file with specified format.
def write_ugrid(self, outfile):
    self.in_ds.to_netcdf(outfile)
