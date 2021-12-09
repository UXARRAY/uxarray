# Grid class and helper functions
# This software is provided under a slightly modified version
# of the Apache Software License. See the accompanying LICENSE file
# for more information.
#
# Description:
#
#
from logging import raiseExceptions
import xarray as xr
import os


# grid class
class Grid:

    #Import methods
    from ._populate_exodus import populate_exo_data
    from ._populate_exodus2 import populate_exo2_data
    from ._populate_ugrid import read_and_populate_ugrid_data
    from ._populate_shpfile import read_and_populate_shpfile_data
    from ._populate_scrip import populate_scrip_data

    # Load the grid file specified by file.
    # The routine will automatically detect if it is a UGrid, SCRIP, Exodus, or shape file.
    def __init__(self, filename):
        self.filename = filename
        self.islatlon = False
        self.concave = False
        self.meshFileType = ""

        # find the file type
        try:
            # extract the file name and extension
            split = os.path.splitext(filename)
            file_name = split[0]
            file_extension = split[1]
            # open dataset with xarray
            self.grid_ds = xr.open_dataset(filename)
        except (TypeError, AttributeError) as e:
            msg = str(e) + ': {}'.format(filename)
            print(msg)
            raise RuntimeError(msg)
            exit
        except (RuntimeError, OSError) as e:
            # check if this is a ugrid file
            # we won't use xarray to load that file
            if file_extension == ".ugrid":
                self.meshFileType = "ugrid"
            elif file_extension == ".shp":
                self.meshFileType = "shp"
            else:
                msg = str(e) + ': {}'.format(filename)
                print(msg)
                raise RuntimeError(msg)
                exit

        print("Done loading: ", filename)

        # Detect mesh file type:
        # if ds has coordx - call it exo1 format
        # if ds has coord - call it exo2 format
        # if ds has grid_size - call it SCRIP format
        # if ds has ? read as shape file format
        try:
            self.grid_ds.coordx
            self.meshFileType = "exo1"
        except (AttributeError) as e:
            pass
        try:
            self.grid_ds.grid_center_lon
            self.meshFileType = "scrip"
        except (AttributeError) as e:
            pass
        try:
            self.grid_ds.coord
            self.meshFileType = "exo2"
        except (AttributeError) as e:
            pass

        if self.meshFileType == "":
            print("mesh file not supported")

        print("Mesh file type is", self.meshFileType)

        # Now set Mesh2 ds
        # exodus file is cartesian grid, must convert to lat/lon?
        # use - pyproj https://gis.stackexchange.com/questions/78838/converting-projected-coordinates-to-lat-lon-using-python?

        if self.meshFileType == "exo1":
            self.populate_exo_data(self.grid_ds)
        elif self.meshFileType == "exo2":
            self.populate_exo2_data(self.grid_ds)
        elif self.meshFileType == "scrip":
            self.populate_scrip_data(self.grid_ds)
        elif self.meshFileType == "ugrid":
            self.read_and_populate_ugrid_data(filename)
        elif self.meshFileType == "shp":
            self.read_and_populate_shpfile_data(filename)

    # renames the grid
    def rename_file(self, filename):
        self.filename = filename

    # A flag indicating the grid is a latitude longitude grid.
    def islatlon(self):
        return self.istlatlon

    # A flag indicating the grid contains concave faces.
    def isconcave(self):
        return self.isconcave

    # DataSet containing uxarray.Grid properties
    def ds(self):
        return self.grid_ds
        # self.grid_ds["Mesh2_node_x"] = self.grid_ds.coordx.values
        # self.grid_ds["Mesh2_node_y"] = self.grid_ds.coordy.values
        # self.grid_ds["Mesh2_node_z"] = self.grid_ds.coordz.values

        # set faces

        # self.grid_ds["Mesh2_faces"]

    # # Create a grid with one face with vertices specified by the given argument.
    # def __init__(self, verts):
    #     pass

    # Write a uxgrid to a file with specified format.
    def write(self, outfile, format):
        pass

    # Calculate the area of all faces.
    def calculate(self):
        pass

    # Build the node-face connectivity array.
    def build_node_face_connectivity(self):
        pass

    # Build the edge-face connectivity array.
    def build_edge_face_connectivity(self):
        pass

    # Build the array of latitude-longitude bounding boxes.
    def buildlatlon_bounds(self):
        pass

    # Validate that the grid conforms to the UXGrid standards.
    def validate(self):
        pass
