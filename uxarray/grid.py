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
import numpy as np
from pathlib import PurePath
import os


# grid class
class Grid:
    # Import methods
    from ._populate_exodus import populate_exo_data
    from ._populate_exodus2 import populate_exo2_data
    from ._populate_ugrid import read_and_populate_ugrid_data
    from ._populate_shpfile import read_and_populate_shpfile_data
    from ._populate_scrip import populate_scrip_data

    # Load the grid file specified by file.
    # The routine will automatically detect if it is a UGrid, SCRIP, Exodus, or shape file.
    def __init__(self, *args, **kwargs):

        # initialize possible variables
        self.filepath = None
        self.gridspec = None
        self.vertices = None
        self.islatlon = None
        self.concave = None
        self.meshFileType = None

        self.grid_ds = None

        # determine initialization type
        in_type = type(args[0])
        # print(in_type, "intype", os.getcwd(), args[0], os.path.isfile(args[0]))

        # initialize for vertices
        if in_type is np.ndarray:
            self.vertices = args[0]
            self._init_vert()

        # initialize for file
        elif in_type is str and os.path.isfile(args[0]):
            self.filepath = args[0]
            self._init_file()

        # initialize for gridspec (??)
        elif in_type is str:
            self.gridspec = args[0]
            self._init_gridspec()

        else:
            pass

    # vertices init
    def _init_vert(self):
        print("initializing with vertices")

    # gridspec init
    def _init_gridspec(self):
        print("initializing with gridspec")

    # file init
    def _init_file(self):
        print("initializing from file")

        # find the file type
        try:
            # extract the file name and extension
            path = PurePath(self.filepath)
            file_extension = path.suffix

            # open dataset with xarray
            self.grid_ds = xr.open_dataset(self.filepath)
        except (TypeError, AttributeError) as e:
            msg = str(e) + ': {}'.format(self.filepath)
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
                msg = str(e) + ': {}'.format(self.filepath)
                print(msg)
                raise RuntimeError(msg)
                exit

        print("Done loading: ", self.filepath)

        # Detect mesh file type:
        # if ds has coordx - call it exo1 format
        # if ds has coord - call it exo2 format
        # if ds has grid_size - call it SCRIP format
        # if ds has ? read as shape file format
        try:
            self.grid_ds.coordx
            self.meshFileType = "exo1"
        except AttributeError as e:
            pass
        try:
            self.grid_ds.grid_center_lon
            self.meshFileType = "scrip"
        except AttributeError as e:
            pass
        try:
            self.grid_ds.coord
            self.meshFileType = "exo2"
        except AttributeError as e:
            pass

        if self.meshFileType is None:
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
            self.read_and_populate_ugrid_data(self.filepath)
        elif self.meshFileType == "shp":
            self.read_and_populate_shpfile_data(self.filepath)

    # renames the grid file
    def rename_file(self, filename):
        path = PurePath(self.filepath)
        old_filename = path.name
        new_filepath = path.parent / filename
        self.filepath = str(new_filepath)

    # A flag indicating the grid is a latitude longitude grid.
    def islatlon(self):
        return self.istlatlon

    # A flag indicating the grid contains concave faces.
    def isconcave(self):
        return self.isconcave

    # DataSet containing uxarray.Grid properties
    def ds(self):
        return self.grid_ds

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
