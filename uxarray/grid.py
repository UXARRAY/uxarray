"""uxarray grid module."""
import os
import xarray as xr
import numpy as np
from pathlib import PurePath


class Grid:
    """The Uxarray Grid object class that describes an unstructured grid.

    Examples
    ========
    import uxarray as ux
    # open an exodus file with Grid object
    mesh = ux.Grid("filename.g")

    # save as ugrid file
    mesh.saveas("outfile.ug")
    """

    # Import read/write methods from python modules in this folder
    from ._exodus import read_exodus, write_exodus
    from ._ugrid import read_ugrid, write_ugrid
    from ._shpfile import read_shpfile
    from ._scrip import read_scrip

    def __init__(self, *args, **kwargs):
        # TODO: fix when adding/exercising gridspec
        self.filepath = None
        self.gridspec = None
        self.vertices = None
        self.islatlon = None
        self.concave = None
        self.mesh_filetype = None

        # this xarray variable holds an existing external netcdf file with loaded with xarray
        self.ext_ds = None

        # internal uxarray representation of mesh stored in internal object in_ds
        self.in_ds = xr.Dataset()
        """Initialize grid variables, decide if loading happens via file, verts
        or gridspec If loading from file, initialization happens via the
        specified file.

        # TODO: Add or remove new Args/kwargs below as this develops further

        Args: input file name with extension as a string
              vertex coordinates that form one face.
        kwargs: optional
        can be a dict initializing specific variables:
        islatlon: bool: specify if the grid is lat/lon based:
        concave: bool: specify if this grid has concave elements (internal checks for this are possible)
        gridspec: bool: specifies gridspec
        mesh_filetype: string: specify the mesh file type, eg. exo, ugrid, shp etc

        example: kwargs = {"concave" : True, "islatlon" : True"}

        Raises:
            RuntimeError: File not found
        """
        # Parse the keyword arguments and initialize class attributes
        for key, value in kwargs.items():
            if key == "latlon":
                self.islatlon = value
            elif key == "gridspec":
                self.gridspec = value

        # determine initialization type - string signifies a file, numpy array signifies a list of verts
        in_type = type(args[0])

        # check if initializing from verts:
        try:
            if not os.path.isfile(args[0]) and in_type is not np.ndarray:
                raise FileNotFoundError("File not found: " + args[0])
            elif in_type is np.ndarray:
                self.vertices = args[0]
                self.__from_vert__()
        except ValueError as e:
            # initialize from vertices
            if in_type is np.ndarray:
                self.vertices = args[0]
                self.__from_vert__()
            else:
                print(
                    "Init called with args other than verts (ndarray) or filename (str)"
                )
                exit()

        # check if initialize from file:
        if in_type is str and os.path.isfile(args[0]):
            self.filepath = args[0]
            self.__from_file__()
        # initialize for gridspec
        elif in_type is str:
            self.gridspec = args[0]
            self.__from_gridspec__()
        else:
            # this may just be local initialization for with no options or options other than above
            pass

    # vertices init
    def __from_vert__(self):
        """Create a grid with one face with vertices specified by the given
        argument."""
        self.__init_mesh2__()
        self.in_ds.Mesh2.attrs['topology_dimension'] = self.vertices[0].size

        self.in_ds["Mesh2"].topology_dimension

        x_coord = self.vertices.transpose()[0]
        y_coord = self.vertices.transpose()[1]

        # single face with all nodes
        num_nodes = x_coord.size
        conn = list(range(0, num_nodes))
        conn = [conn]

        self.in_ds["Mesh2_node_x"] = xr.DataArray(data=xr.DataArray(x_coord),
                                                  dims=["nMesh2_node"])
        self.in_ds["Mesh2_node_y"] = xr.DataArray(data=xr.DataArray(y_coord),
                                                  dims=["nMesh2_node"])
        self.in_ds["Mesh2_face_nodes"] = xr.DataArray(
            data=xr.DataArray(conn),
            dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
            attrs={
                "cf_role": "face_node_connectivity",
                "_FillValue": -1,
                "start_index": 0
            })

    # TODO: gridspec init
    def __from_gridspec__(self):
        print("initializing with gridspec")

    # load mesh from a file
    def __from_file__(self):
        """Loads a mesh file Also, called by __init__ routine This routine will
        automatically detect if it is a UGrid, SCRIP, Exodus, or shape file.

        Raises:
            RuntimeError: Invalid file type
        """
        # call function to set mesh file type: self.mesh_filetype
        self.__find_type__()

        # call reader as per mesh_filetype
        if self.mesh_filetype == "exo":
            self.read_exodus(self.ext_ds)
        elif self.mesh_filetype == "scrip":
            self.read_scrip(self.ext_ds)
        elif self.mesh_filetype == "ugrid":
            self.read_ugrid(self.filepath)
        elif self.mesh_filetype == "shp":
            self.read_shpfile(self.filepath)
        else:
            raise RuntimeError("unknown file format:" + self.mesh_filetype)

    # helper function to find file type
    def __find_type__(self):
        """Checks file path and contents to determine file type Also, called by
        __init__ routine This routine will automatically detect if it is a
        UGrid, SCRIP, Exodus, or shape file.

        Raises:
            RuntimeError: Invalid file type
        """

        try:
            # extract the file name and extension
            path = PurePath(self.filepath)
            file_extension = path.suffix

            # try to open file with xarray
            self.ext_ds = xr.open_dataset(self.filepath, mask_and_scale=False)
            #
        except (TypeError, AttributeError) as e:
            msg = str(e) + ': {}'.format(self.filepath)
            print(msg)
            raise RuntimeError(msg)
            exit()
        except (RuntimeError, OSError) as e:
            # check if this is a shp file
            # we won't use xarray to load that file
            if file_extension == ".shp":
                self.mesh_filetype = "shp"
            else:
                msg = str(e) + ': {}'.format(self.filepath)
                print(msg)
                raise RuntimeError(msg)
                exit()
        except ValueError as e:
            # check if this is a shp file
            # we won't use xarray to load that file
            if file_extension == ".shp":
                self.mesh_filetype = "shp"
            else:
                msg = str(e) + ': {}'.format(self.filepath)
                print(msg)
                raise RuntimeError(msg)
                exit

        # Detect mesh file type, based on attributes of ext_ds:
        # if ext_ds has coordx or coord - call it exo format
        # if ext_ds has grid_size - call it SCRIP format
        # if ext_ds has ? read as shape file populate_scrip_dataformat
        # TODO: add detection of shpfile etc.
        if self.ext_ds is None:  # this is None, when xarray was not used to load the input mesh file
            print(
                "external file was not loaded by xarray, setting meshfile type to shp"
            )
            # TODO: add more checks for detecting shp file
            self.mesh_filetype = "shp"
        elif "coordx" in self.ext_ds.data_vars or "coord" in self.ext_ds.data_vars:
            self.mesh_filetype = "exo"
        elif "grid_center_lon" in self.ext_ds.data_vars:
            self.mesh_filetype = "scrip"
        elif "Mesh2" in self.ext_ds.data_vars:
            self.mesh_filetype = "ugrid"
        else:
            print("mesh file not supported")
            self.mesh_filetype = None

        return self.mesh_filetype

    # initialize mesh2 DataVariable for uxarray
    def __init_mesh2__(self):
        # set default values and initialize Datavariable "Mesh2" for uxarray
        self.in_ds["Mesh2"] = xr.DataArray(
            data=0,
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Topology data of unstructured mesh",
                "topology_dimension": -1,
                "node_coordinates": "Mesh2_node_x Mesh2_node_y Mesh2_node_z",
                "node_dimension": "nMesh2_node",
                "face_node_connectivity": "Mesh2_face_nodes",
                "face_dimension": "nMesh2_face"
            })

    # renames the grid file
    def saveas_file(self, filename):
        path = PurePath(self.filepath)
        new_filepath = path.parent / filename
        self.filepath = str(new_filepath)
        self.write_ugrid(self.filepath)
        print(self.filepath)

    # Calculate the area of all faces.
    def calculate_total_face_area(self):
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

    def write(self, outfile, format=""):
        if format == "":
            path = PurePath(outfile)
            format = path.suffix

        if format == ".ugrid" or format == ".ug":
            self.write_ugrid(outfile)
        elif format == ".g" or format == ".exo":
            self.write_exodus(outfile)
        else:
            print("Format not supported for writing. ", format)
