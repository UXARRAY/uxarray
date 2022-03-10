"""uxarray grid module."""
import os
import xarray as xr
import numpy as np
from warnings import warn
from pathlib import PurePath

# reader and writer imports
from ._exodus import read_exodus, write_exodus
from ._ugrid import read_ugrid, write_ugrid
from ._shapefile import read_shpfile
from ._scrip import read_scrip
from .helpers import determine_file_type


class Grid:
    """The Uxarray Grid object class that describes an unstructured grid.

    Examples
    ----------
    Open an exodus file with Uxarray Grid object

    >>> mesh = ux.Grid("filename.g")

    Save as ugrid file

    >>> mesh.saveas("outfile.ug")
    """

    def __init__(self,
                 *args,
                 filepath=None,
                 gridspec=None,
                 vertices=None,
                 concave=None,
                 islatlon=None,
                 mesh_filetype=None):
        """Initialize grid variables, decide if loading happens via file, verts
        or gridspec If loading from file, initialization happens via the
        specified file.

        # TODO: Add or remove new Args/kwargs below as this develops further

        Parameters
        ----------
        args : string, optional
            - Input file name with extension or
            - Vertex coordinates that form one face.
        islatlon : bool, optional
            Specify if the grid is lat/lon based:
        concave: bool, optional
            Specify if this grid has concave elements (internal checks for this are possible)
        gridspec: bool, optional
            Specifies gridspec
        mesh_filetype: string, optional
            Specify the mesh file type, eg. exo, ugrid, shp etc
        kwargs : dict, optional
            A dict initializing specific variables as stated above
            - example: kwargs = {"concave" : True, "islatlon" : True"}

        Raises
        ------
            RuntimeError: File not found
        """
        # TODO: fix when adding/exercising gridspec
        # pull in optional arguments
        self.filepath = filepath
        self.gridspec = gridspec
        self.vertices = vertices
        self.islatlon = islatlon
        self.concave = concave
        self.mesh_filetype = mesh_filetype

        # internal uxarray representation of mesh stored in internal object in_ds
        self.in_ds = xr.Dataset()

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
                raise RuntimeError(
                    "Init called with args other than verts (ndarray) or filename (str)"
                )
        except TypeError as e:
            raise RuntimeError("sting or np.ndarray supported as input")

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
        self.in_ds["Mesh2"] = xr.DataArray(
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Topology data of unstructured mesh",
                "topology_dimension": -1,
                "node_coordinates": "Mesh2_node_x Mesh2_node_y Mesh2_node_z",
                "node_dimension": "nMesh2_node",
                "face_node_connectivity": "Mesh2_face_nodes",
                "face_dimension": "nMesh2_face"
            })
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

    # load mesh from a file
    def __from_file__(self):
        """Loads a mesh file Also, called by __init__ routine This routine will
        automatically detect if it is a UGrid, SCRIP, Exodus, or shape file.

        Raises:
            RuntimeError: Unknown file format
        """
        # call function to set mesh file type: self.mesh_filetype
        self.mesh_filetype = determine_file_type(self.filepath)

        # call reader as per mesh_filetype
        if self.mesh_filetype == "exo":
            self.in_ds = read_exodus(self.filepath)
        elif self.mesh_filetype == "scrip":
            self.in_ds = read_scrip(self.filepath)
        elif self.mesh_filetype == "ugrid":
            self.in_ds = read_ugrid(self.filepath)
        elif self.mesh_filetype == "shp":
            self.in_ds = read_shpfile(self.filepath)
        else:
            raise RuntimeError("unknown file format:" + self.mesh_filetype)

    # renames the grid file
    def saveas_file(self, filepath):
        """Saves the loaded mesh file as the file with desired type. Internally
        calls the write function.

        Parameters
        ----------
        filename : string, required
        """

        path = PurePath(filepath)
        if not os.path.isdir(path.parent):
            raise FileNotFoundError("Check file path:" + filepath)
        # call write
        self.write(filepath)
        # set filepath to this new file
        self.filepath = str(filepath)

    def write(self, outfile, format=""):
        """Writes mesh file as per extension supplied in the outfile string.

        Parameters
        ----------
        output filename : string, required
        format : extension, optional
            Automatically detected if nothing is specified, defaults to ""
        """
        if format == "":
            outfile_path = PurePath(outfile)
            format = outfile_path.suffix
            if not os.path.isdir(outfile_path.parent):
                raise ("File directory not found: " + outfile)

        if format == ".ugrid" or format == ".ug":
            write_ugrid(self.in_ds, outfile)
        elif format == ".g" or format == ".exo":
            write_exodus(self.in_ds, outfile)
        else:
            print("Format not supported for writing. ", format)

    # Calculate the area of all faces.
    def calculate_total_face_area(self):
        """Not implemented."""
        warn("Function placeholder, implementation coming soon.")

    # Build the node-face connectivity array.
    def build_node_face_connectivity(self):
        """Not implemented."""
        warn("Function placeholder, implementation coming soon.")

    # Build the edge-face connectivity array.
    def build_edge_face_connectivity(self):
        """Not implemented."""
        warn("Function placeholder, implementation coming soon.")

    # Build the array of latitude-longitude bounding boxes.
    def buildlatlon_bounds(self):
        """Not implemented."""
        warn("Function placeholder, implementation coming soon.")

    # Validate that the grid conforms to the UXGrid standards.
    def validate(self):
        """Not implemented."""
        warn("Function placeholder, implementation coming soon.")
