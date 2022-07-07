"""uxarray grid module."""
import os
import xarray as xr
import numpy as np
from warnings import warn
from pathlib import PurePath

# reader and writer imports
from ._exodus import _read_exodus, _write_exodus
from ._ugrid import _read_ugrid, _write_ugrid
from ._shapefile import _read_shpfile
from ._scrip import _read_scrip
from .helpers import determine_file_type
from .helpers import get_all_face_area


class Grid:
    """The Uxarray Grid object class that describes an unstructured grid.

    Examples
    ----------

    Open an exodus file with Uxarray Grid object

    >>> mesh = ux.Grid("filename.g")

    Save as ugrid file

    >>> mesh.write("outfile.ug")
    """

    def __init__(self, *args, **kwargs):
        """Initialize grid variables, decide if loading happens via file, verts
        or gridspec If loading from file, initialization happens via the
        specified file.

        # TODO: Add or remove new Args/kwargs below as this develops further

        Parameters
        ----------

        data_arg : string, ndarray, list, tuple, required
            - Input file name with extension or
            - Vertex coordinates that form one face.

        Other Parameters
        ----------------

        islatlon : bool, optional
            Specify if the grid is lat/lon based:
        concave: bool, optional
            Specify if this grid has concave elements (internal checks for this are possible)
        gridspec: bool, optional
            Specifies gridspec
        mesh_filetype: string, optional
            Specify the mesh file type, eg. exo, ugrid, shp etc

        Raises
        ------

            RuntimeError: File not found
        """
        # initialize internal variable names
        self.__init_ds_var_names__()

        # initialize face_area variable
        self._face_areas = None

        # TODO: fix when adding/exercising gridspec
        # unpacking args
        data_arg = args[0]

        # unpack kwargs
        # sets default values for all kwargs to None
        kwargs_list = [
            'gridspec', 'vertices', 'islatlon', 'concave', 'mesh_filetype'
        ]
        for key in kwargs_list:
            setattr(self, key, kwargs.get(key, None))

        # internal uxarray representation of mesh stored in internal object ds
        self.ds = xr.Dataset()

        # check if initializing from verts:
        if isinstance(data_arg, (list, tuple, np.ndarray)):
            self.vertices = data_arg
            self.__from_vert__()

        # check if initializing from string
        # TODO: re-add gridspec initialization when implemented
        elif isinstance(data_arg, str):
            # check if file exists
            if not os.path.isfile(data_arg):
                raise RuntimeError("File not found: " + data_arg)

            self.filepath = data_arg
            # call the appropriate reader
            self.__from_file__()

        # check if invalid initialization
        else:
            raise RuntimeError(data_arg + " is not a valid input type")

    # vertices init
    def __from_vert__(self):
        """Create a grid with one face with vertices specified by the given
        argument."""
        self.ds["Mesh2"] = xr.DataArray(
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Topology data of unstructured mesh",
                "topology_dimension": -1,
                "node_coordinates": "Mesh2_node_x Mesh2_node_y Mesh2_node_z",
                "node_dimension": "nMesh2_node",
                "face_node_connectivity": "Mesh2_face_nodes",
                "face_dimension": "nMesh2_face"
            })
        self.ds.Mesh2.attrs['topology_dimension'] = self.vertices[0].size

        # set default coordinate units to spherical coordinates
        # users can change to cartesian if using cartesian for initialization
        x_units = "degrees_east"
        y_units = "degrees_north"
        if self.vertices[0].size > 2:
            z_units = "elevation"

        x_coord = self.vertices.transpose()[0]
        y_coord = self.vertices.transpose()[1]
        if self.vertices[0].size > 2:
            z_coord = self.vertices.transpose()[2]

        # single face with all nodes
        num_nodes = x_coord.size
        conn = list(range(0, num_nodes))
        conn = [conn]

        self.ds["Mesh2_node_x"] = xr.DataArray(data=xr.DataArray(x_coord),
                                               dims=["nMesh2_node"],
                                               attrs={"units": x_units})
        self.ds["Mesh2_node_y"] = xr.DataArray(data=xr.DataArray(y_coord),
                                               dims=["nMesh2_node"],
                                               attrs={"units": y_units})
        if self.vertices[0].size > 2:
            self.ds["Mesh2_node_z"] = xr.DataArray(data=xr.DataArray(z_coord),
                                                   dims=["nMesh2_node"],
                                                   attrs={"units": z_units})

        self.ds["Mesh2_face_nodes"] = xr.DataArray(
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
            self.ds = _read_exodus(self.filepath, self.ds_var_names)
        elif self.mesh_filetype == "scrip":
            self.ds = _read_scrip(self.filepath)
        elif self.mesh_filetype == "ugrid":
            self.ds, self.ds_var_names = _read_ugrid(self.filepath,
                                                     self.ds_var_names)
        elif self.mesh_filetype == "shp":
            self.ds = _read_shpfile(self.filepath)
        else:
            raise RuntimeError("unknown file format: " + self.mesh_filetype)

    def write(self, outfile, extension=""):
        """Writes mesh file as per extension supplied in the outfile string.

        Parameters
        ----------

        outfile : string, required
        extension : file extension, optional
            Defaults to ""
        """
        if extension == "":
            outfile_path = PurePath(outfile)
            extension = outfile_path.suffix
            if not os.path.isdir(outfile_path.parent):
                raise ("File directory not found: " + outfile)

        if extension == ".ugrid" or extension == ".ug":
            _write_ugrid(self.ds, outfile, self.ds_var_names)
        elif extension == ".g" or extension == ".exo":
            _write_exodus(self.ds, outfile, self.ds_var_names)
        else:
            print("Format not supported for writing: ", extension)

    def calculate_total_face_area(self):
        """Function to calculate the total surface area of all the faces in a
        mesh.

        Returns
        -------

        float: Sum of area of all the faces in the mesh
        """

        if self._face_areas is None:
            self.calculate_each_face_area()

        return np.sum(self._face_areas)

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

    def __init_ds_var_names__(self):
        """A dictionary for storing uxarray's internal representation of xarray
        object.

        ugrid conventions are flexible with names of variables, this dict stores the conversion
        http://ugrid-conventions.github.io/ugrid-conventions/
        """
        self.ds_var_names = {
            "Mesh2": "Mesh2",
            "Mesh2_node_x": "Mesh2_node_x",
            "Mesh2_node_y": "Mesh2_node_y",
            "Mesh2_node_z": "Mesh2_node_z",
            "Mesh2_face_nodes": "Mesh2_face_nodes",
            # initialize dims
            "nMesh2_node": "nMesh2_node",
            "nMesh2_face": "nMesh2_face",
            "nMaxMesh2_face_nodes": "nMaxMesh2_face_nodes"
        }

    def integrate(self, var_key):
        """ Integrates over all the faces of the given mesh.
        Parameters
        ----------

        var_key : string, required
            Name of variable for integration.

        Returns
        -------

        double: integration result.

        Examples
        --------

        Open grid file only
        >>> grid = ux.open_dataset("grid.ug", "centroid_pressure_data_ug")


        Open grid file along with data
        >>> integral_psi = grid.integrate("psi")
        """
        integral = 0.0

        if self._face_areas is None:
            self.calculate_each_face_area()

        face_vals = self.ds.get(var_key).to_numpy()
        integral = np.dot(self._face_areas, face_vals)

        return integral

    def calculate_each_face_area(self):
        """Face area calculation property for grid class, calculates area of
        all faces in the mesh.

        Returns
        -------

        area of all the faces in the mesh. : ndarray

        Examples
        --------

        Open a uxarray grid file

        >>> grid = ux.open_dataset("/home/jain/uxarray/test/meshfiles/outCSne30.ug")

        Get area of all faces in the same order as listed in grid.ds.Mesh2_face_nodes

        >>> grid.calculate_each_face_area
        array([0.00211174, 0.00211221, 0.00210723, ..., 0.00210723, 0.00211221,
            0.00211174])
        """
        if self._face_areas is None:
            # area of a face call needs the units for coordinate conversion if spherical grid is used
            coords_type = "spherical"
            if not "degree" in self.ds.Mesh2_node_x.units:
                coords_type = "cartesian"

            face_nodes = self.ds.Mesh2_face_nodes.data
            dim = self.ds.Mesh2.attrs['topology_dimension']

            # initialize z
            z = np.zeros((self.ds.nMesh2_node.size))

            # call func to cal face area of all nodes
            x = self.ds[self.ds_var_names["Mesh2_node_x"]].data
            y = self.ds[self.ds_var_names["Mesh2_node_y"]].data
            # check if z dimension
            if self.ds.Mesh2.topology_dimension > 2:
                z = self.ds[self.ds_var_names["Mesh2_node_z"]].data

            # call function to get area of all the faces as a np array
            self._face_areas = get_all_face_area(x, y, z, face_nodes, dim,
                                                 coords_type)

        return self._face_areas
