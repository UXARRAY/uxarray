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
from .helpers import determine_file_type, calculate_face_area


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

        # if type == "spherical":

        # set default coordinate units to spherical coordinates
        # users can change to cartesian if using cartesian for initialization
        x_units = "degees_east"
        y_units = "degrees_north"
        if self.vertices[0].size > 2:
            z_units = "elevation"
        # elif type == "cartesian":
        # x_units = "cartesian unit"
        # y_units = "cartesian unit"
        # if self.vertices[0].size > 2:
        #     z_units = "cartesian unit"

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
        mesh."""
        total_face_area = 0

        # get the coordinate units
        units = "spherical"
        if not "degree" in self.ds.Mesh2_node_x.units:
            units = "cartesian"

        for i in range(self.ds.nMesh2_face.size):
            x = []
            y = []
            z = []

            face_node_var = self.ds_var_names["Mesh2_face_nodes"]
            node_x_var = self.ds_var_names["Mesh2_node_x"]
            node_y_var = self.ds_var_names["Mesh2_node_y"]

            for j in range(len(self.ds[face_node_var][i])):
                node_id = self.ds[face_node_var].data[i][j]
                x.append(self.ds[node_x_var].data[node_id])
                y.append(self.ds[node_y_var].data[node_id])
                if self.ds.Mesh2.topology_dimension > 2:
                    node_z_var = self.ds_var_names["Mesh2_node_z"]
                    z.append(self.ds[node_z_var].data[node_id])
                else:
                    z.append(0)

            total_face_area += calculate_face_area(x, y, z, units)

        return total_face_area

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

        # area of a face call needs the units for coordinate conversion if spherical grid is used
        units = "spherical"
        if not "degree" in self.ds.Mesh2_node_x.units:
            units = "cartesian"

        num_faces = self.ds.get(var_key).data.size
        for i in range(num_faces):
            x = []
            y = []
            z = []
            for j in range(len(self.ds.Mesh2_face_nodes[i])):
                node_id = self.ds.Mesh2_face_nodes.data[i][j]

                x.append(
                    self.ds[self.ds_var_names["Mesh2_node_x"]].data[node_id])
                y.append(
                    self.ds[self.ds_var_names["Mesh2_node_y"]].data[node_id])
                if self.ds.Mesh2.topology_dimension > 2:
                    z.append(self.ds[
                        self.ds_var_names["Mesh2_node_z"]].data[node_id])
                else:
                    z.append(0)

            # After getting all the nodes of a face assembled call the  cal. face area routine
            face_area = calculate_face_area(x, y, z, units)
            # get the value from the data file
            face_val = self.ds.get(var_key).to_numpy().data[i]

            integral += face_area * face_val

        # print("Integral of ", var_key, " over the surface is ", integral)
        return integral
