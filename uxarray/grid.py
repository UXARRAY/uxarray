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
from .helpers import get_all_face_area_from_coords, convert_node_lonlat_rad_to_xyz, convert_node_xyz_to_lonlat_rad



class Grid:
    """The Uxarray Grid object class that describes an unstructured grid.

    Examples
    ----------

    Open an exodus file with Uxarray Grid object

    >>> mesh = ux.Grid("filename.g")

    Save as ugrid file

    >>> mesh.write("outfile.ug")
    """

    def __init__(self, dataset, **kwargs):
        """Initialize grid variables, decide if loading happens via file, verts
        or gridspec.

        Parameters
        ----------
        dataset : xarray.Dataset, ndarray, list, tuple, required
            Input xarray.Dataset or vertex coordinates that form one face.

        Other Parameters
        ----------------
        islatlon : bool, optional
            Specify if the grid is lat/lon based
        concave: bool, optional
            Specify if this grid has concave elements (internal checks for this are possible)
        gridspec: bool, optional
            Specifies gridspec
        mesh_filetype: str, optional
            Specify the mesh file type, eg. exo, ugrid, shp, etc

        Raises
        ------
            RuntimeError
                If specified file not found
        """
        # initialize internal variable names
        self.__init_ds_var_names__()

        # initialize face_area variable
        self._face_areas = None

        # TODO: fix when adding/exercising gridspec

        # unpack kwargs
        # sets default values for all kwargs to None
        kwargs_list = [
            'gridspec', 'vertices', 'islatlon', 'concave', 'mesh_filetype',
            'source_grid', 'source_datasets'
        ]
        for key in kwargs_list:
            setattr(self, key, kwargs.get(key, None))

        # internal uxarray representation of mesh stored in internal object ds
        self.ds = xr.Dataset()

        # check if initializing from verts:
        if isinstance(dataset, (list, tuple, np.ndarray)):
            self.vertices = dataset
            self.__from_vert__()
            self.source_grid = "From vertices"
            self.source_datasets = None

        # check if initializing from string
        # TODO: re-add gridspec initialization when implemented
        elif isinstance(dataset, xr.Dataset):
            self.__from_ds__(dataset=dataset)
        else:
            raise RuntimeError(f"{dataset} is not a valid input type.")

        # initialize convenience attributes
        self.__init_grid_var_attrs__()

    def __from_vert__(self):
        """Create a grid with one face with vertices specified by the given
        argument.

        Called by :func:`__init__`.
        """
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
        connectivity = [list(range(0, num_nodes))]

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
            data=xr.DataArray(connectivity),
            dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
            attrs={
                "cf_role": "face_node_connectivity",
                "_FillValue": -1,
                "start_index": 0
            })

    # load mesh from a file
    def __from_ds__(self, dataset):
        """Loads a mesh dataset."""
        # call reader as per mesh_filetype
        if self.mesh_filetype == "exo":
            self.ds = _read_exodus(dataset, self.ds_var_names)
        elif self.mesh_filetype == "scrip":
            self.ds = _read_scrip(dataset)
        elif self.mesh_filetype == "ugrid":
            self.ds, self.ds_var_names = _read_ugrid(dataset, self.ds_var_names)
        elif self.mesh_filetype == "shp":
            self.ds = _read_shpfile(self.filepath)
        else:
            raise RuntimeError("unknown file format: " + self.mesh_filetype)
        dataset.close()

    def write(self, outfile, extension=""):
        """Writes mesh file as per extension supplied in the outfile string.

        Parameters
        ----------
        outfile : str, required
            Path to output file
        extension : str, optional
            Extension of output file. Defaults to empty string.
            Currently supported options are ".ugrid", ".ug", ".g", ".exo", and ""

        Raises
        ------
        RuntimeError
            If unsupported extension provided or directory not found
        """
        if extension == "":
            outfile_path = PurePath(outfile)
            extension = outfile_path.suffix
            if not os.path.isdir(outfile_path.parent):
                raise RuntimeError("File directory not found: " + outfile)

        if extension == ".ugrid" or extension == ".ug":
            _write_ugrid(self.ds, outfile, self.ds_var_names)
        elif extension == ".g" or extension == ".exo":
            _write_exodus(self.ds, outfile, self.ds_var_names)
        else:
            raise RuntimeError("Format not supported for writing: ", extension)

    def calculate_total_face_area(self, quadrature_rule="triangular", order=4):
        """Function to calculate the total surface area of all the faces in a
        mesh.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        Sum of area of all the faces in the mesh : float
        """

        # call function to get area of all the faces as a np array
        face_areas = self.compute_face_areas(quadrature_rule, order)

        return np.sum(face_areas)

    # Build the node-face connectivity array.
    def build_node_face_connectivity(self):
        """Not implemented."""
        warn("Function placeholder, implementation coming soon.")

    # Build the edge-face connectivity array.
    def build_edge_face_connectivity(self):
        """Not implemented."""
        mesh2_edge_nodes_set = set(
        )  # Use the set data structure to store Edge object (undirected)

        # Also generate the face_edge_connectivity:Mesh2_face_edges for the latlonbox building
        mesh2_face_edges = []

        mesh2_face_nodes = self.ds["Mesh2_face_nodes"].values

        # Loop over each face
        for face in mesh2_face_nodes:
            cur_face_edge = []
            # Loop over nodes in a face
            for i in range(0, face.size - 1):
                # with _FillValue=-1 used when faces have fewer nodes than MaxNumNodesPerFace.
                if face[i] == -1 or face[i + 1] == -1:
                    continue
                # Two nodes are connected to one another if they’re adjacent in the array
                mesh2_edge_nodes_set.add(Edge([face[i], face[i + 1]]))
                cur_face_edge.append([face[i], face[i + 1]])
            # Two nodes are connected if one is the first element of the array and the other is the last

            # First make sure to skip the dummy _FillValue=-1 node
            last_node = face.size - 1
            start_node = 0
            while face[last_node] == -1 and last_node > 0:
                last_node -= 1
            while face[start_node] == -1 and start_node > 0:
                start_node += 1
            if face[last_node] < 0 or face[last_node] < 0:
                raise Exception('Invalid node index')
            mesh2_edge_nodes_set.add(Edge([face[last_node], face[start_node]]))
            cur_face_edge.append([face[last_node], face[start_node]])
            mesh2_face_edges.append(cur_face_edge)

        # Convert the Edge object set into list
        mesh2_edge_nodes = []
        for edge in mesh2_edge_nodes_set:
            mesh2_edge_nodes.append(edge.get_nodes())

        self.ds["Mesh2_edge_nodes"] = xr.DataArray(data=mesh2_edge_nodes,
                                                   dims=["nMesh2_edge", "Two"])

        for i in range(0, len(mesh2_face_edges)):
            while len(mesh2_face_edges[i]) < len(mesh2_face_nodes[0]):
                # Append dummy edges
                mesh2_face_edges[i].append([-1, -1])

        self.ds["Mesh2_face_edges"] = xr.DataArray(
            data=mesh2_face_edges,
            dims=["nMesh2_face", "nMaxMesh2_face_edges", "Two"])

    # Build the array of latitude-longitude bounding boxes.
    def buildlatlon_bounds(self):
        """Not implemented."""
        warn("Function placeholder, implementation coming soon.")

    # Validate that the grid conforms to the UXGrid standards.
    def validate(self):
        """Not implemented."""
        warn("Function placeholder, implementation coming soon.")

    def __init_ds_var_names__(self):
        """Populates a dictionary for storing uxarray's internal representation
        of xarray object.

        Note ugrid conventions are flexible with names of variables, see:
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

    def integrate(self, var_key, quadrature_rule="triangular", order=4):
        """Integrates over all the faces of the given mesh.

        Parameters
        ----------
        var_key : str, required
            Name of dataset variable for integration
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        Calculated integral : float

        Examples
        --------
        Open grid file only

        >>> grid = ux.open_dataset("grid.ug", "centroid_pressure_data_ug")

        Open grid file along with data

        >>> integral_psi = grid.integrate("psi")
        """
        integral = 0.0

        # call function to get area of all the faces as a np array
        face_areas = self.compute_face_areas(quadrature_rule, order)

        face_vals = self.ds.get(var_key).to_numpy()
        integral = np.dot(face_areas, face_vals)

        return integral

    def compute_face_areas(self, quadrature_rule="triangular", order=4):
        """Face areas calculation function for grid class, calculates area of
        all faces in the grid.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        Area of all the faces in the mesh : np.ndarray

        Examples
        --------
        Open a uxarray grid file

        >>> grid = ux.open_dataset("/home/jain/uxarray/test/meshfiles/outCSne30.ug")

        Get area of all faces in the same order as listed in grid.ds.Mesh2_face_nodes

        >>> grid.get_face_areas
        array([0.00211174, 0.00211221, 0.00210723, ..., 0.00210723, 0.00211221,
            0.00211174])
        """
        if self._face_areas is None:
            # area of a face call needs the units for coordinate conversion if spherical grid is used
            coords_type = "spherical"
            if not "degree" in self.Mesh2_node_x.units:
                coords_type = "cartesian"

            face_nodes = self.Mesh2_face_nodes.data
            dim = self.Mesh2.attrs['topology_dimension']

            # initialize z
            z = np.zeros((self.ds.nMesh2_node.size))

            # call func to cal face area of all nodes
            x = self.Mesh2_node_x.data
            y = self.Mesh2_node_y.data
            # check if z dimension
            if self.Mesh2.topology_dimension > 2:
                z = self.Mesh2_node_z.data

            # call function to get area of all the faces as a np array
            self._face_areas = get_all_face_area_from_coords(
                x, y, z, face_nodes, dim, quadrature_rule, order, coords_type)

        return self._face_areas

    # use the property keyword for declaration on face_areas property
    @property
    def face_areas(self):
        """Declare face_areas as a property."""

        if self._face_areas is None:
            self.compute_face_areas()
        return self._face_areas

    def __init_grid_var_attrs__(self):
        """Initialize attributes for directly accessing Coordinate and Data
        variables through ugrid conventions.

        Examples
        ----------
        Assuming the mesh node coordinates for longitude are stored with an input
        name of 'mesh_node_x', we store this variable name in the `ds_var_names`
        dictionary with the key 'Mesh2_node_x'. In order to access it:

        >>> x = grid.ds[grid.ds_var_names["Mesh2_node_x"]]

        With the help of this function, we can directly access it through the
        use of a standardized name (ugrid convention)

        >>> x = grid.Mesh2_node_x
        """

        # Set UGRID standardized attributes
        for key, value in self.ds_var_names.items():
            # Present Data Names
            if self.ds.data_vars is not None:
                if value in self.ds.data_vars:
                    setattr(self, key, self.ds[value])

            # Present Coordinate Names
            if self.ds.coords is not None:
                if value in self.ds.coords:
                    setattr(self, key, self.ds[value])

    def __populate_cartesian_xyz_coord(self):
        """
        A helper function that populates the xyz attribute in Mesh.ds
        use case: If the grid file's Mesh2_node_x 's unit is in degree
        """
        #   Mesh2_node_x
        #      unit x = "lon" degree
        #   Mesh2_node_y
        #      unit x = "lat" degree
        #   Mesh2_node_z
        #      unit x = "m"
        #   Mesh2_node_cart_x
        #      unit m
        #   Mesh2_node_cart_y
        #      unit m
        #   Mesh2_node_cart_z
        #      unit m

        # Check if the cartesian coordinates are already populated
        if "Mesh2_node_cart_x" in self.ds.keys():
            return

        # check for units and create Mesh2_node_cart_x/y/z set to self.ds
        num_nodes = self.ds.Mesh2_node_x.size
        node_cart_list_x = [0.0] * num_nodes
        node_cart_list_y = [0.0] * num_nodes
        node_cart_list_z = [0.0] * num_nodes
        for i in range(num_nodes):
            if "degree" in self.ds.Mesh2_node_x.units:
                node = [np.deg2rad(self.ds["Mesh2_node_x"].values[i]),
                        np.deg2rad(self.ds["Mesh2_node_y"].values[i])]  # [lon, lat]
                node_cart = convert_node_lonlat_rad_to_xyz(node)  # [x, y, z]
                node_cart_list_x[i] = node_cart[0]
                node_cart_list_y[i] = node_cart[1]
                node_cart_list_z[i] = node_cart[2]

        self.ds["Mesh2_node_cart_x"] = xr.DataArray(data=node_cart_list_x)
        self.ds["Mesh2_node_cart_y"] = xr.DataArray(data=node_cart_list_y)
        self.ds["Mesh2_node_cart_z"] = xr.DataArray(data=node_cart_list_z)

    def __populate_lonlat_coord(self):
        """
         Helper function that populates the longitude and latitude and store it into the Mesh2_node_x and Mesh2_node_y
          use case: If the grid file's Mesh2_node_x 's unit is in meter
        """

        # Check if the "Mesh2_node_x" is already in longitude
        if "degree" in self.ds.Mesh2_node_x.units:
            return
        num_nodes = self.ds.Mesh2_node_x.size
        node_latlon_list_lat = [0.0] * num_nodes
        node_latlon_list_lon = [0.0] * num_nodes
        node_cart_list_x = [0.0] * num_nodes
        node_cart_list_y = [0.0] * num_nodes
        node_cart_list_z = [0.0] * num_nodes
        for i in range(num_nodes):
            if "m" in self.ds.Mesh2_node_x.units:
                node = [self.ds["Mesh2_node_x"][i],
                        self.ds["Mesh2_node_y"][i],
                        self.ds["Mesh2_node_z"][i]]  # [x, y, z]
                node_lonlat = convert_node_xyz_to_lonlat_rad(node)  # [lon, lat]
                node_cart_list_x[i] = self.ds["Mesh2_node_x"].values[i]
                node_cart_list_y[i] = self.ds["Mesh2_node_y"].values[i]
                node_cart_list_z[i] = self.ds["Mesh2_node_z"].values[i]
                node_lonlat[0] = np.rad2deg(node_lonlat[0])
                node_lonlat[1] = np.rad2deg(node_lonlat[1])
                node_latlon_list_lon[i] = node_lonlat[0]
                node_latlon_list_lat[i] = node_lonlat[1]

        self.ds["Mesh2_node_cart_x"] = xr.DataArray(data=node_cart_list_x)
        self.ds["Mesh2_node_cart_y"] = xr.DataArray(data=node_cart_list_y)

        self.ds["Mesh2_node_x"].values = node_latlon_list_lon
        self.ds["Mesh2_node_y"].values = node_latlon_list_lat
        self.ds.Mesh2_node_x.units = "degree_east"