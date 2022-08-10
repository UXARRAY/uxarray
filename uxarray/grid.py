"""uxarray grid module."""
import copy
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

from .helpers import get_all_face_area_from_coords, parse_grid_type, insert_pt_in_latlonbox, Edge,  \
    get_intersection_point, convert_node_lonlat_rad_to_xyz, _spherical_to_cartesian_unit_, convert_node_xyz_to_lonlat_rad

from .utilities import normalize_in_place


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
            'gridspec', 'vertices', 'islatlon', 'concave', 'mesh_filetype'
        ]
        for key in kwargs_list:
            setattr(self, key, kwargs.get(key, None))

        # internal uxarray representation of mesh stored in internal object ds
        self.ds = xr.Dataset()

        # check if initializing from verts:
        if isinstance(dataset, (list, tuple, np.ndarray)):
            self.vertices = dataset
            self.__from_vert__()

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
                # Two nodes are connected to one another if they’re adjacent in the array
                mesh2_edge_nodes_set.add(Edge([face[i], face[i + 1]]))
                cur_face_edge.append([face[i], face[i + 1]])
            # Two nodes are connected if one is the first element of the array and the other is the last
            mesh2_edge_nodes_set.add(Edge([face[face.size - 1], face[0]]))
            cur_face_edge.append([face[face.size - 1], face[0]])
            mesh2_face_edges.append(cur_face_edge)

        # Convert the Edge object set into list
        mesh2_edge_nodes = []
        for edge in mesh2_edge_nodes_set:
            mesh2_edge_nodes.append(edge.get_nodes())

        self.ds["Mesh2_edge_nodes"] = xr.DataArray(data=mesh2_edge_nodes,
                                                   dims=["nMesh2_edge", "Two"])

        self.ds["Mesh2_face_edges"] = xr.DataArray(
            data=mesh2_face_edges,
            dims=["nMesh2_face", "nMaxMesh2_face_edges", "Two"])

        warn("Function placeholder, implementation coming soon.")

    # Build the array of latitude-longitude bounding boxes.
    def buildlatlon_bounds(self):
        """Not implemented."""

        # First make sure the Grid object has the Mesh2_face_edges

        self.build_edge_face_connectivity()

        # TODO: Change to check if the ds already has the ds["Mesh2_node_cart_x"] coordinates

        if "Mesh2_node_cart_x" not in self.ds.keys():
            self.__populate_cartesian_xyz_coord()

        # All value are inialized as 404.0 to indicate that they're null
        temp_latlon_array = [[[404.0, 404.0], [404.0, 404.0]]
                            ] * self.ds["Mesh2_face_edges"].sizes["nMesh2_face"]


        reference_tolerance = 1.0e-12

        for i in range(0, len(self.ds["Mesh2_face_edges"])):
            face = self.ds["Mesh2_face_edges"][i]

            debug_flag = -1

            # Check if face contains pole points
            _lambda = 0
            v1 = [0, 0, 1]
            v2 = [np.cos(_lambda), np.sin(_lambda), -1]

            num_intersects = self.__count_face_edge_intersection(face, [v1, v2])
            if num_intersects == -1:
                # if one edge of the grid cell is parallel to the arc (v1 , v2 ) then vary the choice of v2 .
                sorted_edges = self.__sort_edges(face)
                # only need to iterate the first two keys to average the longitude:
                cnt = 0
                sum_lon = 0.0
                for key in sorted_edges:
                    sum_lon += key[0]
                    cnt += 1
                    if cnt >= 2:
                        break

                v2 = [np.cos(sum_lon / 2), np.sin(sum_lon / 2), 0]
                num_intersects = self.__count_face_edge_intersection(
                    face, [v1, v2])

            if num_intersects % 2 != 0:
                # if the face contains the pole point
                for j in range(0, len(face)):
                    edge = face[j]
                    # All the following calculation is based on the 3D XYZ coord
                    # And assume the self.ds["Mesh2_node_x"] always store the lon info
                    n1 = []
                    n2 = []

                    # Get the edge end points in 3D [x, y, z] coordinates
                    n1 = [self.ds["Mesh2_node_cart_x"].values[edge[0]],
                          self.ds["Mesh2_node_cart_y"].values[edge[0]],
                          self.ds["Mesh2_node_cart_z"].values[edge[0]]]
                    n2 = [self.ds["Mesh2_node_cart_x"].values[edge[1]],
                          self.ds["Mesh2_node_cart_y"].values[edge[1]],
                          self.ds["Mesh2_node_cart_z"].values[edge[1]]]

                    # Set the latitude extent
                    d_lat_extent_rad = 0.0
                    if j == 0:
                        if n1[2] < 0.0:
                            d_lat_extent_rad = -0.5 * np.pi
                        else:
                            d_lat_extent_rad = 0.5 * np.pi

                    # insert edge endpoint into box
                    if np.absolute(self.ds["Mesh2_node_y"].values[
                            edge[0]]) < d_lat_extent_rad:
                        d_lat_extent_rad = self.ds["Mesh2_node_y"].values[
                            edge[0]]

                    # TODO: Consider about the constant latitude edge type.
                    # Determine if latitude is maximized between endpoints
                    dot_n1_n2 = np.dot(n1, n2)
                    d_de_nom = (n1[2] + n2[2]) * (dot_n1_n2 - 1.0)
                    if np.absolute(d_de_nom) < reference_tolerance:
                        continue

                    d_a_max = (n1[2] * dot_n1_n2 - n2[2]) / d_de_nom
                    if (d_a_max > 0.0) and (d_a_max < 1.0):
                        node3 = [0.0, 0.0, 0.0]
                        node3[0] = n1[0] * (1 - d_a_max) + n2[0] * d_a_max
                        node3[1] = n1[1] * (1 - d_a_max) + n2[1] * d_a_max
                        node3[2] = n1[2] * (1 - d_a_max) + n2[2] * d_a_max
                        node3 = normalize_in_place(node3)

                        d_lat_rad = node3[2]

                        if d_lat_rad > 1.0:
                            d_lat_rad = 0.5 * np.pi
                        elif d_lat_rad < -1.0:
                            d_lat_rad = -0.5 * np.pi
                        else:
                            d_lat_rad = np.arcsin(d_lat_rad)

                        if np.absolute(d_lat_rad) < np.absolute(
                                d_lat_extent_rad):
                            d_lat_extent_rad = d_lat_rad

                    if d_lat_extent_rad < 0.0:
                        lon_list = [0.0, 2.0 * np.pi]
                        lat_list = [-0.5 * np.pi, d_lat_extent_rad]
                    else:
                        lon_list = [0.0, 2.0 * np.pi]
                        lat_list = [d_lat_extent_rad, 0.5 * np.pi]

                    temp_latlon_array[i] = [lat_list, lon_list]
                    debug_flag = 1
            else:
                # normal face
                for j in range(0, len(face)):
                    edge = face[j]

                    # Only consider the great circles arcs
                    # All the following calculation is based on the 3D XYZ coord
                    # And assume the self.ds["Mesh2_node_x"] always store the lon info

                    # Get the edge end points in 3D [x, y, z] coordinates
                    n1 = [self.ds["Mesh2_node_cart_x"].values[edge[0]],
                          self.ds["Mesh2_node_cart_y"].values[edge[0]],
                          self.ds["Mesh2_node_cart_z"].values[edge[0]]]
                    n2 = [self.ds["Mesh2_node_cart_x"].values[edge[1]],
                          self.ds["Mesh2_node_cart_y"].values[edge[1]],
                          self.ds["Mesh2_node_cart_z"].values[edge[1]]]

                    # Determine if latitude is maximized between endpoints
                    dot_n1_n2 = np.dot(n1, n2)
                    d_de_nom = (n1[2] + n2[2]) * (dot_n1_n2 - 1.0)

                    # insert edge endpoint into box
                    d_lat_rad = np.deg2rad(
                        self.ds["Mesh2_node_y"].values[edge[0]])
                    d_lon_rad = np.deg2rad(
                        self.ds["Mesh2_node_x"].values[edge[0]])
                    temp_latlon_array[i] = insert_pt_in_latlonbox(
                        copy.deepcopy(temp_latlon_array[i]),
                        [d_lat_rad, d_lon_rad])

                    if np.absolute(d_de_nom) < reference_tolerance:
                        continue

                    # Maximum latitude occurs between endpoints of edge
                    d_a_max = (n1[2] * dot_n1_n2 - n2[2]) / d_de_nom
                    if 0.0 < d_a_max < 1.0:
                        node3 = [0.0, 0.0, 0.0]
                        node3[0] = n1[0] * (1 - d_a_max) + n2[0] * d_a_max
                        node3[1] = n1[1] * (1 - d_a_max) + n2[1] * d_a_max
                        node3[2] = n1[2] * (1 - d_a_max) + n2[2] * d_a_max
                        node3 = normalize_in_place(node3)

                        d_lat_rad = node3[2]

                        if d_lat_rad > 1.0:
                            d_lat_rad = 0.5 * np.pi
                        elif d_lat_rad < -1.0:
                            d_lat_rad = -0.5 * np.pi
                        else:
                            d_lat_rad = np.arcsin(d_lat_rad)

                        temp_latlon_array[i] = insert_pt_in_latlonbox(
                            copy.deepcopy(temp_latlon_array[i]),
                            [d_lat_rad, d_lon_rad])

            assert temp_latlon_array[i][0][0] != temp_latlon_array[i][0][1]
            assert temp_latlon_array[i][1][0] != temp_latlon_array[i][1][1]

        self.ds["Mesh2_latlon_bounds"] = xr.DataArray(
            data=temp_latlon_array, dims=["nMesh2_face", "Latlon", "Two"])

        warn("Function placeholder, implementation coming soon.")

    # Helper function to sort edges of a face based on lowest longitude
    def __sort_edges(self, face):
        """Helper function to sort a list of edges in ascending order based on
        their longnitude.

        Parameters
        ----------
        edge_list: 2D float array:
        [[lon, lat],
         [lon, lat]
         ...
         [lon, lat]
        ]

        Returns: Python dictionary:
        keys are the minimum longnitude of each edge, in sorted order
        values are the corresponding edge
        {
           lon0: edge0
           lon1: edge1
           lon2: edge2, edge3
        }
        """
        edge_dict = {}
        for edge in face.values:
            n1 = [
                self.ds["Mesh2_node_x"].values[edge[0]],
                self.ds["Mesh2_node_y"].values[edge[0]]
            ]
            n2 = [
                self.ds["Mesh2_node_x"].values[edge[1]],
                self.ds["Mesh2_node_y"].values[edge[1]]
            ]
            edge_dict[min(n1[0], n2[0])] = edge
        edge_dict = sorted(edge_dict.items())

        return edge_dict

    # Count the number of total intersections of an edge and face (Algo. 2.4 Determining if a grid cell contains a
    # given point)
    def __count_face_edge_intersection(self, face, ref_edge):
        """Helper function to count the total number of intersections points
        between the reference edge and a face.

        Parameters
        ----------
        face: xarray.DataArray, list, required
        ref_edge: 2D list, the reference edge that intersect with the face (stored in 3D xyz coordinates) [[x1, y1, z1], [x2, y2, z2]]

        Returns:
        num_intersection: number of intersections
        -1: the ref_edge is parallel to one of the edge of the face and need to vary the ref_edge
        """
        v1 = ref_edge[0]
        v2 = ref_edge[1]
        num_intersection = 0
        for edge in face:
            # All the following calculation is based on the 3D XYZ coord
            w1 = []
            w2 = []

            # Convert the 2D [lon, lat] to 3D [x, y, z]
            w1 = convert_node_lonlat_rad_to_xyz([
                np.deg2rad(self.ds["Mesh2_node_x"].values[edge[0]]),
                np.deg2rad(self.ds["Mesh2_node_y"].values[edge[0]])
            ])
            w2 = convert_node_lonlat_rad_to_xyz([
                np.deg2rad(self.ds["Mesh2_node_x"].values[edge[1]]),
                np.deg2rad(self.ds["Mesh2_node_y"].values[edge[1]])
            ])

            res = get_intersection_point(w1, w2, v1, v2)

            # two vectors are intersected within range and not parralel
            if (res != [0, 0, 0]) and (res != [-1, -1, -1]):
                num_intersection += 1
            elif res[0] == 0 and res[1] == 0 and res[2] == 0:
                # if two vectors are parallel
                return -1

        return num_intersection

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
            "Mesh2":
                "Mesh2",
            "Mesh2_node_x":
                "Mesh2_node_x",
            "Mesh2_node_y":
                "Mesh2_node_y",
            "Mesh2_node_z":
                "Mesh2_node_z",
            "Mesh2_face_nodes":
                "Mesh2_face_nodes",
            # initialize dims
            "nMesh2_node":
                "nMesh2_node",
            "nMesh2_face":
                "nMesh2_face",
            "nMaxMesh2_face_nodes":
                "nMaxMesh2_face_nodes"
                # initialize cart storage
                "Mesh2_node"
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

        # TODO: Update the self.ds.Mesh2_node_x.units to "degree"
        self.ds.Mesh2_node_x.units = "degree_east"

