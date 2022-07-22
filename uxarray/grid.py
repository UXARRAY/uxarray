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
from .helpers import parse_grid_type, insert_pt_in_latlonbox, Edge,  \
    get_intersection_point, convert_node_latlon_rad_to_xyz, dot_product,  normalize_in_place


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
        or gridspec If loading from file, initialization happens via the
        specified file.

        # TODO: Add or remove new Args/kwargs below as this develops further

        Parameters
        ----------

        dataset : xarray.Dataset, ndarray, list, tuple, required
            - Input xarray.Dataset or
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
        self.Mesh2_face_edges = None
        self.Mesh2_latlon_bounds = None
        self.Mesh2_edge_nodes = None
        self.__init_ds_var_names__()

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

        x_coord = self.vertices.transpose()[0]
        y_coord = self.vertices.transpose()[1]

        # single face with all nodes
        num_nodes = x_coord.size
        conn = list(range(0, num_nodes))
        conn = [conn]

        self.ds["Mesh2_node_x"] = xr.DataArray(data=xr.DataArray(x_coord),
                                               dims=["nMesh2_node"])
        self.ds["Mesh2_node_y"] = xr.DataArray(data=xr.DataArray(y_coord),
                                               dims=["nMesh2_node"])
        self.ds["Mesh2_face_nodes"] = xr.DataArray(
            data=xr.DataArray(conn),
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
        mesh2_edge_nodes_set = set()  # Use the set data structure to store Edge object (undirected)

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
            mesh2_edge_nodes_set.add(Edge([face[0], face[face.size - 1]]))
            cur_face_edge.append([face[0], face[face.size - 1]])
            mesh2_face_edges.append(cur_face_edge)

        # Convert the Edge object set into list
        mesh2_edge_nodes = []
        for edge in mesh2_edge_nodes_set:
            mesh2_edge_nodes.append(edge.get_nodes())

        self.Mesh2_edge_nodes = xr.DataArray(
            data=mesh2_edge_nodes,
            dims=["nMesh2_edge", "Two"]
        )

        self.Mesh2_face_edges = xr.DataArray(
            data=mesh2_face_edges,
            dims=["nMesh2_face", "nMaxMesh2_face_edges", "Two"]
        )

        warn("Function placeholder, implementation coming soon.")

    # Build the array of latitude-longitude bounding boxes.
    def buildlatlon_bounds(self):
        """Not implemented."""

        # First make sure the Grid object has the Mesh2_face_edges

        if self.Mesh2_edge_nodes is None:
            self.build_edge_face_connectivity()

        temp_latlon_array = [[[0.0, 0.0], [0.0, 0.0]]] * self.Mesh2_face_edges.sizes["nMesh2_face"]

        reference_tolerance = 1.0e-12

        for i in range(0, len(self.Mesh2_face_edges)):
            face = self.Mesh2_face_edges[i]

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
                    sum_lon += key
                    cnt += 1
                    if cnt >= 2:
                        break

                v2 = [np.cos(sum_lon / 2), np.sin(sum_lon / 2), 0]
                num_intersects = self.__count_face_edge_intersection(face, [v1, v2])

            if num_intersects % 2 != 0:
                # if the face contains the pole point
                for j in range(0, len(face)):
                    edge = face[j]
                    # All the following calculation is based on the 3D XYZ coord
                    # And assume the self.ds["Mesh2_node_x"] always store the lon info
                    n1 = []
                    n2 = []

                    # Convert the 2D [lon, lat] to 3D [x, y, z]
                    n1 = convert_node_latlon_rad_to_xyz([self.ds["Mesh2_node_x"].values[edge[0]],
                                                         self.ds["Mesh2_node_y"].values[edge[0]]])
                    n2 = convert_node_latlon_rad_to_xyz([self.ds["Mesh2_node_x"].values[edge[1]],
                                                         self.ds["Mesh2_node_y"].values[edge[1]]])

                    # Set the latitude extent
                    d_lat_extent_rad = 0
                    if j == 0:
                        if n1[2] < 0.0:
                            d_lat_extent_rad = -0.5 * np.pi
                        else:
                            d_lat_extent_rad = 0.5 * np.pi

                    # insert edge endpoint into box
                    # TODO: Make sure ds["Mesh2_node_x"] and ds["Mesh2_node_y"] always store the lon/lat value
                    if np.absolute(self.ds["Mesh2_node_y"].values[edge[0]]) < d_lat_extent_rad:
                        d_lat_extent_rad = self.ds["Mesh2_node_y"].values[edge[0]]

                    # TODO: Consider about the constant latitude edge type.
                    # Determine if latitude is maximized between endpoints
                    dot_n1_n2 = dot_product(n1, n2)
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
                            d_lat_rad + -0.5 * np.pi
                        else:
                            d_lat_rad = np.arcsin(d_lat_rad)

                        if np.absolute(d_lat_rad) < np.absolute(d_lat_extent_rad):
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

                    # Convert the 2D [lon, lat] to 3D [x, y, z]
                    n1 = convert_node_latlon_rad_to_xyz([self.ds["Mesh2_node_x"].values[edge[0]],
                                                         self.ds["Mesh2_node_y"].values[edge[0]]])
                    n2 = convert_node_latlon_rad_to_xyz([self.ds["Mesh2_node_x"].values[edge[1]],
                                                         self.ds["Mesh2_node_y"].values[edge[1]]])

                    # Determine if latitude is maximized between endpoints
                    dot_n1_n2 = dot_product(n1, n2)
                    d_de_nom = (n1[2] + n2[2]) * (dot_n1_n2 - 1.0)

                    # insert edge endpoint into box
                    # TODO: Make sure ds["Mesh2_node_x"] and ds["Mesh2_node_y"] always store the lon/lat value
                    d_lat_rad = np.deg2rad(self.ds["Mesh2_node_y"].values[edge[0]])
                    d_lon_rad = np.deg2rad(self.ds["Mesh2_node_x"].values[edge[0]])
                    temp_latlon_array[i] = insert_pt_in_latlonbox(copy.deepcopy(temp_latlon_array[i]),
                                                                  [d_lat_rad, d_lon_rad])

                    cur_latlon_box_0 = insert_pt_in_latlonbox(copy.deepcopy(temp_latlon_array[i]),
                                                              [d_lat_rad, d_lon_rad])
                    cur_latlon_box_1 = copy.deepcopy(cur_latlon_box_0)

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

                        temp_latlon_array[i] = insert_pt_in_latlonbox(copy.deepcopy(temp_latlon_array[i]),
                                                                      [d_lat_rad, d_lon_rad])
                        cur_latlon_box_1 = insert_pt_in_latlonbox(copy.deepcopy(temp_latlon_array[i]),
                                                                  [d_lat_rad, d_lon_rad])

                    debug_flag = 2

            assert temp_latlon_array[i][0][0] != temp_latlon_array[i][0][1]
            assert temp_latlon_array[i][1][0] != temp_latlon_array[i][1][1]

        self.Mesh2_latlon_bounds = xr.DataArray(
            data=temp_latlon_array,
            dims=["nMesh2_face", "Latlon", "Two"]
        )

        warn("Function placeholder, implementation coming soon.")

    # Helper function to sort edges of a face based on lowest longitude
    def __sort_edges(self, face):
        """Helper function to sort a list of edges in ascending order based on their longnitude

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
            n1 = [self.ds["Mesh2_node_x"].values[edge[0]], self.ds["Mesh2_node_y"].values[edge[0]]]
            n2 = [self.ds["Mesh2_node_x"].values[edge[1]], self.ds["Mesh2_node_y"].values[edge[1]]]
            edge_dict[min(n1[0], n2[0])] = edge
        edge_dict = sorted(edge_dict.items())

        return edge_dict

    # Count the number of total intersections of an edge and face (Algo. 2.4 Determining if a grid cell contains a
    # given point)
    def __count_face_edge_intersection(self, face, ref_edge):
        """Helper function to count the total number of intersections points between the reference edge and a face

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
            w1 = convert_node_latlon_rad_to_xyz([self.ds["Mesh2_node_x"].values[edge[0]],
                                                 self.ds["Mesh2_node_y"].values[edge[0]]])
            w2 = convert_node_latlon_rad_to_xyz([self.ds["Mesh2_node_x"].values[edge[1]],
                                                 self.ds["Mesh2_node_y"].values[edge[1]]])

            res = get_intersection_point(w1, w2, v1, v2)

            # two vectors are intersected within range and not parralel
            if (res != [0, 0, 0]) and (res != [-1, -1, -1]):
                num_intersection += 1
            elif res[0] * res[1] * res[2] == 0:
                # if two vectors are parallel
                return -1

        return num_intersection

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

        # Set UGRID standardized attribtues
        for key, value in self.ds_var_names.items():
            # Present Data Names
            if self.ds.data_vars is not None:
                if value in self.ds.data_vars:
                    setattr(self, key, self.ds[value])

            # Present Coordinate Names
            if self.ds.coords is not None:
                if value in self.ds.coords:
                    setattr(self, key, self.ds[value])
