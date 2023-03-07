"""uxarray grid module."""
import os
import xarray as xr
import numpy as np
import copy
from warnings import warn
from pathlib import PurePath
from intervaltree import Interval, IntervalTree

# reader and writer imports
from ._exodus import _read_exodus, _write_exodus
from ._ugrid import _read_ugrid, _write_ugrid
from ._shapefile import _read_shpfile
from ._scrip import _read_scrip
from .helpers import get_all_face_area_from_coords, convert_node_lonlat_rad_to_xyz, convert_node_xyz_to_lonlat_rad, \
    normalize_in_place, _within, _get_radius_of_latitude_rad, get_intersection_pt
from ._latlonbound_utilities import insert_pt_in_latlonbox, get_intersection_point_gcr_gcr



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
        self._latlonbound_tree = None
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
                mesh2_edge_nodes_set.add(frozenset({face[i], face[i + 1]}))
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
            mesh2_edge_nodes_set.add(frozenset({face[last_node], face[start_node]}))
            cur_face_edge.append([face[last_node], face[start_node]])
            mesh2_face_edges.append(cur_face_edge)

        # Convert the Edge object set into list
        mesh2_edge_nodes = []
        for edge in mesh2_edge_nodes_set:
            mesh2_edge_nodes.append(list(edge))

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

        # First make sure the Grid object has the Mesh2_face_edges

        if "Mesh2_face_edges" not in self.ds.keys():
            self.build_edge_face_connectivity()

        if "Mesh2_node_cart_x" not in self.ds.keys():
            self.__populate_cartesian_xyz_coord()

        # All value are inialized as 404.0 to indicate that they're null
        temp_latlon_array = [[[404.0, 404.0], [404.0, 404.0]]
                             ] * self.ds["Mesh2_face_edges"].sizes["nMesh2_face"]

        reference_tolerance = 1.0e-12

        # Build an Interval tree based on the Latitude interval to store latitude-longitude boundaries
        self._latlonbound_tree = IntervalTree()


        for i in range(0, len(self.ds["Mesh2_face_edges"])):
            face = self.ds["Mesh2_face_edges"][i]
            if i == 4694:
                pass
            # Check if face contains pole points
            _lambda = 0
            v1 = [0, 0, 1]
            v2 = normalize_in_place([np.cos(_lambda), np.sin(_lambda), -1])

            num_intersects = self.__count_face_edge_intersection(face, [v1, v2],i)
            if num_intersects == -1:
                # if one edge of the grid cell is parallel to the arc (v1 , v2 ) then vary the choice of v2 .
                sorted_edges_avg_lon = self.__avg_edges_longitude(face)
                # only need to iterate the first two keys to average the longitude:
                sum_lon = sorted_edges_avg_lon[0] + sorted_edges_avg_lon[1]

                v2 = [np.cos(sum_lon / 2), np.sin(sum_lon / 2), 0]
                num_intersects = self.__count_face_edge_intersection(
                    face, [v1, v2])

            if num_intersects % 2 != 0:
                # if the face contains the pole point
                for j in range(0, len(face)):
                    edge = face[j]
                    # Skip the dummy edges
                    if edge[0] == -1 or edge[1] == -1:
                        continue
                    # All the following calculation is based on the 3D XYZ coord
                    # And assume the self.ds["Mesh2_node_x"] always store the lon info

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
            else:
                # normal face
                for j in range(0, len(face)):
                    edge = face[j]
                    # Skip the dummy edges
                    if edge[0] == -1 or edge[1] == -1:
                        continue

                    # For each edge, we only need to consider the first end point in each loop
                    # Check if the end point is the pole point
                    n1 = [self.ds["Mesh2_node_x"].values[edge[0]],
                          self.ds["Mesh2_node_y"].values[edge[0]]]

                    # North Pole:
                    if (np.absolute(n1[0] - 0) < reference_tolerance and np.absolute(
                            n1[1] - 90) < reference_tolerance) or (
                            np.absolute(n1[0] - 180) < reference_tolerance and np.absolute(
                        n1[1] - 90) < reference_tolerance):
                        # insert edge endpoint into box
                        d_lat_rad = np.deg2rad(
                            self.ds["Mesh2_node_y"].values[edge[0]])
                        d_lon_rad = 404.0
                        temp_latlon_array[i] = insert_pt_in_latlonbox(
                            copy.deepcopy(temp_latlon_array[i]),
                            [d_lat_rad, d_lon_rad])
                        continue

                    # South Pole:
                    if (np.absolute(n1[0] - 0) < reference_tolerance and np.absolute(
                            n1[1] - (-90)) < reference_tolerance) or (
                            np.absolute(n1[0] - 180) < reference_tolerance and np.absolute(
                        n1[1] - (-90)) < reference_tolerance):
                        d_lat_rad = np.deg2rad(
                            self.ds["Mesh2_node_y"].values[edge[0]])
                        d_lon_rad = 404.0
                        temp_latlon_array[i] = insert_pt_in_latlonbox(
                            copy.deepcopy(temp_latlon_array[i]),
                            [d_lat_rad, d_lon_rad])
                        continue

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
                    # TODO: Replace this with the get_gcr_max_lat_rad function
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
            lat_list = temp_latlon_array[i][0]
            lon_list = temp_latlon_array[i][1]
            self._latlonbound_tree[lat_list[0]:lat_list[1]] = i

        self.ds["Mesh2_latlon_bounds"] = xr.DataArray(
            data=temp_latlon_array, dims=["nMesh2_face", "Latlon", "Two"])

        # Helper function to get the average longitude of each edge in sorted order (ascending0

    def __avg_edges_longitude(self, face):
        """Helper function to get the average longitude of each edge in sorted order (ascending0
        Parameters
        ----------
        edge_list: 2D float array:
        [[lon, lat],
         [lon, lat]
         ...
         [lon, lat]
        ]
        Returns: 1D float array, record the average longitude of each edge
        """
        edge_list = []
        for edge in face.values:
            # Skip the dump edges
            if edge[0] == -1 or edge[1] == -1:
                continue
            n1 = [
                self.ds["Mesh2_node_x"].values[edge[0]],
                self.ds["Mesh2_node_y"].values[edge[0]]
            ]
            n2 = [
                self.ds["Mesh2_node_x"].values[edge[1]],
                self.ds["Mesh2_node_y"].values[edge[1]]
            ]

            # Since we only want to sort the Edge based on their longitude,
            # We can utilize the Edge class < operator here by creating the Edge only using the longitude
            edge_list.append((n1[0] + n2[0]) / 2)

        edge_list.sort()

        return edge_list

        # Count the number of total intersections of an edge and face (Algo. 2.4 Determining if a grid cell contains a
        # given point)

    def __count_face_edge_intersection(self, face, ref_edge,i=-1):
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
        intersection_set = set()
        num_intersection = 0
        for edge in face:
            # Skip the dump edges
            if edge[0] == -1 or edge[1] == -1:
                continue
            # All the following calculation is based on the 3D XYZ coord

            # [Test]
            w1_rad = [
                self.ds["Mesh2_node_x"].values[edge[0]],
                self.ds["Mesh2_node_y"].values[edge[0]]
            ]

            w2_rad = [
                self.ds["Mesh2_node_x"].values[edge[1]],
                self.ds["Mesh2_node_y"].values[edge[1]]
            ]
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


            res = get_intersection_point_gcr_gcr(w1, w2, v1, v2,i)

            # two vectors are intersected within range and not parralel
            if (res != [0, 0, 0]) and (res != [-1, -1, -1]):
                intersection_set.add(frozenset(np.round(res,decimals=12).tolist()))
                num_intersection += 1
            elif res[0] == 0 and res[1] == 0 and res[2] == 0:
                # if two vectors are parallel
                return -1

        # If the intersection point number is 1, make sure the gcr is not going through a vertex of the face
        # In this situation, the intersection number will be 0 because the gcr doesn't go across the face technically
        if len(intersection_set) == 1:
            intersection_pt = intersection_set.pop()
            for edge in face:
                if edge[0] == -1 or edge[1] == -1:
                    continue
                w1 = convert_node_lonlat_rad_to_xyz([
                    np.deg2rad(self.ds["Mesh2_node_x"].values[edge[0]]),
                    np.deg2rad(self.ds["Mesh2_node_y"].values[edge[0]])
                ])
                w2 = convert_node_lonlat_rad_to_xyz([
                    np.deg2rad(self.ds["Mesh2_node_x"].values[edge[1]]),
                    np.deg2rad(self.ds["Mesh2_node_y"].values[edge[1]])
                ])

                if list(intersection_pt) == w1 or list(intersection_pt) == w2:
                    return 0


        return len(intersection_set)

    # Get the non-conservative zonal average of the input variable
    def get_nc_zonal_avg(self, var_key, latitude_rad):
        '''
         Algorithm:
            For each face:
                Find the constantLat Arc intersection points with the face:
                How to find:
                    Use the root calculation to get the approximate point location
                    Then based on the approximate results, use the newton-raphson method to

        '''

        face_vals = self.ds.get(var_key).to_numpy()
        if "Mesh2_latlon_bounds" not in self.ds.keys() or "Mesh2_latlon_bounds" is None:
            self.buildlatlon_bounds()

        #  First Get the list of faces that falls into this latitude range
        candidate_faces_index_list = []

        # Search through the interval tree for all the candidates face
        candidate_face_set = self._latlonbound_tree.at(latitude_rad)
        for interval in candidate_face_set:
            candidate_faces_index_list.append(interval.data)
        candidate_faces_weight_list = self._get_zonal_face_weights_at_constlat(candidate_faces_index_list, latitude_rad)
        # Get the candidate face values:
        face_vals = self.ds.get(var_key).to_numpy()
        candidate_faces_vals_list = [0.0] * len(candidate_faces_index_list)
        for i in range(0, len(candidate_faces_index_list)):
            face_index = candidate_faces_index_list[i]
            candidate_faces_vals_list[i] = face_vals[face_index]
        zonal_average = np.dot(candidate_faces_weight_list, candidate_faces_vals_list)
        return zonal_average

    def _get_zonal_face_weights_at_constlat(self, candidate_faces_index_list, latitude_rad):
        # Then calculate the weight of each face
        # First calculate the perimeter this constant latitude circle
        candidate_faces_weight_list = [0.0] * len(candidate_faces_index_list)

        for i in range(0, len(candidate_faces_index_list)):
            face_index = candidate_faces_index_list[i]
            [face_lon_bound_min, face_lon_bound_max] = self.ds["Mesh2_latlon_bounds"].values[face_index][1]
            face = self.ds["Mesh2_face_edges"].values[face_index]
            x = self.ds["Mesh2_node_cart_x"].values[face[:,0]]
            y = self.ds["Mesh2_node_cart_y"].values[face[:,0]]
            z = self.ds["Mesh2_node_cart_z"].values[face[:,0]]
            pt_lon_min = 3 * np.pi
            pt_lon_max = -3 * np.pi
            face_cart_temp = list(np.stack((x,y,z), axis=-1))
            face_latlon = list(map(convert_node_xyz_to_lonlat_rad,face_cart_temp))
            intersections_pts_list_lonlat = []
            for j in range(0, len(face)):
                edge = face[j]
                # Get the edge end points in 3D [x, y, z] coordinates
                n1 = [self.ds["Mesh2_node_cart_x"].values[edge[0]],
                      self.ds["Mesh2_node_cart_y"].values[edge[0]],
                      self.ds["Mesh2_node_cart_z"].values[edge[0]]]
                n2 = [self.ds["Mesh2_node_cart_x"].values[edge[1]],
                      self.ds["Mesh2_node_cart_y"].values[edge[1]],
                      self.ds["Mesh2_node_cart_z"].values[edge[1]]]
                n1_lonlat = convert_node_xyz_to_lonlat_rad(n1)
                n2_lonlat = convert_node_xyz_to_lonlat_rad(n2)
                intersections = get_intersection_pt([n1, n2], latitude_rad)
                if intersections[0] == [-1, -1, -1] and intersections[1] == [-1, -1, -1]:
                    # The constant latitude didn't cross this edge
                    continue
                elif intersections[0] != [-1, -1, -1] and intersections[1] != [-1, -1, -1]:
                    # The constant latitude goes across this edge ( 1 in and 1 out):
                    pts1_lonlat = convert_node_xyz_to_lonlat_rad(intersections[0])
                    pts2_lonlat = convert_node_xyz_to_lonlat_rad(intersections[1])
                    intersections_pts_list_lonlat.append(convert_node_xyz_to_lonlat_rad(intersections[0]))
                    intersections_pts_list_lonlat.append(convert_node_xyz_to_lonlat_rad(intersections[1]))
                else:
                    if intersections[0] != [-1, -1, -1]:
                        pts1_lonlat = convert_node_xyz_to_lonlat_rad(intersections[0])
                        intersections_pts_list_lonlat.append(convert_node_xyz_to_lonlat_rad(intersections[0]))
                    else:
                        pts2_lonlat = convert_node_xyz_to_lonlat_rad(intersections[1])
                        intersections_pts_list_lonlat.append(convert_node_xyz_to_lonlat_rad(intersections[1]))
            if len(intersections_pts_list_lonlat) == 2:
                [pt_lon_min, pt_lon_max] = np.sort([intersections_pts_list_lonlat[0][0], intersections_pts_list_lonlat[1][0]])
            if face_lon_bound_min < face_lon_bound_max:
                # Normal case
                cur_face_mag_rad = pt_lon_max - pt_lon_min
            else:
                # Longitude wrap-around
                #TODO: Need to think more marginal cases

                if pt_lon_max >= np.pi and pt_lon_min >= np.pi:
                    # They're both on the "left side" of the 0-lon
                    cur_face_mag_rad = pt_lon_max - pt_lon_min
                if pt_lon_max <= np.pi and pt_lon_min <= np.pi:
                    # They're both on the "right side" of the 0-lon
                    cur_face_mag_rad = pt_lon_max - pt_lon_min
                else:
                    # They're at the different side of the 0-lon
                    cur_face_mag_rad = 2 * np.pi - pt_lon_max + pt_lon_min
            if cur_face_mag_rad > np.pi:
                print("At face: "+str(face_index)+"Problematic lat is "+str(latitude_rad)+" And the cur_face_mag_rad is "+str(cur_face_mag_rad))
            # assert(cur_face_mag_rad <= np.pi)

            # Calculate the weight from each face by |intersection line length| / total perimeter
            candidate_faces_weight_list[i] = cur_face_mag_rad

        # Sum up all the weights to get the total
        candidate_faces_weight_list = np.array(candidate_faces_weight_list) / np.sum(candidate_faces_weight_list)
        return candidate_faces_weight_list











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
