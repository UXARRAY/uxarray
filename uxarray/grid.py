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
from .helpers import determine_file_type, insert_pt_in_latlonbox, convert_node_XYZ_2_latlon_rad


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
        self.Mesh2_latlon_bounds = None
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
        temp_latlon_array = np.zeros(self.ds_var_names["Mesh2_face_nodes"])

        reference_tolerance = 1.0e-12

        '''
        Generate the face_edge_connectivity:Mesh2_face_edges
        from Mesh2_face_nodes and Mesh2_edge_nodes
        '''
        # For each edge in the Grid, check if both nodes are in the self.Mesh2_face_nodes
        mesh2_face_edges = np.zeros(self.ds_var_names["Mesh2_face_nodes"])
        for i in range(0, self.ds_var_names["Mesh2_face_nodes"]):
            cur_Mesh2_face_nodes = self.Mesh2_face_nodes.values[i]
            cur_mesh2_face_edges = []

            for edge in self.Mesh2_edge_nodes:
                node_1 = edge.values[0]
                node_2 = edge.values[1]
                if node_1 in cur_Mesh2_face_nodes and node_2 in edge.values[1]:
                    cur_mesh2_face_edges.append(edge)

            mesh2_face_edges[i] = cur_mesh2_face_edges

        '''
        Loop over each face to generate LatlonBox
        '''
        for i in range(len(self.mesh2_face_edges)):
            face = self.mesh2_face_edges[i]
            i_winding_number = 0
            for edge in face:
                node_1 = edge[0]
                node_2 = edge[1]

                if (self.Mesh2_node_z[node_1] <= -reference_tolerance) and (
                        self.Mesh2_node_z[node_2] >= reference_tolerance):
                    i_winding_number = 0
                    break

                if (self.Mesh2_node_z[node_1] >= reference_tolerance) and (
                        self.Mesh2_node_z[node_2] <= -reference_tolerance):
                    i_winding_number = 0
                    break

                if np.absolute(np.absolute(self.Mesh2_node_z[node_1]) - 1.0) < reference_tolerance:
                    i_winding_number = 2000
                    break

                if np.absolute(self.Mesh2_node_y[node_1]) < reference_tolerance:
                    if np.absolute(self.Mesh2_node_y[node_2]) < reference_tolerance:
                        if (self.Mesh2_node_x[node_1] >= reference_tolerance) and (
                                self.Mesh2_node_x[node_1] <= reference_tolerance):
                            i_winding_number = 3000
                            break
                        if (self.Mesh2_node_x[node_1] <= reference_tolerance) and (
                                self.Mesh2_node_x[node_1] >= reference_tolerance):
                            i_winding_number = 4000
                            break
                    continue

                if (self.Mesh2_node_y[node_1] <= reference_tolerance) and (
                        self.Mesh2_node_y[node_2] <= reference_tolerance):
                    continue

                if (self.Mesh2_node_y[node_1] >= reference_tolerance) and (
                        self.Mesh2_node_y[node_2] >= reference_tolerance):
                    continue

                d_denom = self.Mesh2_node_y[node_1] * self.Mesh2_node_z[node_2] - self.Mesh2_node_y[node_2] * \
                          self.Mesh2_node_y[node_1] * self.Mesh2_node_z[node_1]

                if np.absolute(d_denom) < reference_tolerance:
                    continue

                d_xint = (self.Mesh2_node_y[node_1] * self.Mesh2_node_x[node_1] * self.Mesh2_node_y[node_2]) / d_denom
                if d_xint > 0.0:
                    if (self.Mesh2_node_y[node_1] < reference_tolerance) and (
                            self.Mesh2_node_y[node_2] > reference_tolerance):
                        i_winding_number += 1
                    else:
                        i_winding_number -= 1
                else:
                    if (self.Mesh2_node_y[node_1] < reference_tolerance) and (
                            self.Mesh2_node_y[node_2] > reference_tolerance):
                        i_winding_number -= 1
                    else:
                        i_winding_number += 1

            # Face containing a pole point; latlon box determined by minimum/maximum latitude
            if np.absolute(i_winding_number) > 1:
                d_lat_extent_rad = 0.0
                for j in range(len(face)):
                    edge = face[i]
                    node_1 = edge[0]
                    node_2 = edge[1]

                    d_lat_rad = 0.0
                    d_lon_rad = 0.0

                    if j == 0:
                        if self.Mesh2_node_z[node_1] < 0.0:
                            d_lat_extent_rad = -0.5 * np.pi
                        else:
                            d_lat_extent_rad = 0.5 * np.pi

                    # Insert edge endpoint into box
                    # temp_latlon_box = [d_lat_rad,d_lon_rad]
                    # temp_latlon_array[j]
                    if np.absolute(d_lat_rad) < d_lat_extent_rad:
                        d_lat_extent_rad = d_lat_rad

                    # Edges that are great circle arc:
                    if self.Mesh2_edge_types[edge] == 0:

                        # Determine if latitude is maximized between endpoints
                        d_dot_n1_n2 = self.Mesh2_node_x[node_1] * self.Mesh2_node_x[node_2] + self.Mesh2_node_y[
                            node_1] * self.Mesh2_node_y[node_2] + self.Mesh2_node_z[node_1] * self.Mesh2_node_z[node_2]
                        d_denom = (self.Mesh2_node_z[node_1] * self.Mesh2_node_z[node_2]) * (d_dot_n1_n2 - 1.0)

                        # Maximum latitude occurs between endpoints of edge
                        d_a_max = (self.Mesh2_node_z[node_1] * d_dot_n1_n2 - self.Mesh2_node_z[node_2]) / d_denom
                        if (d_a_max > 0.0) and (d_a_max < 1.0):
                            node_3_x = self.Mesh2_node_x[node_1] * (1.0 - d_a_max) + self.Mesh2_node_x[node_2] * d_a_max
                            node_3_y = self.Mesh2_node_y[node_1] * (1.0 - d_a_max) + self.Mesh2_node_y[node_2] * d_a_max
                            node_3_z = self.Mesh2_node_z[node_1] * (1.0 - d_a_max) + self.Mesh2_node_z[node_2] * d_a_max
                            magnitude = np.sqrt(node_3_x ** 2 + node_3_y ** 2 + node_3_z ** 2)
                            node_3_x /= magnitude
                            node_3_y /= magnitude
                            node_3_z /= magnitude

                            d_lat_rad = node_3_z
                            if d_lat_rad > 1.0:
                                d_lat_rad = 0.5 * np.pi
                            elif d_lat_rad < -1.0:
                                d_lat_rad = -0.5 * np.pi
                            else:
                                d_lat_rad = np.arcsin(d_lat_rad)

                            if np.absolute(d_lat_rad) < np.absolute(d_lat_extent_rad):
                                d_lat_extent_rad = d_lat_rad
                    # Edges that are lines of constant latitude
                    elif self.Mesh2_edge_types[edge] == 1:
                        pass
                    else:
                        raise Exception('Unsupported edge type {}'.format(self.Mesh2_edge_types[edge]))

                    if d_lat_extent_rad < 0.0:
                        lat_lon_box = [[-0.5 * np.pi, d_lat_extent_rad], [0.0, 2.0 * np.pi]]#[lat_array, lon_array]
                        temp_latlon_array[i] = lat_lon_box
                    else:
                        lat_lon_box = [[d_lat_extent_rad, 0.5 * np.pi], [0.0, 2.0 * np.pi]]#[lat_array, lon_array]
                        temp_latlon_array[i] = lat_lon_box
            # Normal face
            else:
                for j in range(len(face)):
                    edge = face[i]

                    # Edges that are great circle arcs
                    if self.Mesh2_edge_types[edge] == 0:
                        edge = face[i]
                        node_1 = edge[0]
                        node_2 = edge[1]

                        # Determine if latitude is maximized between endpoints
                        d_dot_n1_n2 = self.Mesh2_node_x[node_1] * self.Mesh2_node_x[node_2] + self.Mesh2_node_y[
                            node_1] * self.Mesh2_node_y[node_2] + self.Mesh2_node_z[node_1] * self.Mesh2_node_z[node_2]
                        d_denom = (self.Mesh2_node_z[node_1] * self.Mesh2_node_z[node_2]) * (d_dot_n1_n2 - 1.0)
                        d_lat_rad = 0.0
                        d_lon_rad = 0.0

                        # Create the first end points of the edge

                        res_pt = convert_node_XYZ_2_latlon_rad([self.Mesh2_node_x[node_1], self.Mesh2_node_y[node_1]], self.Mesh2_node_z[node_1])
                        d_lat_rad = res_pt[0]
                        d_lon_rad = res_pt[1]


                        lat_lon_box = [[d_lat_rad, d_lat_rad],[d_lon_rad, d_lon_rad]]
                        temp_latlon_array[i] =  lat_lon_box

                        # Either repeated point or Amax out of range
                        if np.absolute(d_denom) < reference_tolerance:
                            continue

                        # Maximum latitude occurs between endpoints of edge
                        d_a_max = (self.Mesh2_node_z[node_1] * d_dot_n1_n2 - self.Mesh2_node_z[node_2]) / d_denom
                        if d_a_max > 0.0 and d_a_max < 1.0:
                            node_3_x = self.Mesh2_node_x[node_1] * (1.0 - d_a_max) + self.Mesh2_node_x[node_2] * d_a_max
                            node_3_y = self.Mesh2_node_y[node_1] * (1.0 - d_a_max) + self.Mesh2_node_y[node_2] * d_a_max
                            node_3_z = self.Mesh2_node_z[node_1] * (1.0 - d_a_max) + self.Mesh2_node_z[node_2] * d_a_max
                            magnitude = np.sqrt(node_3_x ** 2 + node_3_y ** 2 + node_3_z ** 2)
                            node_3_x /= magnitude
                            node_3_y /= magnitude
                            node_3_z /= magnitude

                            d_lat_rad = node_3_z

                            if d_lat_rad > 1.0:
                                d_lat_rad = 0.5 * np.pi
                            elif d_lat_rad < -1.0:
                                d_lat_rad = -0.5 * np.pi
                            else:
                                d_lat_rad = np.arcsin(d_lat_rad)

                            temp_latlon_array[i] = insert_pt_in_latlonbox(temp_latlon_array[i], [d_lat_rad, d_lon_rad])
                        # Edges that are lines of constant latitude
                        elif self.Mesh2_edge_types[edge] == 1:
                            edge = face[i]
                            node_1 = edge[0]
                            node_2 = edge[1]

                            res_pt = convert_node_XYZ_2_latlon_rad(
                                [self.Mesh2_node_x[node_1], self.Mesh2_node_y[node_1]], self.Mesh2_node_z[node_1])
                            d_lat_rad = res_pt[0]
                            d_lon_rad = res_pt[1]

                            temp_latlon_array[i] = insert_pt_in_latlonbox(temp_latlon_array[i], [d_lat_rad, d_lon_rad])

                            res_pt = convert_node_XYZ_2_latlon_rad(
                                [self.Mesh2_node_x[node_2], self.Mesh2_node_y[node_2]], self.Mesh2_node_z[node_2])
                            d_lat_rad = res_pt[0]
                            d_lon_rad = res_pt[1]

                            temp_latlon_array[i] = insert_pt_in_latlonbox(temp_latlon_array[i], [d_lat_rad, d_lon_rad])



                            # n1.ToLatLonRad(dLatRad, dLonRad);
                            # vecLatLonBoxes[i].insert(dLatRad, dLonRad);
                            #
                            # n2.ToLatLonRad(dLatRad, dLonRad);
                            # vecLatLonBoxes[i].insert(dLatRad, dLonRad);

                        else:
                            raise Exception('Unsupported edge type {}'.format(self.Mesh2_edge_types[edge]))


























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
