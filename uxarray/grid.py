"""uxarray grid module."""
import os
import xarray as xr
import numpy as np
from pathlib import PurePath
from datetime import datetime


class Grid:
    """The Uxarray Grid object class that describes an unstructured grid.
    Examples
    ----------
    Open an exodus file with Grid object
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
                 mesh_filetype=None,
                 **kwargs):
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

        Raises:
            RuntimeError: File not found
        """
        # TODO: fix when adding/exercising gridspec
        self.filepath = None
        self.gridspec = None
        self.vertices = None
        self.islatlon = None
        self.concave = None
        self.mesh_filetype = None

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

    # load mesh from a file
    def __from_file__(self):
        """Loads a mesh file Also, called by __init__ routine This routine will
        automatically detect if it is a UGrid, SCRIP, Exodus, or shape file.

        Raises:
            RuntimeError: Invalid file type
        """
        # call function to set mesh file type: self.mesh_filetype
        self.mesh_filetype = self.find_type(self.filepath)

        # call reader as per mesh_filetype
        if self.mesh_filetype == "exo":
            self.read_exodus(self.filepath)
        elif self.mesh_filetype == "scrip":
            self.read_scrip(self.filepath)
        elif self.mesh_filetype == "ugrid":
            self.read_ugrid(self.filepath)
        elif self.mesh_filetype == "shp":
            self.read_shpfile(self.filepath)
        else:
            raise RuntimeError("unknown file format:" + self.mesh_filetype)

    # helper function to find file type
    def find_type(self, filepath):
        """Checks file path and contents to determine file type, called by
        __from_file__ routine. Automatically detection supported for UGrid,
        SCRIP, Exodus, or shape file.

        Raises:
            RuntimeError: Invalid file type
        """
        msg = ""
        # exodus with coord
        try:
            # extract the file name and extension
            path = PurePath(filepath)
            file_extension = path.suffix

            # try to open file with xarray and test for exodus
            ext_ds = xr.open_dataset(filepath, mask_and_scale=False)["coord"]
            mesh_filetype = "exo"
        except KeyError as e:
            # exodus with coordx
            try:
                ext_ds = xr.open_dataset(filepath,
                                         mask_and_scale=False)["coordx"]
                mesh_filetype = "exo"
            except KeyError as e:
                # scrip with grid_center_lon
                try:
                    ext_ds = xr.open_dataset(
                        filepath, mask_and_scale=False)["grid_center_lon"]
                    mesh_filetype = "scrip"
                except KeyError as e:
                    # ugrid with Mesh2
                    try:
                        ext_ds = xr.open_dataset(filepath,
                                                 mask_and_scale=False)["Mesh2"]
                        mesh_filetype = "ugrid"
                    except KeyError as e:
                        print("This is not a supported NetCDF file")
        except (TypeError, AttributeError) as e:
            msg = str(e) + ': {}'.format(filepath)
        except (RuntimeError, OSError) as e:
            # check if this is a shp file
            # we won't use xarray to load that file
            if file_extension == ".shp":
                mesh_filetype = "shp"
            else:
                msg = str(e) + ': {}'.format(filepath)
        except ValueError as e:
            # check if this is a shp file
            # we won't use xarray to load that file
            if file_extension == ".shp":
                mesh_filetype = "shp"
            else:
                msg = str(e) + ': {}'.format(filepath)
        finally:
            if (msg != ""):
                msg = str(e) + ': {}'.format(filepath)
                print(msg)
                raise RuntimeError(msg)

        return mesh_filetype

    # initialize mesh2 DataVariable for uxarray
    def __init_mesh2__(self):
        # set default attrs and initialize Datavariable "Mesh2" for uxarray
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

    # renames the grid file
    def saveas_file(self, filename):
        """Saves the loaded mesh file as UGRID file
        Parameters
        ----------
        filename : string, required"""
        path = PurePath(self.filepath)
        new_filepath = path.parent / filename
        self.filepath = str(new_filepath)
        self.write_ugrid(self.filepath)
        print(self.filepath)

    def write(self, outfile, format=""):
        """General write function It calls the approriate file writer based on
        the extension of the output file requested."""
        if format == "":
            path = PurePath(outfile)
            format = path.suffix

        if format == ".ugrid" or format == ".ug":
            self.write_ugrid(outfile)
        elif format == ".g" or format == ".exo":
            self.write_exodus(outfile)
        else:
            print("Format not supported for writing. ", format)

    # Exodus Number is one-based.
    def read_exodus(self, filepath):
        """Exodus file reader."""
        print("Reading exodus file: ", self.filepath)

        # Not loading specific variables.
        # as there is no way to know number of face types etc. without loading
        # connect1, connect2, connect3, etc..
        ext_ds = xr.open_dataset(self.filepath, mask_and_scale=False)

        # populate self.in_ds
        self.__init_mesh2__()

        # find max face nodes
        max_face_nodes = 0
        for dim in ext_ds.dims:
            if "num_nod_per_el" in dim:
                if ext_ds.dims[dim] > max_face_nodes:
                    max_face_nodes = ext_ds.dims[dim]

        # create an empty conn array for storing all blk face_nodes_data
        conn = np.empty((0, max_face_nodes))

        for key, value in ext_ds.variables.items():
            if key == "qa_records":
                pass
            elif key == "coord":
                self.in_ds.Mesh2.attrs['topology_dimension'] = np.int32(
                    ext_ds.dims['num_dim'])
                self.in_ds["Mesh2_node_x"] = xr.DataArray(
                    data=ext_ds.coord[0],
                    dims=["nMesh2_node"],
                    attrs={
                        "standard_name": "longitude",
                        "long_name": "longitude of mesh nodes",
                        "units": "degress_east",
                    })
                self.in_ds["Mesh2_node_y"] = xr.DataArray(
                    data=ext_ds.coord[1],
                    dims=["nMesh2_node"],
                    attrs={
                        "standard_name": "lattitude",
                        "long_name": "latitude of mesh nodes",
                        "units": "degrees_north",
                    })
                self.in_ds["Mesh2_node_z"] = xr.DataArray(
                    data=ext_ds.coord[2],
                    dims=["nMesh2_node"],
                    attrs={
                        "standard_name": "spherical",
                        "long_name": "elevation",
                        "units": "degree",
                    })
            elif key == "coordx":
                self.in_ds["Mesh2_node_x"] = xr.DataArray(
                    data=ext_ds.coordx,
                    dims=["nMesh2_node"],
                    attrs={
                        "standard_name": "longitude",
                        "long_name": "longitude of mesh nodes",
                        "units": "degress_east",
                    })
            elif key == "coordy":
                self.in_ds["Mesh2_node_y"] = xr.DataArray(
                    data=ext_ds.coordx,
                    dims=["nMesh2_node"],
                    attrs={
                        "standard_name": "lattitude",
                        "long_name": "latitude of mesh nodes",
                        "units": "degrees_north",
                    })
            elif key == "coordz":
                self.in_ds["Mesh2_node_z"] = xr.DataArray(
                    data=ext_ds.coordx,
                    dims=["nMesh2_node"],
                    attrs={
                        "standard_name": "spherical",
                        "long_name": "elevation",
                        "units": "degree",
                    })
            elif "connect" in key:
                # check if num face nodes is less than max.
                if (value.data.shape[1] < max_face_nodes):
                    # create a temporary array to store connectivity
                    tmp_conn = np.empty((value.data.shape[0], max_face_nodes))
                    tmp_conn.fill(
                        0
                    )  # exodus in 1-based, fill with zeros here; during assignment subtract 1
                    tmp_conn[:value.data.shape[0], :value.data.
                             shape[1]] = value.data
                elif (value.data.shape[1] == max_face_nodes):
                    tmp_conn = value.data
                else:
                    raise (
                        "found face_nodes_dim greater than nMaxMesh2_face_nodes"
                    )

                # concatenate to the previous blk
                conn = np.concatenate((conn, tmp_conn))
                # find the elem_type as etype for this element
                for k, v in value.attrs.items():
                    if k == "elem_type":
                        # TODO: etype if not used now, remove if it'll never be required
                        etype = v

        # outside the k,v for loop
        # set the face nodes data compiled in "connect" section
        self.in_ds["Mesh2_face_nodes"] = xr.DataArray(
            data=(conn[:] - 1),
            dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
            attrs={
                "cf_role":
                    "face_node_connectivity",
                "_FillValue":
                    -1,
                "start_index":
                    np.int32(
                        0
                    )  #NOTE: This might cause an error if numbering has holes
            })
        print("Finished reading exodus file.")
        # done reading exodus flie, close the external xarray ds object
        ext_ds.close()
        return self.in_ds

    def write_exodus(self, outfile):
        """Exodus file writer
        Parameters
        ----------
        outfile : string, required
            Name of output file"""
        # Note this is 1-based unlike native Mesh2 construct
        print("Writing exodus file: ", outfile)

        self.exo_ds = xr.Dataset()

        path = PurePath(outfile)
        out_filename = path.name

        now = datetime.now()
        date = now.strftime("%m/%d/%Y")
        time = now.strftime("%H:%M:%S")

        title = f"uxarray(" + str(out_filename) + ")" + date + ": " + time
        fp_word = np.int32(8)
        version = np.float32(5.0)
        api_version = np.float32(5.0)
        self.exo_ds.attrs = {
            "api_version": api_version,
            "version": version,
            "floating_point_word_size": fp_word,
            "file_size": 0,
            "title": title
        }

        self.exo_ds["time_whole"] = xr.DataArray(data=[], dims=["time_step"])

        # qa_records
        qa_records = [["uxarray"], ["1.0"], [date], [time]]
        self.exo_ds["qa_records"] = xr.DataArray(data=xr.DataArray(
            np.array(qa_records, dtype="S33")),
                                                 dims=["four", "num_qa_rec"])

        # get orig dimention from Mesh2 attribute topology dimension
        dim = self.in_ds["Mesh2"].topology_dimension

        c_data = []
        if dim == 2:
            c_data = xr.DataArray([
                self.in_ds.Mesh2_node_x.data.tolist(),
                self.in_ds.Mesh2_node_y.data.tolist()
            ])
        elif dim == 3:
            c_data = xr.DataArray([
                self.in_ds.Mesh2_node_x.data.tolist(),
                self.in_ds.Mesh2_node_y.data.tolist(),
                self.in_ds.Mesh2_node_z.data.tolist()
            ])

        self.exo_ds["coord"] = xr.DataArray(data=c_data,
                                            dims=["num_dim", "num_nodes"])

        # process face nodes, this array holds num faces at corresponding location
        # eg num_el_all_blks = [0, 0, 6, 12] signifies 6 TRI and 12 SHELL elements
        num_el_all_blks = np.zeros(self.in_ds.nMaxMesh2_face_nodes.size, "i4")
        # this list stores connectivity without filling
        conn_nofill = []

        # store the number of faces in an array
        for row in self.in_ds.Mesh2_face_nodes.data:

            # find out -1 in each row, this indicates lower than max face nodes
            arr = np.where(row == -1)
            # arr[0].size returns the location of first -1 in the conn list
            # if > 0, arr[0][0] is the num_nodes forming the face
            if arr[0].size > 0:
                # increment the number of faces at the corresponding location
                num_el_all_blks[arr[0][0] - 1] += 1
                # append without -1s eg. [1, 2, 3, -1] to [1, 2, 3]
                # convert to list (for sorting later)
                row = row[:(arr[0][0])].tolist()
                list_node = list(map(int, row))
                conn_nofill.append(list_node)
            elif arr[0].size == 0:
                # increment the number of faces for this "nMaxMesh2_face_nodes" face
                num_el_all_blks[self.in_ds.nMaxMesh2_face_nodes.size - 1] += 1
                # get integer list nodes
                list_node = list(map(int, row.tolist()))
                conn_nofill.append(list_node)
            else:
                raise RuntimeError(
                    "num nodes in conn array is greater than nMaxMesh2_face_nodes. Abort!"
                )
        # get number of blks found
        num_blks = np.count_nonzero(num_el_all_blks)

        # sort connectivity by size, lower dim faces first
        conn_nofill.sort(key=len)

        # get index of blocks found
        nonzero_el_index_blks = np.nonzero(num_el_all_blks)

        # break Mesh2_face_nodes into blks
        start = 0
        for blk in range(num_blks):
            blkID = blk + 1
            str_el_in_blk = "num_el_in_blk" + str(blkID)
            str_nod_per_el = "num_nod_per_el" + str(blkID)
            str_att_in_blk = "num_att_in_blk" + str(blkID)
            str_global_id = "global_id" + str(blkID)
            str_edge_type = "edge_type" + str(blkID)
            str_attrib = "attrib" + str(blkID)
            str_connect = "connect" + str(blkID)

            # get element type
            num_nodes = len(conn_nofill[start])
            element_type = self.__get_element_type__(num_nodes)

            # get number of faces for this block
            num_faces = num_el_all_blks[nonzero_el_index_blks[0][blk]]
            # assign Data variables
            # convert list to np.array, sorted list guarantees we have the correct info
            conn_blk = conn_nofill[start:start + num_faces]
            conn_np = np.array([np.array(xi, dtype="i4") for xi in conn_blk])
            self.exo_ds[str_connect] = xr.DataArray(
                data=xr.DataArray((conn_np[:] + 1)),
                dims=[str_el_in_blk, str_nod_per_el],
                attrs={"elem_type": element_type})

            # edgetype
            self.exo_ds[str_edge_type] = xr.DataArray(
                data=xr.DataArray(np.zeros((num_faces, num_nodes), "i4")),
                dims=[str_el_in_blk, str_nod_per_el])

            # global id
            gid = np.arange(start + 1, start + num_faces + 1, 1)
            self.exo_ds[str_global_id] = xr.DataArray(data=(gid),
                                                      dims=[str_el_in_blk])

            # attrib
            # TODO: fix num attr
            num_attr = 1
            self.exo_ds[str_attrib] = xr.DataArray(
                data=xr.DataArray(np.zeros((num_faces, num_attr), float)),
                dims=[str_el_in_blk, str_att_in_blk])

            start = num_faces

        # blk for loop ends

        # eb_prop1
        prop1_vals = np.arange(1, num_blks + 1, 1)
        self.exo_ds["eb_prop1"] = xr.DataArray(data=prop1_vals,
                                               dims=["num_el_blk"],
                                               attrs={"name": "ID"})
        # eb_status
        self.exo_ds["eb_status"] = xr.DataArray(data=xr.DataArray(
            np.ones([num_blks], dtype="i4")),
                                                dims=["num_el_blk"])

        # eb_names
        eb_names = np.empty(num_blks, dtype="S33")
        eb_names.fill("")
        self.exo_ds["eb_names"] = xr.DataArray(data=xr.DataArray(eb_names),
                                               dims=["num_el_blk"])

        if dim == 2:
            cnames = ["x", "y"]
        elif dim == 3:
            cnames = ["x", "y", "z"]

        self.exo_ds["coor_names"] = xr.DataArray(data=xr.DataArray(
            np.array(cnames, dtype="S33")),
                                                 dims=["num_dim"])

        # done processing write the file to disk
        self.exo_ds.to_netcdf(outfile)
        print("Wrote: ", outfile)
        # now close the dataset
        self.exo_ds.close()

    def __get_element_type__(self, num_nodes):
        ELEMENT_TYPE_DICT = {
            2: "BEAM",
            3: "TRI",
            4: "SHELL4",
            5: "SHELL5",
            6: "TRI6",
            7: "TRI7",
            8: "SHELL8",
        }
        element_type = ELEMENT_TYPE_DICT[num_nodes]
        return element_type

    def read_ugrid(self, filename):
        """Returns the xarray Dataset loaded during init."""

        ext_ds = xr.open_dataset(self.filepath, mask_and_scale=False)
        # simply return the xarray object loaded
        self.in_ds = ext_ds
        ext_ds.close()
        return self.in_ds

    # Write a uxgrid to a file with specified format.
    def write_ugrid(self, outfile):
        """Function to write ugrid, uses to_netcdf from xarray object."""
        print("Writing ugrid file: ", outfile)
        self.in_ds.to_netcdf(outfile)

    def read_scrip(self, filename):
        """This is just a function placeholder.

        Not implemented.
        """
        print("Function placeholder, implementation coming soon.")
        print("Reading SCRIP file: ", self.filepath)
        # populate self.in_ds

    def read_shpfile(self, filename):
        """This is just a function placeholder.

        Not implemented.
        """
        print("Function placeholder, implementation coming soon.")
        print("Reading shape file: ", self.filepath)
        # populate self.in_ds

    # Calculate the area of all faces.
    def calculate_total_face_area(self):
        """This is just a function placeholder.

        Not implemented.
        """
        print("Function placeholder, implementation coming soon.")

    # Build the node-face connectivity array.
    def build_node_face_connectivity(self):
        """This is just a function placeholder.

        Not implemented.
        """
        print("Function placeholder, implementation coming soon.")

    # Build the edge-face connectivity array.
    def build_edge_face_connectivity(self):
        """This is just a function placeholder.

        Not implemented.
        """
        print("Function placeholder, implementation coming soon.")

    # Build the array of latitude-longitude bounding boxes.
    def buildlatlon_bounds(self):
        """This is just a function placeholder.

        Not implemented.
        """
        print("Function placeholder, implementation coming soon.")

    # Validate that the grid conforms to the UXGrid standards.
    def validate(self):
        """This is just a function placeholder.

        Not implemented.
        """
        print("Function placeholder, implementation coming soon.")
