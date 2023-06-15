"""uxarray.core.grid module."""
import xarray as xr
import numpy as np

# reader and writer imports
from uxarray.io._exodus import _read_exodus, _encode_exodus
from uxarray.io._mpas import _read_mpas
from uxarray.io._ugrid import _read_ugrid, _encode_ugrid
from uxarray.io._shapefile import _read_shpfile
from uxarray.io._scrip import _read_scrip, _encode_scrip
from uxarray.utils.helpers import (get_all_face_area_from_coords,
                                   parse_grid_type, node_xyz_to_lonlat_rad,
                                   node_lonlat_rad_to_xyz, close_face_nodes)
from uxarray.utils.constants import INT_DTYPE, INT_FILL_VALUE


class Grid:
    """Unstructured grid topology definition.

    Can be used standalone to explore an unstructured grid topology, or
    can be seen as the property of ``uxarray.UxDataset`` and ``uxarray.DataArray``
    to make them unstructured grid-aware data sets and arrays.

    Parameters
    ----------
    input_obj : xarray.Dataset, ndarray, list, tuple, required
        Input ``xarray.Dataset`` or vertex coordinates that form faces.

    Other Parameters
    ----------------
    gridspec: bool, optional
        Specifies gridspec
    islatlon : bool, optional
        Specify if the grid is lat/lon based
    isconcave: bool, optional
        Specify if this grid has concave elements (internal checks for this are possible)
    use_dual: bool, optional
        Specify whether to use the primal (use_dual=False) or dual (use_dual=True) mesh if the file type is mpas

    Raises
    ------
        RuntimeError
            If specified file not found or recognized

    Examples
    ----------

    >>> import uxarray as ux

    1. Open a grid file with `uxarray.open_grid()`:

    >>> uxgrid = ux.open_grid("filename.g")

    2. Open an unstructured grid dataset file with
    `uxarray.open_dataset()`, then access `Grid` info:

    >>> uxds = ux.open_dataset("filename.g")
    """

    def __init__(self, input_obj, **kwargs):
        # initialize internal variable names
        self.__init_grid_var_names__()

        # initialize face_area variable
        self._face_areas = None

        # TODO: fix when adding/exercising gridspec

        # unpack kwargs
        # sets default values for all kwargs to None
        kwargs_list = [
            'gridspec', 'vertices', 'islatlon', 'isconcave', 'source_grid',
            'use_dual'
        ]
        for key in kwargs_list:
            setattr(self, key, kwargs.get(key, None))

        # check if initializing from verts:
        if isinstance(input_obj, (list, tuple, np.ndarray)):
            input_obj = np.asarray(input_obj)
            self.mesh_type = "From vertices"
            # grid with multiple faces
            if input_obj.ndim == 3:
                self.__from_vert__(input_obj)
                self.source_grid = "From vertices"
            # grid with a single face
            elif input_obj.ndim == 2:
                input_obj = np.array([input_obj])
                self.__from_vert__(input_obj)
                self.source_grid = "From vertices"
            else:
                raise RuntimeError(
                    f"Invalid Input Dimension: {input_obj.ndim}. Expected dimension should be "
                    f"3: [nMesh2_face, nMesh2_node, Two/Three] or 2 when only "
                    f"one face is passed in.")

        # check if initializing from string
        # TODO: re-add gridspec initialization when implemented
        elif isinstance(input_obj, xr.Dataset):
            self.mesh_type = parse_grid_type(input_obj)
            self.__from_ds__(dataset=input_obj)
        else:
            raise RuntimeError("Dataset is not a valid input type.")

        # {"Standardized Name" : "Original Name"}
        self._inverse_grid_var_names = {
            v: k for k, v in self.grid_var_names.items()
        }

        # construct nNodes_per_Face
        self._build_nNodes_per_face()

    def __init_grid_var_names__(self):
        """Populates a dictionary for storing uxarray's internal representation
        of xarray object.

        Note ugrid conventions are flexible with names of variables, see:
        http://ugrid-conventions.github.io/ugrid-conventions/
        """
        self.grid_var_names = {
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

    def __from_vert__(self, dataset):
        """Create a grid with faces constructed from vertices specified by the
        given argument.

        Parameters
        ----------
        dataset : ndarray, list, tuple, required
            Input vertex coordinates that form our face(s)
        """
        self._ds = xr.Dataset()
        self._ds["Mesh2"] = xr.DataArray(
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Topology data of unstructured mesh",
                "topology_dimension": -1,
                "node_coordinates": "Mesh2_node_x Mesh2_node_y Mesh2_node_z",
                "node_dimension": "nMesh2_node",
                "face_node_connectivity": "Mesh2_face_nodes",
                "face_dimension": "nMesh2_face"
            })

        self._ds.Mesh2.attrs['topology_dimension'] = dataset.ndim

        if self.islatlon is not None and self.islatlon is False:
            x_units = 'm'
            y_units = 'm'
            z_units = 'm'
        else:
            x_units = "degrees_east"
            y_units = "degrees_north"
            z_units = "elevation"

        x_coord = dataset[:, :, 0].flatten()
        y_coord = dataset[:, :, 1].flatten()

        if dataset[0][0].size > 2:
            z_coord = dataset[:, :, 2].flatten()
        else:
            z_coord = x_coord * 0.0

        # Identify unique vertices and their indices
        unique_verts, indices = np.unique(dataset.reshape(
            -1, dataset.shape[-1]),
                                          axis=0,
                                          return_inverse=True)

        # Nodes index that contain a fill value
        fill_value_mask = np.logical_or(unique_verts[:, 0] == INT_FILL_VALUE,
                                        unique_verts[:, 1] == INT_FILL_VALUE)
        if dataset[0][0].size > 2:
            fill_value_mask = np.logical_or(
                unique_verts[:, 0] == INT_FILL_VALUE,
                unique_verts[:, 1] == INT_FILL_VALUE,
                unique_verts[:, 2] == INT_FILL_VALUE)

        # Get the indices of all the False values in fill_value_mask
        false_indices = np.where(fill_value_mask == True)[0]

        # Check if any False values were found
        indices = indices.astype(INT_DTYPE)
        if false_indices.size > 0:

            # Remove the rows corresponding to False values in unique_verts
            unique_verts = np.delete(unique_verts, false_indices, axis=0)

            # Update indices accordingly
            for i, idx in enumerate(false_indices):
                indices[indices == idx] = INT_FILL_VALUE
                indices[(indices > idx) & (indices != INT_FILL_VALUE)] -= 1

        # Create coordinate DataArrays
        self._ds["Mesh2_node_x"] = xr.DataArray(data=unique_verts[:, 0],
                                                dims=["nMesh2_node"],
                                                attrs={"units": x_units})
        self._ds["Mesh2_node_y"] = xr.DataArray(data=unique_verts[:, 1],
                                                dims=["nMesh2_node"],
                                                attrs={"units": y_units})
        if dataset.shape[-1] > 2:
            self._ds["Mesh2_node_z"] = xr.DataArray(data=unique_verts[:, 2],
                                                    dims=["nMesh2_node"],
                                                    attrs={"units": z_units})
        else:
            self._ds["Mesh2_node_z"] = xr.DataArray(data=unique_verts[:, 1] *
                                                    0.0,
                                                    dims=["nMesh2_node"],
                                                    attrs={"units": z_units})

        # Create connectivity array using indices of unique vertices
        connectivity = indices.reshape(dataset.shape[:-1])
        self._ds["Mesh2_face_nodes"] = xr.DataArray(
            data=xr.DataArray(connectivity).astype(INT_DTYPE),
            dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
            attrs={
                "cf_role": "face_node_connectivity",
                "_FillValue": INT_FILL_VALUE,
                "start_index": 0
            })

    # load mesh from a file
    def __from_ds__(self, dataset):
        """Loads a mesh dataset."""
        # call reader as per mesh_type
        if self.mesh_type == "exo":
            self._ds = _read_exodus(dataset, self.grid_var_names)
        elif self.mesh_type == "scrip":
            self._ds = _read_scrip(dataset)
        elif self.mesh_type == "ugrid":
            self._ds, self.grid_var_names = _read_ugrid(dataset,
                                                        self.grid_var_names)
        elif self.mesh_type == "shp":
            self._ds = _read_shpfile(dataset)
        elif self.mesh_type == "mpas":
            # select whether to use the dual mesh
            if self.use_dual is not None:
                self._ds = _read_mpas(dataset, self.use_dual)
            else:
                self._ds = _read_mpas(dataset)
        else:
            raise RuntimeError("unknown mesh type")

        dataset.close()

    def __repr__(self):
        """Constructs a string representation of the contents of a ``Grid``."""

        prefix = "<uxarray.Grid>\n"
        original_grid_str = f"Original Grid Type: {self.mesh_type}\n"
        dims_heading = "Grid Dimensions:\n"
        dims_str = ""
        # if self.grid_var_names["Mesh2_node_x"] in self._ds:
        #     dims_str += f"  * nMesh2_node: {self.nMesh2_node}\n"
        # if self.grid_var_names["Mesh2_face_nodes"] in self._ds:
        #     dims_str += f"  * nMesh2_face: {self.nMesh2_face}\n"
        #     dims_str += f"  * nMesh2_face: {self.nMesh2_face}\n"

        for key, value in zip(self._ds.dims.keys(), self._ds.dims.values()):
            if key in self._inverse_grid_var_names:
                dims_str += f"  * {self._inverse_grid_var_names[key]}: {value}\n"

        if "nMesh2_edge" in self._ds.dims:
            dims_str += f"  * nMesh2_edge: {self.nMesh2_edge}\n"

        if "nMaxMesh2_face_edges" in self._ds.dims:
            dims_str += f"  * nMaxMesh2_face_edges: {self.nMaxMesh2_face_edges}\n"

        coord_heading = "Grid Coordinate Variables:\n"
        coords_str = ""
        if self.grid_var_names["Mesh2_node_x"] in self._ds:
            coords_str += f"  * Mesh2_node_x: {self.Mesh2_node_x.shape}\n"
            coords_str += f"  * Mesh2_node_y: {self.Mesh2_node_y.shape}\n"
        if "Mesh2_node_cart_x" in self._ds:
            coords_str += f"  * Mesh2_node_cart_x: {self.Mesh2_node_cart_x.shape}\n"
            coords_str += f"  * Mesh2_node_cart_y: {self.Mesh2_node_cart_y.shape}\n"
            coords_str += f"  * Mesh2_node_cart_z: {self.Mesh2_node_cart_z.shape}\n"
        if "Mesh2_face_x" in self._ds:
            coords_str += f"  * Mesh2_face_x: {self.Mesh2_face_x.shape}\n"
            coords_str += f"  * Mesh2_face_y: {self.Mesh2_face_y.shape}\n"

        connectivity_heading = "Grid Connectivity Variables:\n"
        connectivity_str = ""
        if self.grid_var_names["Mesh2_face_nodes"] in self._ds:
            connectivity_str += f"  * Mesh2_face_nodes: {self.Mesh2_face_nodes.shape}\n"
        if "Mesh2_edge_nodes" in self._ds:
            connectivity_str += f"  * Mesh2_edge_nodes: {self.Mesh2_edge_nodes.shape}\n"
        if "Mesh2_face_edges" in self._ds:
            connectivity_str += f"  * Mesh2_face_edges: {self.Mesh2_face_edges.shape}\n"
        connectivity_str += f"  * nNodes_per_face: {self.nNodes_per_face.shape}\n"

        return prefix + original_grid_str + dims_heading + dims_str + coord_heading + coords_str + \
            connectivity_heading + connectivity_str

    # attribute properties

    @property
    def parsed_attrs(self):
        """Dictionary of parsed attributes from the source grid."""
        return self._ds.attrs

    @property
    def Mesh2(self):
        """UGRID Attribute ``Mesh2``, which indicates the topology data of a 2D
        unstructured mesh."""
        return self._ds[self.grid_var_names["Mesh2"]]

    # dimension properties

    @property
    def nMesh2_node(self):
        """UGRID Dimension ``nMesh2_node``, which represents the total number
        of nodes."""
        return self._ds[self.grid_var_names["Mesh2_node_x"]].shape[0]

    @property
    def nMesh2_face(self):
        """UGRID Dimension ``nMesh2_face``, which represents the total number
        of faces."""
        return self._ds[self.grid_var_names["Mesh2_face_nodes"]].shape[0]

    @property
    def nMesh2_edge(self):
        """UGRID Dimension ``nMesh2_edge``, which represents the total number
        of edges."""

        if "Mesh2_edge_nodes" not in self._ds:
            self._build_edge_node_connectivity(repopulate=True)

        return self._ds['Mesh2_edge_nodes'].shape[0]

    @property
    def nMaxMesh2_face_nodes(self):
        """UGRID Dimension ``nMaxMesh2_face_nodes``, which represents the
        maximum number of faces nodes that a face may contain."""
        return self.Mesh2_face_nodes.shape[1]

    @property
    def nMaxMesh2_face_edges(self):
        """Dimension ``nMaxMesh2_face_edges``, which represents the maximum
        number of edges per face.

        Equivalent to ``nMaxMesh2_face_nodes``
        """

        if "Mesh2_face_edges" not in self._ds:
            self._build_face_edges_connectivity()

        return self._ds["Mesh2_face_edges"].shape[1]

    @property
    def nNodes_per_face(self):
        """Dimension Variable ``nNodes_per_face``, which contains the number of
        non-fill-value nodes per face.

        Dimensions (``nMesh2_nodes``) and DataType ``INT_DTYPE``.
        """
        return self._ds["nNodes_per_face"]

    # coordinate properties

    @property
    def Mesh2_node_x(self):
        """UGRID Coordinate Variable ``Mesh2_node_x``, which contains the
        longitude of each node in degrees.

        Dimensions (``nMesh2_node``)
        """
        return self._ds[self.grid_var_names["Mesh2_node_x"]]

    @property
    def Mesh2_node_cart_x(self):
        """Coordinate Variable ``Mesh2_node_cart_x``, which contains the x
        location in meters.

        Dimensions (``nMesh2_node``)
        """
        if "Mesh2_node_cart_x" not in self._ds:
            self._populate_cartesian_xyz_coord()
        return self._ds['Mesh2_node_cart_x']

    @property
    def Mesh2_face_x(self):
        """UGRID Coordinate Variable ``Mesh2_face_x``, which contains the
        longitude of each face center.

        Dimensions (``nMesh2_face``)
        """
        if "Mesh2_face_x" in self._ds:
            return self._ds["Mesh2_face_x"]
        else:
            return None

    @property
    def Mesh2_node_y(self):
        """UGRID Coordinate Variable ``Mesh2_node_y``, which contains the
        latitude of each node.

        Dimensions (``nMesh2_node``)
        """
        return self._ds[self.grid_var_names["Mesh2_node_y"]]

    @property
    def Mesh2_node_cart_y(self):
        """Coordinate Variable ``Mesh2_node_cart_y``, which contains the y
        location in meters.

        Dimensions (``nMesh2_node``)
        """
        if "Mesh2_node_cart_y" not in self._ds:
            self._populate_cartesian_xyz_coord()
        return self._ds['Mesh2_node_cart_y']

    @property
    def Mesh2_face_y(self):
        """UGRID Coordinate Variable ``Mesh2_face_y``, which contains the
        latitude of each face center.

        Dimensions (``nMesh2_face``)
        """
        if "Mesh2_face_y" in self._ds:
            return self._ds["Mesh2_face_y"]
        else:
            return None

    @property
    def _Mesh2_node_z(self):
        """Coordinate Variable ``_Mesh2_node_z``, which contains the level of
        each node. It is only a placeholder for now as a protected attribute.
        UXarray does not support this yet and only handles the 2D flexibile
        meshes.

        If we introduce handling of 3D meshes in the future, it might be only
        levels, i.e. the same level(s) for all nodes, instead of separate
        level for each node that ``_Mesh2_node_z`` suggests.

        Dimensions (``nMesh2_node``)
        """
        if self.grid_var_names["Mesh2_node_z"] in self._ds:
            return self._ds[self.grid_var_names["Mesh2_node_z"]]
        else:
            return None

    @property
    def Mesh2_node_cart_z(self):
        """Coordinate Variable ``Mesh2_node_cart_z``, which contains the z
        location in meters.

        Dimensions (``nMesh2_node``)
        """
        if "Mesh2_node_cart_z" not in self._ds:
            self._populate_cartesian_xyz_coord()
        return self._ds['Mesh2_node_cart_z']

    # connectivity properties

    @property
    def Mesh2_face_nodes(self):
        """UGRID Connectivity Variable ``Mesh2_face_nodes``, which maps each
        face to its corner nodes.

        Dimensions (``nMesh2_face``, ``nMaxMesh2_face_nodes``) and
        DataType ``INT_DTYPE``.

        Faces can have arbitrary length, with _FillValue=-1 used when faces
        have fewer nodes than MaxNumNodesPerFace.

        Nodes are in counter-clockwise order.
        """

        return self._ds[self.grid_var_names["Mesh2_face_nodes"]]

    @property
    def Mesh2_edge_nodes(self):
        """UGRID Connectivity Variable ``Mesh2_edge_nodes``, which maps every
        edge to the two nodes that it connects.

        Dimensions (``nMesh2_edge``, ``Two``) and DataType
        ``INT_DTYPE``.

        Nodes are in arbitrary order.
        """
        if "Mesh2_edge_nodes" not in self._ds:
            self._build_edge_node_connectivity()

        return self._ds['Mesh2_edge_nodes']

    @property
    def Mesh2_face_edges(self):
        """UGRID Connectivity Variable ``Mesh2_face_edges``, which maps every
        face to its edges.

        Dimensions (``nMesh2_face``, ``nMaxMesh2_face_nodes``) and
        DataType ``INT_DTYPE``.
        """
        if "Mesh2_face_edges" not in self._ds:
            self._build_face_edges_connectivity()

        return self._ds["Mesh2_face_edges"]

    def copy(self):
        """Returns a deep copy of this grid."""
        return Grid(xr.Dataset(self._ds),
                    gridspec=self.gridspec,
                    vertices=self.vertices,
                    islatlon=self.islatlon,
                    isconcave=self.isconcave,
                    source_grid=self.source_grid,
                    use_dual=self.use_dual)

    def encode_as(self, grid_type):
        """Encodes the grid as a new `xarray.Dataset` per grid format supplied
        in the `grid_type` argument.

        Parameters
        ----------
        grid_type : str, required
            Grid type of output dataset.
            Currently supported options are "ugrid", "exodus", and "scrip"

        Returns
        -------
        out_ds : xarray.Dataset
            The output `xarray.Dataset` that is encoded from the this grid.

        Raises
        ------
        RuntimeError
            If provided grid type or file type is unsupported.
        """

        if grid_type == "ugrid":
            out_ds = _encode_ugrid(self._ds)

        elif grid_type == "exodus":
            out_ds = _encode_exodus(self._ds, self.grid_var_names)

        elif grid_type == "scrip":
            out_ds = _encode_scrip(self.Mesh2_face_nodes, self.Mesh2_node_x,
                                   self.Mesh2_node_y, self.face_areas)
        else:
            raise RuntimeError("The grid type not supported: ", grid_type)

        return out_ds

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

        >>> grid = ux.open_dataset("/home/jain/uxarray/test/meshfiles/ugrid/outCSne30/outCSne30.ug")

        Get area of all faces in the same order as listed in grid._ds.Mesh2_face_nodes

        >>> grid.face_areas
        array([0.00211174, 0.00211221, 0.00210723, ..., 0.00210723, 0.00211221,
            0.00211174])
        """
        # if self._face_areas is None: # this allows for using the cached result,
        # but is not the expected behavior behavior as we are in need to recompute if this function is called with different quadrature_rule or order

        # area of a face call needs the units for coordinate conversion if spherical grid is used
        coords_type = "spherical"
        if not "degree" in self.Mesh2_node_x.units:
            coords_type = "cartesian"

        face_nodes = self.Mesh2_face_nodes.data
        nNodes_per_face = self.nNodes_per_face.data
        dim = self.Mesh2.attrs['topology_dimension']

        # initialize z
        z = np.zeros((self.nMesh2_node))

        # call func to cal face area of all nodes
        x = self.Mesh2_node_x.data
        y = self.Mesh2_node_y.data
        # check if z dimension
        if self.Mesh2.topology_dimension > 2:
            z = self._Mesh2_node_z.data

        # Note: x, y, z are np arrays of type float
        # Using np.issubdtype to check if the type is float
        # if not (int etc.), convert to float, this is to avoid numba errors
        x, y, z = (arr.astype(float)
                   if not np.issubdtype(arr[0], np.floating) else arr
                   for arr in (x, y, z))

        # call function to get area of all the faces as a np array
        self._face_areas = get_all_face_area_from_coords(
            x, y, z, face_nodes, nNodes_per_face, dim, quadrature_rule, order,
            coords_type)

        return self._face_areas

    def __eq__(self, other):
        """Two grids are equal if they have matching grid topology variables,
        coordinates, and dims all of which are equal.

        Parameters
        ----------
        other : uxarray.Grid
            The second grid object to be compared with `self`

        Returns
        -------
        If two grids are equal : bool
        """
        if other is not None:
            # Iterate over dict to set access attributes
            for key, value in self.grid_var_names.items():
                # Check if all grid variables are equal
                if self._ds.data_vars is not None:
                    if value in self._ds.data_vars:
                        if not self._ds[value].equals(
                                other._ds[other.grid_var_names[key]]):
                            return False
        else:
            return False

        return True

    def __ne__(self, other):
        """Two grids are not equal if they have differing grid topology
        variables, coordinates, or dims.

        Parameters
        ----------
        other : uxarray.Grid
            The second grid object to be compared with `self`

        Returns
        -------
        If two grids are not equal : bool
        """
        return not self.__eq__(other)

    # use the property keyword for declaration on face_areas property
    @property
    def face_areas(self):
        """Declare face_areas as a property."""
        # if self._face_areas is not None: it allows for using the cached result
        if self._face_areas is None:
            self.compute_face_areas()
        return self._face_areas

    # TODO: Make a decision on whether to provide Dataset- or DataArray-specific
    # functions from within Grid
    # def integrate(self, var_ds, quadrature_rule="triangular", order=4):
    #     """Integrates a xarray.Dataset over all the faces of the given mesh.
    #
    #     Parameters
    #     ----------
    #     var_ds : Xarray dataset, required
    #         Xarray dataset containing values to integrate on this grid
    #     quadrature_rule : str, optional
    #         Quadrature rule to use. Defaults to "triangular".
    #     order : int, optional
    #         Order of quadrature rule. Defaults to 4.
    #
    #     Returns
    #     -------
    #     Calculated integral : float
    #
    #     Examples
    #     --------
    #     Open grid file only
    #
    #     >>> xr_grid = xr.open_dataset("grid.ug")
    #     >>> grid = ux.Grid.(xr_grid)
    #     >>> var_ds = xr.open_dataset("centroid_pressure_data_ug")
    #
    #     # Compute the integral
    #     >>> integral_psi = grid.integrate(var_ds)
    #     """
    #     integral = 0.0
    #
    #     # call function to get area of all the faces as a np array
    #     face_areas = self.compute_face_areas(quadrature_rule, order)
    #
    #     var_key = list(var_ds.keys())
    #     if len(var_key) > 1:
    #         # warning: print message
    #         print(
    #             "WARNING: The xarray dataset file has more than one variable, using the first variable for integration"
    #         )
    #     var_key = var_key[0]
    #     face_vals = var_ds[var_key].to_numpy()
    #     integral = np.dot(face_areas, face_vals)
    #
    #     return integral

    def _build_edge_node_connectivity(self, repopulate=False):
        """Constructs the UGRID connectivity variable (``Mesh2_edge_nodes``)
        and stores it within the internal (``Grid._ds``) and through the
        attribute (``Grid.Mesh2_edge_nodes``).

        Additionally, the attributes (``inverse_indices``) and
        (``fill_value_mask``) are stored for constructing other
        connectivity variables.

        Parameters
        ----------
        repopulate : bool, optional
            Flag used to indicate if we want to overwrite the existed `Mesh2_edge_nodes` and generate a new
            inverse_indices, default is False
        """

        # need to derive edge nodes
        if "Mesh2_edge_nodes" not in self._ds or repopulate:
            padded_face_nodes = close_face_nodes(self.Mesh2_face_nodes.values,
                                                 self.nMesh2_face,
                                                 self.nMaxMesh2_face_nodes)

            # array of empty edge nodes where each entry is a pair of indices
            edge_nodes = np.empty(
                (self.nMesh2_face * self.nMaxMesh2_face_nodes, 2),
                dtype=INT_DTYPE)

            # first index includes starting node up to non-padded value
            edge_nodes[:, 0] = padded_face_nodes[:, :-1].ravel()

            # second index includes second node up to padded value
            edge_nodes[:, 1] = padded_face_nodes[:, 1:].ravel()
        else:
            # If "Mesh2_edge_nodes" already exists, directly return the function call
            return

        # sorted edge nodes
        edge_nodes.sort(axis=1)

        # unique edge nodes
        edge_nodes_unique, inverse_indices = np.unique(edge_nodes,
                                                       return_inverse=True,
                                                       axis=0)
        # find all edge nodes that contain a fill value
        fill_value_mask = np.logical_or(
            edge_nodes_unique[:, 0] == INT_FILL_VALUE,
            edge_nodes_unique[:, 1] == INT_FILL_VALUE)

        # all edge nodes that do not contain a fill value
        non_fill_value_mask = np.logical_not(fill_value_mask)
        edge_nodes_unique = edge_nodes_unique[non_fill_value_mask]

        # Update inverse_indices accordingly
        indices_to_update = np.where(fill_value_mask)[0]

        remove_mask = np.isin(inverse_indices, indices_to_update)
        inverse_indices[remove_mask] = INT_FILL_VALUE

        # Compute the indices where inverse_indices exceeds the values in indices_to_update
        indexes = np.searchsorted(indices_to_update,
                                  inverse_indices,
                                  side='right')
        # subtract the corresponding indexes from `inverse_indices`
        for i in range(len(inverse_indices)):
            if inverse_indices[i] != INT_FILL_VALUE:
                inverse_indices[i] -= indexes[i]

        # add Mesh2_edge_nodes to internal dataset
        self._ds['Mesh2_edge_nodes'] = xr.DataArray(
            edge_nodes_unique,
            dims=["nMesh2_edge", "Two"],
            attrs={
                "cf_role":
                    "edge_node_connectivity",
                "_FillValue":
                    INT_FILL_VALUE,
                "long_name":
                    "Maps every edge to the two nodes that it connects",
                "start_index":
                    INT_DTYPE(0),
                "inverse_indices":
                    inverse_indices,
                "fill_value_mask":
                    fill_value_mask
            })

    def _build_face_edges_connectivity(self):
        """Constructs the UGRID connectivity variable (``Mesh2_face_edges``)
        and stores it within the internal (``Grid._ds``) and through the
        attribute (``Grid.Mesh2_face_edges``)."""
        if ("Mesh2_edge_nodes" not in self._ds or
                "inverse_indices" not in self._ds['Mesh2_edge_nodes'].attrs):
            self._build_edge_node_connectivity(repopulate=True)

        inverse_indices = self._ds['Mesh2_edge_nodes'].inverse_indices
        inverse_indices = inverse_indices.reshape(self.nMesh2_face,
                                                  self.nMaxMesh2_face_nodes)

        self._ds["Mesh2_face_edges"] = xr.DataArray(
            data=inverse_indices,
            dims=["nMesh2_face", "nMaxMesh2_face_edges"],
            attrs={
                "cf_role":
                    "face_edges_connectivity",
                "start_index":
                    INT_DTYPE(0),
                "long_name":
                    "Maps every edge to the two nodes that it connects",
            })

    def _populate_cartesian_xyz_coord(self):
        """A helper function that populates the xyz attribute in
        UXarray.Grid._ds. This function is called when we need to use the
        cartesian coordinates for each node to do the calculation but the input
        data only has the "Mesh2_node_x" and "Mesh2_node_y" in degree.

        Note
        ----
        In the UXarray, we abide the UGRID convention and make sure the following attributes will always have its
        corresponding units as stated below:

        Mesh2_node_x
         unit:  "degree_east" for longitude
        Mesh2_node_y
         unit:  "degrees_north" for latitude
        Mesh2_node_z
         unit:  "m"
        Mesh2_node_cart_x
         unit:  "m"
        Mesh2_node_cart_y
         unit:  "m"
        Mesh2_node_cart_z
         unit:  "m"
        """

        # Check if the cartesian coordinates are already populated
        if "Mesh2_node_cart_x" in self._ds.keys():
            return

        # check for units and create Mesh2_node_cart_x/y/z set to self._ds
        nodes_lon_rad = np.deg2rad(self.Mesh2_node_x.values)
        nodes_lat_rad = np.deg2rad(self.Mesh2_node_y.values)
        nodes_rad = np.stack((nodes_lon_rad, nodes_lat_rad), axis=1)
        nodes_cart = np.asarray(
            list(map(node_lonlat_rad_to_xyz, list(nodes_rad))))

        self._ds["Mesh2_node_cart_x"] = xr.DataArray(
            data=nodes_cart[:, 0],
            dims=["nMesh2_node"],
            attrs={
                "standard_name": "cartesian x",
                "units": "m",
            })
        self._ds["Mesh2_node_cart_y"] = xr.DataArray(
            data=nodes_cart[:, 1],
            dims=["nMesh2_node"],
            attrs={
                "standard_name": "cartesian y",
                "units": "m",
            })
        self._ds["Mesh2_node_cart_z"] = xr.DataArray(
            data=nodes_cart[:, 2],
            dims=["nMesh2_node"],
            attrs={
                "standard_name": "cartesian z",
                "units": "m",
            })

    def _populate_lonlat_coord(self):
        """Helper function that populates the longitude and latitude and store
        it into the Mesh2_node_x and Mesh2_node_y. This is called when the
        input data has "Mesh2_node_x", "Mesh2_node_y", "Mesh2_node_z" in
        meters. Since we want "Mesh2_node_x" and "Mesh2_node_y" always have the
        "degree" units. For more details, please read the following.

        Raises
        ------
            RuntimeError
                Mesh2_node_x/y/z are not represented in the cartesian format with the unit 'm'/'meters' when calling this function"

        Note
        ----
        In the UXarray, we abide the UGRID convention and make sure the following attributes will always have its
        corresponding units as stated below:

        Mesh2_node_x
         unit:  "degree_east" for longitude
        Mesh2_node_y
         unit:  "degrees_north" for latitude
        Mesh2_node_z
         unit:  "m"
        Mesh2_node_cart_x
         unit:  "m"
        Mesh2_node_cart_y
         unit:  "m"
        Mesh2_node_cart_z
         unit:  "m"
        """

        # Check if the "Mesh2_node_x" is already in longitude
        if "degree" in self._ds.Mesh2_node_x.units:
            return

        # Check if the input Mesh2_node_xyz" are represented in the cartesian format with the unit "m"
        if ("m" not in self._ds.Mesh2_node_x.units) or ("m" not in self._ds.Mesh2_node_y.units) \
                or ("m" not in self._ds.Mesh2_node_z.units):
            raise RuntimeError(
                "Expected: Mesh2_node_x/y/z should be represented in the cartesian format with the "
                "unit 'm' when calling this function")

        # Put the cartesian coordinates inside the proper data structure
        self._ds["Mesh2_node_cart_x"] = xr.DataArray(
            data=self._ds["Mesh2_node_x"].values)
        self._ds["Mesh2_node_cart_y"] = xr.DataArray(
            data=self._ds["Mesh2_node_y"].values)
        self._ds["Mesh2_node_cart_z"] = xr.DataArray(
            data=self._ds["Mesh2_node_z"].values)

        # convert the input cartesian values into the longitude latitude degree
        nodes_cart = np.stack(
            (self._ds["Mesh2_node_x"].values, self._ds["Mesh2_node_y"].values,
             self._ds["Mesh2_node_z"].values),
            axis=1).tolist()
        nodes_rad = list(map(node_xyz_to_lonlat_rad, nodes_cart))
        nodes_degree = np.rad2deg(nodes_rad)
        self._ds["Mesh2_node_x"] = xr.DataArray(
            data=nodes_degree[:, 0],
            dims=["nMesh2_node"],
            attrs={
                "standard_name": "longitude",
                "long_name": "longitude of mesh nodes",
                "units": "degrees_east",
            })
        self._ds["Mesh2_node_y"] = xr.DataArray(
            data=nodes_degree[:, 1],
            dims=["nMesh2_node"],
            attrs={
                "standard_name": "lattitude",
                "long_name": "latitude of mesh nodes",
                "units": "degrees_north",
            })

    def _build_nNodes_per_face(self):
        """Constructs ``nNodes_per_face``, which contains the number of non-
        fill-value nodes for each face in ``Mesh2_face_nodes``"""

        # padding to shape [nMesh2_face, nMaxMesh2_face_nodes + 1]
        closed = np.ones((self.nMesh2_face, self.nMaxMesh2_face_nodes + 1),
                         dtype=INT_DTYPE) * INT_FILL_VALUE

        closed[:, :-1] = self.Mesh2_face_nodes.copy()

        nNodes_per_face = np.argmax(closed == INT_FILL_VALUE, axis=1)

        # add to internal dataset
        self._ds["nNodes_per_face"] = xr.DataArray(
            data=nNodes_per_face,
            dims=["nMesh2_face"],
            attrs={"long_name": "number of non-fill value nodes for each face"})
