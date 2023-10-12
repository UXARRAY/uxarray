"""uxarray.core.grid module."""
import xarray as xr
import numpy as np

from typing import Any, Dict, Optional, Union

# reader and writer imports
from uxarray.io._exodus import _read_exodus, _encode_exodus
from uxarray.io._mpas import _read_mpas
from uxarray.io._ugrid import _read_ugrid, _encode_ugrid, _validate_minimum_ugrid
from uxarray.io._shapefile import _read_shpfile
from uxarray.io._scrip import _read_scrip, _encode_scrip
from uxarray.io._vertices import _read_face_vertices

from uxarray.io.utils import _parse_grid_type
from uxarray.grid.area import get_all_face_area_from_coords
from uxarray.grid.coordinates import _populate_centroid_coord
from uxarray.grid.connectivity import (
    _build_edge_node_connectivity,
    _build_face_edges_connectivity,
    _build_nNodes_per_face,
    _build_node_faces_connectivity,
)

from uxarray.grid.coordinates import (_populate_lonlat_coord,
                                      _populate_cartesian_xyz_coord)

from uxarray.grid.geometry import (_build_antimeridian_face_indices,
                                   _grid_to_polygon_geodataframe,
                                   _grid_to_matplotlib_polycollection,
                                   _grid_to_matplotlib_linecollection,
                                   _grid_to_polygons)

from uxarray.grid.neighbors import BallTree

from uxarray.plot.accessor import GridPlotAccessor

from xarray.core.utils import UncachedAccessor

from warnings import warn


class Grid:
    """Represents a two-dimensional unstructured grid encoded following the
    UGRID conventions and provides grid-specific functionality.

    Can be used standalone to work with unstructured grids, or can be paired with either a ``ux.UxDataArray`` or
    ``ux.UxDataset`` and accessed through the ``.uxgrid`` attribute.

    For constructing a grid from non-UGRID datasets or other types of supported data, see our ``ux.open_grid`` method or
    specific class methods (``Grid.from_dataset``, ``Grid.from_face_verticies``, etc.)


    Parameters
    ----------
    grid_ds : xr.Dataset
        ``xarray.Dataset`` encoded in the UGRID conventions

    source_grid_spec : str, default="UGRID"
        Original unstructured grid format (i.e. UGRID, MPAS, etc.)

    source_dims_dict : dict, default={}
        Mapping of dimensions from the source dataset to their UGRID equivalent (i.e. {nCell : nMesh2_face})

    Examples
    ----------

    >>> import uxarray as ux
    >>> grid_path = "/path/to/grid.nc"
    >>> data_path = "/path/to/data.nc"

    1. Open a grid file with `uxarray.open_grid()`:

    >>> uxgrid = ux.open_grid(grid_path)

    2. Open an unstructured grid dataset file with
    `uxarray.open_dataset()`, then access the ``Grid``.:

    >>> uxds = ux.open_dataset(grid_path, data_path)
    >>> uxds.uxgrid
    """

    def __init__(self,
                 grid_ds: xr.Dataset,
                 source_grid_spec: Optional[str] = None,
                 source_dims_dict: Optional[dict] = {}):

        # check if inputted dataset is a minimum representable 2D UGRID unstructured grid
        if not _validate_minimum_ugrid(grid_ds):
            raise ValueError(
                "Direct use of Grid constructor requires grid_ds to follow the internal unstructured grid definition, "
                "including variable and dimension names. This grid_ds does not satisfy those requirements. If you are "
                "not sure about how to do that, using ux.open_grid() or ux.from_dataset() is suggested."
            )  # TODO: elaborate once we have a formal definition

        # grid spec not provided, check if grid_ds is a minimum representable UGRID dataset
        if source_grid_spec is None:
            warn(
                "Attempting to construct a Grid without passing in source_grid_spec. Direct use of Grid constructor"
                "is only advised if grid_ds is following the internal unstructured grid definition, including"
                "variable and dimension names. Using ux.open_grid() or ux.from_dataset() is suggested.",
                Warning)
            # TODO: more checks for validate grid (lat/lon coords, etc)

        # mapping of ugrid dimensions and variables to source dataset's conventions
        self._source_dims_dict = source_dims_dict

        # source grid specification (i.e. UGRID, MPAS, SCRIP, etc.)
        self.source_grid_spec = source_grid_spec

        # internal xarray dataset for storing grid variables
        self._ds = grid_ds

        # initialize attributes
        self._antimeridian_face_indices = None
        self._face_areas = None

        # initialize cached data structures (visualization)
        self._gdf = None
        self._poly_collection = None
        self._line_collection = None
        self._centroid_points_df_proj = [None, None]
        self._corner_points_df_proj = [None, None]
        self._raster_data_id = None

        # initialize cached data structures (nearest neighbor operations)
        self._ball_tree = None

        self._mesh2_warning_raised = False

    # declare plotting accessor
    plot = UncachedAccessor(GridPlotAccessor)

    def _mesh2_future_warning(self):
        """Raises a FutureWarning about the 'Mesh2' prefix removal.

        Only raises the warning once when a effected property is called.
        """
        if not self._mesh2_warning_raised:
            self._mesh2_warning_raised = True
            warn(
                "'Mesh2' prefix used in dimension, coordinate, and connectivity attributes (i.e. Mesh2_face_nodes) will"
                " be dropped in a future release.", FutureWarning, 1)

    @classmethod
    def from_dataset(cls,
                     dataset: xr.Dataset,
                     use_dual: Optional[bool] = False):
        """Constructs a ``Grid`` object from an ``xarray.Dataset``.

        Parameters
        ----------
        dataset : xr.Dataset
            ``xarray.Dataset`` containing unstructured grid coordinates and connectivity variables
        use_dual : bool, default=False
            When reading in MPAS formatted datasets, indicates whether to use the Dual Mesh
        """
        if not isinstance(dataset, xr.Dataset):
            raise ValueError("Input must be an xarray.Dataset")

        # determine grid/mesh specification
        source_grid_spec = _parse_grid_type(dataset)

        if source_grid_spec == "Exodus":
            grid_ds, source_dims_dict = _read_exodus(dataset)
        elif source_grid_spec == "Scrip":
            grid_ds, source_dims_dict = _read_scrip(dataset)
        elif source_grid_spec == "UGRID":
            grid_ds, source_dims_dict = _read_ugrid(dataset)
        elif source_grid_spec == "MPAS":
            grid_ds, source_dims_dict = _read_mpas(dataset, use_dual=use_dual)
        elif source_grid_spec == "Shapefile":
            raise ValueError("Shapefiles not yet supported")
        else:
            raise ValueError("Unsupported Grid Format")

        return cls(grid_ds, source_grid_spec, source_dims_dict)

    @classmethod
    def from_face_vertices(cls,
                           face_vertices: Union[list, tuple, np.ndarray],
                           latlon: Optional[bool] = True):
        """Constructs a ``Grid`` object from user-defined face vertices.

        Parameters
        ----------
        face_vertices : list, tuple, np.ndarray
            array-like input containing the face vertices to construct the grid from
        latlon : bool, default=True
            Indicates whether the inputted vertices are in lat/lon, with units in degrees
        """
        if not isinstance(face_vertices, (list, tuple, np.ndarray)):
            raise ValueError(
                "Input must be either a list, tuple, or np.ndarray")

        face_vertices = np.asarray(face_vertices)

        if face_vertices.ndim == 3:
            grid_ds = _read_face_vertices(face_vertices, latlon)

        elif face_vertices.ndim == 2:
            grid_ds = _read_face_vertices(np.array([face_vertices]), latlon)

        else:
            raise RuntimeError(
                f"Invalid Input Dimension: {face_vertices.ndim}. Expected dimension should be "
                f"3: [nMesh2_face, nMesh2_node, Two/Three] or 2 when only "
                f"one face is passed in.")

        return cls(grid_ds, source_grid_spec="Face Vertices")

    def __repr__(self):
        """Constructs a string representation of the contents of a ``Grid``."""
        prefix = "<uxarray.Grid>\n"
        original_grid_str = f"Original Grid Type: {self.source_grid_spec}\n"
        dims_heading = "Grid Dimensions:\n"
        dims_str = ""
        # if self.grid_var_names["Mesh2_node_x"] in self._ds:
        #     dims_str += f"  * nMesh2_node: {self.nMesh2_node}\n"
        # if self.grid_var_names["Mesh2_face_nodes"] in self._ds:
        #     dims_str += f"  * nMesh2_face: {self.nMesh2_face}\n"
        #     dims_str += f"  * nMesh2_face: {self.nMesh2_face}\n"

        for key, value in zip(self._ds.dims.keys(), self._ds.dims.values()):
            dims_str += f"  * {key}: {value}\n"
            # if key in self._inverse_grid_var_names:
            #     dims_str += f"  * {self._inverse_grid_var_names[key]}: {value}\n"

        if "nMesh2_edge" in self._ds.dims:
            dims_str += f"  * nMesh2_edge: {self.nMesh2_edge}\n"

        if "nMaxMesh2_face_edges" in self._ds.dims:
            dims_str += f"  * nMaxMesh2_face_edges: {self.nMaxMesh2_face_edges}\n"

        coord_heading = "Grid Coordinate Variables:\n"
        coords_str = ""
        if "Mesh2_node_x" in self._ds:
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
        if "Mesh2_face_nodes" in self._ds:
            connectivity_str += f"  * Mesh2_face_nodes: {self.Mesh2_face_nodes.shape}\n"
        if "Mesh2_edge_nodes" in self._ds:
            connectivity_str += f"  * Mesh2_edge_nodes: {self.Mesh2_edge_nodes.shape}\n"
        if "Mesh2_face_edges" in self._ds:
            connectivity_str += f"  * Mesh2_face_edges: {self.Mesh2_face_edges.shape}\n"
        connectivity_str += f"  * nNodes_per_face: {self.nNodes_per_face.shape}\n"

        return prefix + original_grid_str + dims_heading + dims_str + coord_heading + coords_str + \
            connectivity_heading + connectivity_str

    def __eq__(self, other) -> bool:
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

        if not isinstance(other, Grid):
            return False

        if self.source_grid_spec != other.source_grid_spec:
            return False

        if not (self.Mesh2_node_x.equals(other.Mesh2_node_x) or
                self.Mesh2_node_y.equals(other.Mesh2_node_y)):
            return False

        if not self.Mesh2_face_nodes.equals(other.Mesh2_face_nodes):
            return False

        return True

    def __ne__(self, other) -> bool:
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

    @property
    def parsed_attrs(self) -> dict:
        """Dictionary of parsed attributes from the source grid."""
        return self._ds.attrs

    @property
    def Mesh2(self) -> xr.DataArray:
        """UGRID Attribute ``Mesh2``, which indicates the topology data of a 2D
        unstructured mesh."""
        self._mesh2_future_warning()
        return self._ds["Mesh2"]

    @property
    def nMesh2_node(self) -> int:
        """UGRID Dimension ``nMesh2_node``, which represents the total number
        of nodes."""
        self._mesh2_future_warning()
        return self._ds.dims["nMesh2_node"]

    @property
    def nMesh2_face(self) -> int:
        """UGRID Dimension ``nMesh2_face``, which represents the total number
        of faces."""
        self._mesh2_future_warning()
        return self._ds["Mesh2_face_nodes"].shape[0]

    @property
    def nMesh2_edge(self) -> int:
        """UGRID Dimension ``nMesh2_edge``, which represents the total number
        of edges."""
        self._mesh2_future_warning()

        if "Mesh2_edge_nodes" not in self._ds:
            _build_edge_node_connectivity(self, repopulate=True)

        return self._ds['Mesh2_edge_nodes'].shape[0]

    @property
    def nMaxMesh2_face_nodes(self) -> int:
        """UGRID Dimension ``nMaxMesh2_face_nodes``, which represents the
        maximum number of nodes that a face may contain."""
        self._mesh2_future_warning()
        return self.Mesh2_face_nodes.shape[1]

    @property
    def nMaxMesh2_face_edges(self) -> xr.DataArray:
        """Dimension ``nMaxMesh2_face_edges``, which represents the maximum
        number of edges per face.

        Equivalent to ``nMaxMesh2_face_nodes``
        """
        self._mesh2_future_warning()
        if "Mesh2_face_edges" not in self._ds:
            _build_face_edges_connectivity(self)

        return self._ds["Mesh2_face_edges"].shape[1]

    @property
    def nNodes_per_face(self) -> xr.DataArray:
        """Dimension Variable ``nNodes_per_face``, which contains the number of
        non-fill-value nodes per face.

        Dimensions (``nMesh2_nodes``) and DataType ``INT_DTYPE``.
        """
        self._mesh2_future_warning()
        if "nNodes_per_face" not in self._ds:
            _build_nNodes_per_face(self)
        return self._ds["nNodes_per_face"]

    @property
    def Mesh2_node_x(self) -> xr.DataArray:
        """UGRID Coordinate Variable ``Mesh2_node_x``, which contains the
        longitude of each node in degrees.

        Dimensions (``nMesh2_node``)
        """
        self._mesh2_future_warning()
        if "Mesh2_node_x" not in self._ds:
            _populate_lonlat_coord(self)
        return self._ds["Mesh2_node_x"]

    @property
    def Mesh2_node_cart_x(self) -> xr.DataArray:
        """Coordinate Variable ``Mesh2_node_cart_x``, which contains the x
        location in meters.

        Dimensions (``nMesh2_node``)
        """
        self._mesh2_future_warning()
        if "Mesh2_node_cart_x" not in self._ds:
            _populate_cartesian_xyz_coord(self)

        return self._ds['Mesh2_node_cart_x']

    @property
    def Mesh2_face_x(self) -> xr.DataArray:
        """UGRID Coordinate Variable ``Mesh2_face_x``, which contains the
        longitude of each face center.

        Dimensions (``nMesh2_face``)
        """

        self._mesh2_future_warning()
        if "Mesh2_face_x" not in self._ds:
            _populate_centroid_coord(self)
        return self._ds['Mesh2_face_x']

    @property
    def Mesh2_node_y(self) -> xr.DataArray:
        """UGRID Coordinate Variable ``Mesh2_node_y``, which contains the
        latitude of each node.

        Dimensions (``nMesh2_node``)
        """
        self._mesh2_future_warning()
        if "Mesh2_node_y" not in self._ds:
            _populate_lonlat_coord(self)

        return self._ds["Mesh2_node_y"]

    @property
    def Mesh2_node_cart_y(self) -> xr.DataArray:
        """Coordinate Variable ``Mesh2_node_cart_y``, which contains the y
        location in meters.

        Dimensions (``nMesh2_node``)
        """
        self._mesh2_future_warning()
        if "Mesh2_node_cart_y" not in self._ds:
            _populate_cartesian_xyz_coord(self)
        return self._ds['Mesh2_node_cart_y']

    @property
    def Mesh2_face_y(self) -> xr.DataArray:
        """UGRID Coordinate Variable ``Mesh2_face_y``, which contains the
        latitude of each face center.

        Dimensions (``nMesh2_face``)
        """
        self._mesh2_future_warning()

        if "Mesh2_face_y" not in self._ds:
            _populate_centroid_coord(self)
        return self._ds['Mesh2_face_y']

    @property
    def Mesh2_face_cart_x(self) -> xr.DataArray:
        """Coordinate ``Mesh2_face_cart_x``, which contains the Cartesian x
        location of each face center in meters.

        Dimensions (``nMesh2_face``)
        """
        if "Mesh2_face_cart_x" not in self._ds:
            _populate_centroid_coord(self)
        return self._ds["Mesh2_face_cart_x"]

    @property
    def Mesh2_face_cart_y(self) -> xr.DataArray:
        """Coordinate ``Mesh2_face_cart_y``, which contains the Cartesian y
        location of each face center in meters.

        Dimensions (``nMesh2_face``)
        """
        if "Mesh2_face_cart_y" not in self._ds:
            _populate_centroid_coord(self)
        return self._ds["Mesh2_face_cart_y"]

    @property
    def Mesh2_face_cart_z(self) -> xr.DataArray:
        """Coordinate ``Mesh2_face_cart_z``, which contains the Cartesian z
        location of each face center in meters.

        Dimensions (``nMesh2_face``)
        """
        if "Mesh2_face_cart_z" not in self._ds:
            _populate_centroid_coord(self)
        return self._ds["Mesh2_face_cart_z"]

    @property
    def Mesh2_node_cart_z(self) -> xr.DataArray:
        """Coordinate Variable ``Mesh2_node_cart_z``, which contains the z
        location in meters.

        Dimensions (``nMesh2_node``)
        """
        self._mesh2_future_warning()
        if "Mesh2_node_cart_z" not in self._ds:
            self._populate_cartesian_xyz_coord()
        return self._ds['Mesh2_node_cart_z']

    @property
    def Mesh2_face_nodes(self) -> xr.DataArray:
        """UGRID Connectivity Variable ``Mesh2_face_nodes``, which maps each
        face to its corner nodes.

        Dimensions (``nMesh2_face``, ``nMaxMesh2_face_nodes``) and
        DataType ``INT_DTYPE``.

        Faces can have arbitrary length, with _FillValue=-1 used when faces
        have fewer nodes than MaxNumNodesPerFace.

        Nodes are in counter-clockwise order.
        """
        self._mesh2_future_warning()
        return self._ds["Mesh2_face_nodes"]

    @property
    def Mesh2_edge_nodes(self) -> xr.DataArray:
        """UGRID Connectivity Variable ``Mesh2_edge_nodes``, which maps every
        edge to the two nodes that it connects.

        Dimensions (``nMesh2_edge``, ``Two``) and DataType
        ``INT_DTYPE``.

        Nodes are in arbitrary order.
        """
        self._mesh2_future_warning()
        if "Mesh2_edge_nodes" not in self._ds:
            _build_edge_node_connectivity(self)

        return self._ds['Mesh2_edge_nodes']

    @property
    def Mesh2_face_edges(self) -> xr.DataArray:
        """UGRID Connectivity Variable ``Mesh2_face_edges``, which maps every
        face to its edges.

        Dimensions (``nMesh2_face``, ``nMaxMesh2_face_nodes``) and
        DataType ``INT_DTYPE``.
        """
        self._mesh2_future_warning()
        if "Mesh2_face_edges" not in self._ds:
            _build_face_edges_connectivity(self)

        return self._ds["Mesh2_face_edges"]

    @property
    def Mesh2_node_faces(self) -> xr.DataArray:
        """UGRID Connectivity Variable ``Mesh2_node_faces``, which maps every
        node to its faces.

        Dimensions (``nMesh2_node``, ``nMaxNumFacesPerNode``) and
        DataType ``INT_DTYPE``.
        """
        self._mesh2_future_warning()
        if "Mesh2_node_faces" not in self._ds:
            _build_node_faces_connectivity(self)

        return self._ds["Mesh2_node_faces"]

    # other properties
    @property
    def antimeridian_face_indices(self) -> np.ndarray:
        """Index of each face that crosses the antimeridian."""
        if self._antimeridian_face_indices is None:
            self._antimeridian_face_indices = _build_antimeridian_face_indices(
                self)
        return self._antimeridian_face_indices

    @property
    def face_areas(self) -> np.ndarray:
        """Declare face_areas as a property."""
        # if self._face_areas is not None: it allows for using the cached result
        if self._face_areas is None:
            self.compute_face_areas()
        return self._face_areas

    def get_ball_tree(self, tree_type: Optional[str] = "nodes"):
        """Get the BallTree data structure of this Grid that allows for nearest
        neighbor queries (k nearest or within some radius) on either the nodes
        (``Mesh2_node_x``, ``Mesh2_node_y``) or face centers (``Mesh2_face_x``,
        ``Mesh2_face_y``).

        Parameters
        ----------
        tree_type : str, default="nodes"
            Selects which tree to query, with "nodes" selecting the Corner Nodes and "face centers" selecting the Face
            Centers of each face

        Returns
        -------
        self._ball_tree : grid.Neighbors.BallTree
            BallTree instance
        """
        if self._ball_tree is None:
            self._ball_tree = BallTree(self,
                                       tree_type=tree_type,
                                       distance_metric='haversine')
        else:
            if tree_type != self._ball_tree._tree_type:
                self._ball_tree.tree_type = tree_type

        return self._ball_tree

    def copy(self):
        """Returns a deep copy of this grid."""

        return Grid(self._ds,
                    source_grid_spec=self.source_grid_spec,
                    source_dims_dict=self._source_dims_dict)

    def encode_as(self, grid_type: str) -> xr.Dataset:
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

        if grid_type == "UGRID":
            out_ds = _encode_ugrid(self._ds)

        elif grid_type == "Exodus":
            out_ds = _encode_exodus(self._ds)

        elif grid_type == "SCRIP":
            out_ds = _encode_scrip(self.Mesh2_face_nodes, self.Mesh2_node_x,
                                   self.Mesh2_node_y, self.face_areas)
        else:
            raise RuntimeError("The grid type not supported: ", grid_type)

        return out_ds

    def calculate_total_face_area(self,
                                  quadrature_rule: Optional[str] = "triangular",
                                  order: Optional[int] = 4) -> float:
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

    def compute_face_areas(self,
                           quadrature_rule: Optional[str] = "triangular",
                           order: Optional[int] = 4,
                           latlon: Optional[bool] = True):
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

        if latlon:
            x = self.Mesh2_node_x.data
            y = self.Mesh2_node_y.data
            z = np.zeros((self.nMesh2_node))
            coords_type = "spherical"  # TODO: should really be called latlon?
        else:
            x = self.Mesh2_node_cart_x.data
            y = self.Mesh2_node_cart_y.data
            z = self.Mesh2_node_cart_z.data
            coords_type = "cartesian"

        # TODO: we dont really need this, but keep for now
        dim = self.Mesh2.attrs['topology_dimension']

        nNodes_per_face = self.nNodes_per_face.data
        face_nodes = self.Mesh2_face_nodes.data

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

    def to_geodataframe(self,
                        override: Optional[bool] = False,
                        cache: Optional[bool] = True,
                        correct_antimeridian_polygons: Optional[bool] = True):
        """Constructs a ``spatialpandas.GeoDataFrame`` with a "geometry"
        column, containing a collection of Shapely Polygons or MultiPolygons
        representing the geometry of the unstructured grid. Additionally, any
        polygon that crosses the antimeridian is split into MultiPolygons.

        Parameters
        ----------
        override : bool
            Flag to recompute the ``GeoDataFrame`` if one is already cached
        cache : bool
            Flag to indicate if the computed ``GeoDataFrame`` should be cached
        correct_antimeridian_polygons: bool, Optional
            Parameter to select whether to correct and split antimeridian polygons

        Returns
        -------
        gdf : spatialpandas.GeoDataFrame
            The output `GeoDataFrame` with a filled out "geometry" collumn
        """

        # use cached geodataframe
        if self._gdf is not None and not override:
            return self._gdf

        # construct a geodataframe with the faces stored as polygons as the geometry
        gdf = _grid_to_polygon_geodataframe(self, correct_antimeridian_polygons)

        # cache computed geodataframe
        if cache:
            self._gdf = gdf

        return gdf

    def to_polycollection(self,
                          override: Optional[bool] = False,
                          cache: Optional[bool] = True,
                          correct_antimeridian_polygons: Optional[bool] = True):
        """Constructs a ``matplotlib.collections.PolyCollection`` object with
        polygons representing the geometry of the unstructured grid, with
        polygons that cross the antimeridian split.

        Parameters
        ----------
        override : bool
            Flag to recompute the ``PolyCollection`` if one is already cached
        cache : bool
            Flag to indicate if the computed ``PolyCollection`` should be cached

        Returns
        -------
        polycollection : matplotlib.collections.PolyCollection
            The output `PolyCollection` containing faces represented as polygons
        corrected_to_original_faces: list
            Original indices used to map the corrected polygon shells to their entries in face nodes
        """

        # use cached polycollection
        if self._poly_collection is not None and not override:
            return self._poly_collection

        poly_collection, corrected_to_original_faces = _grid_to_matplotlib_polycollection(
            self)

        # cache computed polycollection
        if cache:
            self._poly_collection = poly_collection

        return poly_collection, corrected_to_original_faces

    def to_linecollection(self,
                          override: Optional[bool] = False,
                          cache: Optional[bool] = True):
        """Constructs a ``matplotlib.collections.LineCollection`` object with
        line segments representing the geometry of the unstructured grid,
        corrected near the antimeridian.

        Parameters
        ----------
        override : bool
            Flag to recompute the ``LineCollection`` if one is already cached
        cache : bool
            Flag to indicate if the computed ``LineCollection`` should be cached

        Returns
        -------
        line_collection : matplotlib.collections.LineCollection
            The output `LineCollection` containing faces represented as polygons
        """

        # use cached line collection
        if self._line_collection is not None and not override:
            return self._line_collection

        line_collection = _grid_to_matplotlib_linecollection(self)

        # cache computed line collection
        if cache:
            self._line_collection = line_collection

        return line_collection

    def to_shapely_polygons(self,
                            correct_antimeridian_polygons: Optional[bool] = True
                           ):
        """Constructs an array of Shapely Polygons representing each face, with
        antimeridian polygons split according to the GeoJSON standards.

         Parameters
        ----------
        correct_antimeridian_polygons: bool, Optional
            Parameter to select whether to correct and split antimeridian polygons

        Returns
        -------
        polygons : np.ndarray
            Array containing Shapely Polygons
        """
        polygons = _grid_to_polygons(self, correct_antimeridian_polygons)
        return polygons
