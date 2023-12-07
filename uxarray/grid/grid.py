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
from uxarray.grid.coordinates import (_populate_centroid_coord,
                                      _set_desired_longitude_range)
from uxarray.grid.connectivity import (_populate_edge_node_connectivity,
                                       _populate_face_edge_connectivity,
                                       _populate_n_nodes_per_face,
                                       _populate_node_face_connectivity,
                                       _populate_edge_face_connectivity)

from uxarray.grid.coordinates import (_populate_lonlat_coord,
                                      _populate_cartesian_xyz_coord)

from uxarray.grid.geometry import (_populate_antimeridian_face_indices,
                                   _grid_to_polygon_geodataframe,
                                   _grid_to_matplotlib_polycollection,
                                   _grid_to_matplotlib_linecollection)

from uxarray.grid.neighbors import BallTree, KDTree

from uxarray.plot.accessor import GridPlotAccessor

from uxarray.grid.validation import _check_connectivity, _check_duplicate_nodes, _check_area

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
        Mapping of dimensions from the source dataset to their UGRID equivalent (i.e. {nCell : n_face})

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
        self._gdf_exclude_am = None
        self._poly_collection = None
        self._line_collection = None
        self._centroid_points_df_proj = [None, None]
        self._corner_points_df_proj = [None, None]
        self._raster_data_id = None

        # initialize cached data structures (nearest neighbor operations)
        self._ball_tree = None
        self._kd_tree = None

        # set desired longitude range to [-180, 180]
        _set_desired_longitude_range(self._ds)

    # declare plotting accessor
    plot = UncachedAccessor(GridPlotAccessor)

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
                f"3: [n_face, n_node, two/three] or 2 when only "
                f"one face is passed in.")

        return cls(grid_ds, source_grid_spec="Face Vertices")

    def validate(self):
        """Validate a grid object check for common errors, such as:

            - Duplicate nodes
            - Connectivity
            - Face areas (non zero)
        Raises
        ------
        RuntimeError
            If unsupported grid type provided
        """
        # If the mesh file is loaded correctly, we have the underlying file format as UGRID
        # Test if the file is a valid ugrid file format or not
        print("Validating the mesh...")

        # call the check_connectivity and check_duplicate_nodes functions from validation.py
        checkDN = _check_duplicate_nodes(self)
        check_C = _check_connectivity(self)
        check_A = _check_area(self)

        if checkDN and check_C and check_A:
            print("Mesh validation successful.")
            return True
        else:
            raise RuntimeError("Mesh validation failed.")

    def __repr__(self):
        """Constructs a string representation of the contents of a ``Grid``."""
        prefix = "<uxarray.Grid>\n"
        original_grid_str = f"Original Grid Type: {self.source_grid_spec}\n"
        dims_heading = "Grid Dimensions:\n"
        dims_str = ""

        for key, value in zip(self._ds.dims.keys(), self._ds.dims.values()):
            dims_str += f"  * {key}: {value}\n"

        dims_str += f"  * n_nodes_per_face: {self.n_nodes_per_face.shape}\n"

        coord_heading = "Grid Coordinates (Spherical):\n"
        coords_str = ""
        if "node_lon" in self._ds:
            coords_str += f"  * node_lon: {self.node_lon.shape}\n"
            coords_str += f"  * node_lat: {self.node_lat.shape}\n"
        if "edge_lon" in self._ds:
            coords_str += f"  * edge_lon: {self.edge_lon.shape}\n"
            coords_str += f"  * edge_lat: {self.edge_lat.shape}\n"
        if "face_lon" in self._ds:
            coords_str += f"  * face_lon: {self.face_lon.shape}\n"
            coords_str += f"  * face_lat: {self.face_lat.shape}\n"

        coords_str += "Grid Coordinates (Cartesian):\n"
        if "node_x" in self._ds:
            coords_str += f"  * node_x: {self.node_x.shape}\n"
            coords_str += f"  * node_y: {self.node_y.shape}\n"
            coords_str += f"  * node_z: {self.node_z.shape}\n"
        if "edge_x" in self._ds:
            coords_str += f"  * edge_x: {self.edge_x.shape}\n"
            coords_str += f"  * edge_y: {self.edge_y.shape}\n"
            coords_str += f"  * edge_z: {self.edge_z.shape}\n"
        if "face_x" in self._ds:
            coords_str += f"  * face_x: {self.face_x.shape}\n"
            coords_str += f"  * face_y: {self.face_y.shape}\n"
            coords_str += f"  * face_z: {self.face_z.shape}\n"

        connectivity_heading = "Grid Connectivity Variables:\n"
        connectivity_str = ""
        if "face_node_connectivity" in self._ds:
            connectivity_str += f"  * face_node_connectivity: {self.face_node_connectivity.shape}\n"

        if "edge_node_connectivity" in self._ds:
            connectivity_str += f"  * edge_node_connectivity: {self.edge_node_connectivity.shape}\n"

        if "node_node_connectivity" in self._ds:
            connectivity_str += f"  * node_node_connectivity: {self.node_node_connectivity.shape}\n"

        if "face_edge_connectivity" in self._ds:
            connectivity_str += f"  * face_edge_connectivity: {self.face_edge_connectivity.shape}\n"

        if "edge_edge_connectivity" in self._ds:
            connectivity_str += f"  * edge_edge_connectivity: {self.edge_edge_connectivity.shape}\n"

        if "node_edge_connectivity" in self._ds:
            connectivity_str += f"  * node_edge_connectivity: {self.node_edge_connectivity.shape}\n"

        if "face_face_connectivity" in self._ds:
            connectivity_str += f"  * face_face_connectivity: {self.face_face_connectivity.shape}\n"

        if "edge_face_connectivity" in self._ds:
            connectivity_str += f"  * edge_face_connectivity: {self.edge_face_connectivity.shape}\n"

        if "node_face_connectivity" in self._ds:
            connectivity_str += f"  * node_face_connectivity: {self.node_face_connectivity.shape}\n"

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

        if not (self.node_lon.equals(other.node_lon) or
                self.node_lat.equals(other.node_lat)):
            return False

        if not self.face_node_connectivity.equals(other.face_node_connectivity):
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
        unstructured mesh.

        Internal use has been deprecated.
        """
        return self._ds["Mesh2"]

    # ==================================================================================================================
    # Grid Dimensions
    @property
    def n_node(self) -> int:
        """Dimension ``n_node``, which represents the total number of unique
        corner nodes."""
        return self._ds.dims["n_node"]

    @property
    def n_face(self) -> int:
        """Dimension ``n_face``, which represents the total number of unique
        faces."""
        return self._ds.dims["n_face"]

    @property
    def n_edge(self) -> int:
        """Dimension ``n_edge``, which represents the total number of unique
        edges."""
        if "edge_node_connectivity" not in self._ds:
            _populate_edge_node_connectivity(self)

        return self._ds.dims["n_edge"]

    # ==================================================================================================================
    @property
    def n_max_face_nodes(self) -> int:
        """Dimension ``n_max_face_nodes``, which represents the maximum number
        of nodes that a face may contain."""
        return self.face_node_connectivity.shape[1]

    @property
    def n_max_face_edges(self) -> xr.DataArray:
        """Dimension ``n_max_face_edges``, which represents the maximum number
        of edges per face.

        Equivalent to ``n_max_face_nodes``
        """
        if "n_max_face_edges" not in self._ds:
            _populate_face_edge_connectivity(self)

        return self._ds["face_edge_connectivity"].shape[1]

    @property
    def n_nodes_per_face(self) -> xr.DataArray:
        """Dimension Variable ``n_nodes_per_face``, which contains the number
        of non-fill-value nodes per face.

        Dimensions (``n_node``) and DataType ``INT_DTYPE``.
        """
        if "n_nodes_per_face" not in self._ds:
            _populate_n_nodes_per_face(self)
        return self._ds["n_nodes_per_face"]

    # ==================================================================================================================
    # Spherical Node Coordinates
    @property
    def node_lon(self) -> xr.DataArray:
        """Coordinate ``node_lon``, which contains the longitude of each node
        in degrees.

        Dimensions (``n_node``)
        """
        if "node_lon" not in self._ds:
            _set_desired_longitude_range(self._ds)
            _populate_lonlat_coord(self)
        return self._ds["node_lon"]

    @property
    def node_lat(self) -> xr.DataArray:
        """Coordinate ``node_lat``, which contains the latitude of each node in
        degrees.

        Dimensions (``n_node``)
        """
        if "node_lat" not in self._ds:
            _set_desired_longitude_range(self._ds)
            _populate_lonlat_coord(self)
        return self._ds["node_lat"]

    # ==================================================================================================================
    # Cartesian Node Coordinates
    @property
    def node_x(self) -> xr.DataArray:
        """Coordinate ``node_x``, which contains the Cartesian x location of
        each node in meters.

        Dimensions (``n_node``)
        """
        if "node_x" not in self._ds:
            _populate_cartesian_xyz_coord(self)

        return self._ds['node_x']

    @property
    def node_y(self) -> xr.DataArray:
        """Coordinate ``node_y``, which contains the Cartesian y location of
        each node in meters.

        Dimensions (``n_node``)
        """
        if "node_y" not in self._ds:
            _populate_cartesian_xyz_coord(self)
        return self._ds['node_y']

    @property
    def node_z(self) -> xr.DataArray:
        """Coordinate ``node_z``, which contains the Cartesian y location of
        each node in meters.

        Dimensions (``n_node``)
        """
        if "node_z" not in self._ds:
            _populate_cartesian_xyz_coord(self)
        return self._ds['node_z']

    # ==================================================================================================================
    # Spherical Edge Coordinates
    @property
    def edge_lon(self) -> xr.DataArray:
        """Coordinate ``edge_lon``, which contains the longitude of each edge
        in degrees.

        Dimensions (``n_edge``)
        """
        if "edge_lon" not in self._ds:
            return None
        # temp until we construct edge lon
        _set_desired_longitude_range(self._ds)
        return self._ds["edge_lon"]

    @property
    def edge_lat(self) -> xr.DataArray:
        """Coordinate ``edge_lat``, which contains the latitude of each edge in
        degrees.

        Dimensions (``n_edge``)
        """
        if "edge_lat" not in self._ds:
            return None
        _set_desired_longitude_range(self._ds)
        return self._ds["edge_lat"]

    # ==================================================================================================================
    # Cartesian Edge Coordinates
    @property
    def edge_x(self) -> xr.DataArray:
        """Coordinate ``edge_x``, which contains the Cartesian x location of
        each edge in meters.

        Dimensions (``n_edge``)
        """
        if "edge_x" not in self._ds:
            return None

        return self._ds['edge_x']

    @property
    def edge_y(self) -> xr.DataArray:
        """Coordinate ``edge_y``, which contains the Cartesian y location of
        each edge in meters.

        Dimensions (``n_edge``)
        """
        if "edge_y" not in self._ds:
            return None
        return self._ds['edge_y']

    @property
    def edge_z(self) -> xr.DataArray:
        """Coordinate ``edge_z``, which contains the Cartesian z location of
        each edge in meters.

        Dimensions (``n_edge``)
        """
        if "edge_z" not in self._ds:
            return None
        return self._ds['edge_z']

    # ==================================================================================================================
    # Spherical Face Coordinates
    @property
    def face_lon(self) -> xr.DataArray:
        """Coordinate ``face_lon``, which contains the longitude of each face
        in degrees.

        Dimensions (``n_face``)
        """
        if "face_lon" not in self._ds:
            _populate_centroid_coord(self)
            _set_desired_longitude_range(self._ds)
        return self._ds["face_lon"]

    @property
    def face_lat(self) -> xr.DataArray:
        """Coordinate ``face_lat``, which contains the latitude of each face in
        degrees.

        Dimensions (``n_face``)
        """
        if "face_lat" not in self._ds:
            _populate_centroid_coord(self)
            _set_desired_longitude_range(self._ds)

        return self._ds["face_lat"]

    # ==================================================================================================================
    # Cartesian Face Coordinates
    @property
    def face_x(self) -> xr.DataArray:
        """Coordinate ``face_x``, which contains the Cartesian x location of
        each face in meters.

        Dimensions (``n_face``)
        """
        if "face_x" not in self._ds:
            return None

        return self._ds['face_x']

    @property
    def face_y(self) -> xr.DataArray:
        """Coordinate ``face_y``, which contains the Cartesian y location of
        each face in meters.

        Dimensions (``n_face``)
        """
        if "face_y" not in self._ds:
            return None
        return self._ds['face_y']

    @property
    def face_z(self) -> xr.DataArray:
        """Coordinate ``face_z``, which contains the Cartesian z location of
        each face in meters.

        Dimensions (``n_face``)
        """
        if "face_z" not in self._ds:
            return None
        return self._ds['face_z']

    # ==================================================================================================================
    # (, node) Connectivity
    @property
    def face_node_connectivity(self) -> xr.DataArray:
        """Connectivity Variable ``face_node_connectivity``, which maps each
        face to its corner nodes.

        Dimensions (``n_face``, ``n_max_face_nodes``) and
        DataType ``INT_DTYPE``.

        Nodes are in counter-clockwise order.
        """
        return self._ds["face_node_connectivity"]

    @property
    def edge_node_connectivity(self) -> xr.DataArray:
        """Connectivity Variable ``edge_node_connectivity``, which maps every
        edge to the two nodes that it connects.

        Dimensions (``n_edge``, ``two``) and DataType
        ``INT_DTYPE``.

        Nodes are in arbitrary order.
        """
        if "edge_node_connectivity" not in self._ds:
            _populate_edge_node_connectivity(self)

        return self._ds['edge_node_connectivity']

    @property
    def node_node_connectivity(self) -> xr.DataArray:
        """Connectivity Variable ``node_node_connectivity``."""
        return None

    # ==================================================================================================================
    # (, edge) Connectivity
    @property
    def face_edge_connectivity(self) -> xr.DataArray:
        """Connectivity Variable ``face_edge_connectivity``, which maps every
        face to its edges.

        Dimensions (``n_face``, ``n_max_face_nodes``) and DataType
        ``INT_DTYPE``.
        """
        if "face_edge_connectivity" not in self._ds:
            _populate_face_edge_connectivity(self)

        return self._ds["face_edge_connectivity"]

    @property
    def edge_edge_connectivity(self) -> xr.DataArray:
        """Connectivity Variable ``edge_edge_connectivity``."""
        return None

    @property
    def node_edge_connectivity(self) -> xr.DataArray:
        """Connectivity Variable ``node_edge_connectivity``."""
        return None

    # ==================================================================================================================
    # (, face) Connectivity
    @property
    def face_face_connectivity(self) -> xr.DataArray:
        """Connectivity Variable ``face_face_connectivity``."""
        return None

    @property
    def edge_face_connectivity(self) -> xr.DataArray:
        """Connectivity Variable ``edge_face_connectivity``, which contains the
        index of the faces that saddle a given edge.

        Dimensions (``n_edge``, ``TWO``) and DataType ``INT_DTYPE``.
        """
        if "edge_face_connectivity" not in self._ds:
            _populate_edge_face_connectivity(self)

        return self._ds["edge_face_connectivity"]

    @property
    def node_face_connectivity(self) -> xr.DataArray:
        """Connectivity Variable ``node_face_connectivity``, which maps every
        node to its faces.

        Dimensions (``n_node``, ``n_max_faces_per_node``) and DataType
        ``INT_DTYPE``.
        """
        if "node_face_connectivity" not in self._ds:
            _populate_node_face_connectivity(self)

        return self._ds["node_face_connectivity"]

    # ==================================================================================================================
    # Distance Quantities
    @property
    def edge_node_distances(self):
        """Contains the distance between the nodes that saddle a given edge.

        Dimensions (``n_edge``) and DataType float.
        """
        if "edge_node_distances" not in self._ds:
            return None
        return self._ds["edge_node_distances"]

    @property
    def edge_face_distances(self):
        """Contains the distance between the faces that saddle a given edge.

        Dimensions (``n_edge``) and DataType float.
        """
        if "edge_face_distances" not in self._ds:
            return None
        return self._ds["edge_face_distances"]

    # ==================================================================================================================
    # Other Grid Descriptor Quantities
    @property
    def antimeridian_face_indices(self) -> np.ndarray:
        """Index of each face that crosses the antimeridian."""
        if self._antimeridian_face_indices is None:
            self._antimeridian_face_indices = _populate_antimeridian_face_indices(
                self)
        return self._antimeridian_face_indices

    @property
    def face_areas(self) -> np.ndarray:
        """Declare face_areas as a property."""
        # if self._face_areas is not None: it allows for using the cached result
        if self._face_areas is None:
            self._face_areas, self._face_jacobian = self.compute_face_areas()
        return self._face_areas

    # ==================================================================================================================

    @property
    def face_jacobian(self):
        """Declare face_jacobian as a property."""
        # if self._face_jacobian is not None: it allows for using the cached result
        if self._face_jacobian is None:
            self._face_areas, self._face_jacobian = self.compute_face_areas()
        return self._face_jacobian

    def get_ball_tree(self, tree_type: Optional[str] = "nodes"):
        """Get the BallTree data structure of this Grid that allows for nearest
        neighbor queries (k nearest or within some radius) on either the nodes
        (``node_lon``, ``node_lat``) or face centers (``face_lon``,
        ``face_lat``).

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

    def get_kd_tree(self, tree_type: Optional[str] = "nodes"):
        """Get the KDTree data structure of this Grid that allows for nearest
        neighbor queries (k nearest or within some radius) on either the nodes
        (``node_x``, ``node_y``, ``node_z``) or face centers (``face_x``,
        ``face_y``, ``face_z``).

        Parameters
        ----------
        tree_type : str, default="nodes"
            Selects which tree to query, with "nodes" selecting the Corner Nodes and "face centers" selecting the Face
            Centers of each face

        Returns
        -------
        self._kd_tree : grid.Neighbors.KDTree
            KDTree instance
        """
        if self._kd_tree is None:
            self._kd_tree = KDTree(self,
                                   tree_type=tree_type,
                                   distance_metric='minkowski')
        else:
            if tree_type != self._kd_tree._tree_type:
                self._kd_tree.tree_type = tree_type

        return self._kd_tree

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
            out_ds = _encode_scrip(self.face_node_connectivity, self.node_lon,
                                   self.node_lat, self.face_areas)
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
        face_areas, face_jacobian = self.compute_face_areas(
            quadrature_rule, order)

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
        1. Area of all the faces in the mesh : np.ndarray
        2. Jacobian of all the faces in the mesh : np.ndarray

        Examples
        --------
        Open a uxarray grid file

        >>> grid = ux.open_dataset("/home/jain/uxarray/test/meshfiles/ugrid/outCSne30/outCSne30.ug")


        >>> grid.face_areas
        array([0.00211174, 0.00211221, 0.00210723, ..., 0.00210723, 0.00211221,
            0.00211174])
        """
        # if self._face_areas is None: # this allows for using the cached result,
        # but is not the expected behavior behavior as we are in need to recompute if this function is called with different quadrature_rule or order

        if latlon:
            x = self.node_lon.data
            y = self.node_lat.data
            z = np.zeros((self.n_node))
            coords_type = "spherical"
        else:
            x = self.node_x.data
            y = self.node_y.data
            z = self.node_z.data
            coords_type = "cartesian"

        dim = 2

        # Note: x, y, z are np arrays of type float
        # Using np.issubdtype to check if the type is float
        # if not (int etc.), convert to float, this is to avoid numba errors
        x, y, z = (arr.astype(float)
                   if not np.issubdtype(arr[0], np.floating) else arr
                   for arr in (x, y, z))

        face_nodes = self.face_node_connectivity.values
        n_nodes_per_face = self.n_nodes_per_face.values

        # call function to get area of all the faces as a np array
        self._face_areas, self._face_jacobian = get_all_face_area_from_coords(
            x, y, z, face_nodes, n_nodes_per_face, dim, quadrature_rule, order,
            coords_type)

        min_jacobian = np.min(self._face_jacobian)
        max_jacobian = np.max(self._face_jacobian)

        if np.any(self._face_jacobian < 0):
            raise ValueError(
                "Negative jacobian found. Min jacobian: {}, Max jacobian: {}".
                format(min_jacobian, max_jacobian))

        return self._face_areas, self._face_jacobian

    def to_geodataframe(self,
                        override: Optional[bool] = False,
                        cache: Optional[bool] = True,
                        exclude_antimeridian: Optional[bool] = False):
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
        exclude_antimeridian: bool
            Selects whether to exclude any face that contains an edge that crosses the antimeridian

        Returns
        -------
        gdf : spatialpandas.GeoDataFrame
            The output `GeoDataFrame` with a filled out "geometry" collumn
        """

        if self._gdf is not None:
            # determine if we need to recompute a cached GeoDataFrame based on antimeridian
            if self._gdf_exclude_am != exclude_antimeridian:
                # cached gdf should match the exclude_antimeridian_flag
                override = True

        # use cached geodataframe
        if self._gdf is not None and not override:
            return self._gdf

        # construct a geodataframe with the faces stored as polygons as the geometry
        gdf = _grid_to_polygon_geodataframe(
            self, exclude_antimeridian=exclude_antimeridian)

        # cache computed geodataframe
        if cache:
            self._gdf = gdf
            self._gdf_exclude_am = exclude_antimeridian

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
