"""uxarray.core.grid module."""

import xarray as xr
import numpy as np

from typing import (
    Optional,
    Union,
)


# reader and writer imports
from uxarray.io._exodus import _read_exodus, _encode_exodus
from uxarray.io._mpas import _read_mpas
from uxarray.io._ugrid import (
    _read_ugrid,
    _encode_ugrid,
    _validate_minimum_ugrid,
)
from uxarray.io._scrip import _read_scrip, _encode_scrip
from uxarray.io._esmf import _read_esmf
from uxarray.io._vertices import _read_face_vertices
from uxarray.io._topology import _read_topology
from uxarray.io._geos import _read_geos_cs

from uxarray.io.utils import _parse_grid_type
from uxarray.grid.area import get_all_face_area_from_coords
from uxarray.grid.coordinates import (
    _populate_face_centroids,
    _populate_edge_centroids,
    _set_desired_longitude_range,
    _populate_node_latlon,
    _populate_node_xyz,
)
from uxarray.grid.connectivity import (
    _populate_edge_node_connectivity,
    _populate_face_edge_connectivity,
    _populate_n_nodes_per_face,
    _populate_node_face_connectivity,
    _populate_edge_face_connectivity,
)

from uxarray.grid.geometry import (
    _populate_antimeridian_face_indices,
    _grid_to_polygon_geodataframe,
    _grid_to_matplotlib_polycollection,
    _grid_to_matplotlib_linecollection,
    _populate_bounds,
)

from uxarray.grid.neighbors import (
    BallTree,
    KDTree,
    _populate_edge_face_distances,
    _populate_edge_node_distances,
)

from uxarray.plot.accessor import GridPlotAccessor

from uxarray.subset import GridSubsetAccessor

from uxarray.grid.validation import (
    _check_connectivity,
    _check_duplicate_nodes,
    _check_area,
)


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

    def __init__(
        self,
        grid_ds: xr.Dataset,
        source_grid_spec: Optional[str] = None,
        source_dims_dict: Optional[dict] = {},
    ):
        # check if inputted dataset is a minimum representable 2D UGRID unstructured grid
        if not _validate_minimum_ugrid(grid_ds):
            raise ValueError(
                "Grid unable to be represented in the UGRID conventions. Representing an unstructured grid requires "
                "at least the following variables: ['node_lon',"
                "'node_lat', and 'face_node_connectivity']"
            )

        # grid spec not provided, check if grid_ds is a minimum representable UGRID dataset
        if source_grid_spec is None:
            warn(
                "Attempting to construct a Grid without passing in source_grid_spec. Direct use of Grid constructor"
                "is only advised if grid_ds is following the internal unstructured grid definition, including"
                "variable and dimension names. Using ux.open_grid() or ux.from_dataset() is suggested.",
                Warning,
            )
            # TODO: more checks for validate grid (lat/lon coords, etc)

        # mapping of ugrid dimensions and variables to source dataset's conventions
        self._source_dims_dict = source_dims_dict

        # source grid specification (i.e. UGRID, MPAS, SCRIP, etc.)
        self.source_grid_spec = source_grid_spec

        # internal xarray dataset for storing grid variables
        self._ds = grid_ds

        # initialize attributes
        self._antimeridian_face_indices = None

        # initialize cached data structures (visualization)
        self._gdf = None
        self._gdf_exclude_am = None
        self._poly_collection = None
        self._line_collection = None
        self._raster_data_id = None

        # initialize cached data structures (nearest neighbor operations)
        self._ball_tree = None
        self._kd_tree = None

        # set desired longitude range to [-180, 180]
        _set_desired_longitude_range(self._ds)

    # declare plotting accessor
    plot = UncachedAccessor(GridPlotAccessor)

    # declare subset accessor
    subset = UncachedAccessor(GridSubsetAccessor)

    @classmethod
    def from_dataset(
        cls, dataset: xr.Dataset, use_dual: Optional[bool] = False, **kwargs
    ):
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

        if "source_grid_spec" not in kwargs:
            # parse to detect source grid spec

            source_grid_spec = _parse_grid_type(dataset)
            if source_grid_spec == "Exodus":
                grid_ds, source_dims_dict = _read_exodus(dataset)
            elif source_grid_spec == "Scrip":
                grid_ds, source_dims_dict = _read_scrip(dataset)
            elif source_grid_spec == "UGRID":
                grid_ds, source_dims_dict = _read_ugrid(dataset)
            elif source_grid_spec == "MPAS":
                grid_ds, source_dims_dict = _read_mpas(dataset, use_dual=use_dual)
            elif source_grid_spec == "ESMF":
                grid_ds, source_dims_dict = _read_esmf(dataset)
            elif source_grid_spec == "GEOS-CS":
                grid_ds, source_dims_dict = _read_geos_cs(dataset)
            elif source_grid_spec == "Shapefile":
                raise ValueError("Shapefiles not yet supported")
            else:
                raise ValueError("Unsupported Grid Format")
        else:
            # custom source grid spec is provided
            source_grid_spec = kwargs.get("source_grid_spec", None)
            grid_ds = dataset
            source_dims_dict = {}

        return cls(grid_ds, source_grid_spec, source_dims_dict)

    @classmethod
    def from_topology(
        cls,
        node_lon: np.ndarray,
        node_lat: np.ndarray,
        face_node_connectivity: np.ndarray,
        fill_value: Optional = None,
        start_index: Optional[int] = 0,
        dims_dict: Optional[dict] = None,
        **kwargs,
    ):
        """Constructs a ``Grid`` object from user-defined topology variables
        provided in the UGRID conventions.

        Note
        ----
        To construct a UGRID-complient grid, the user must provide at least ``node_lon``, ``node_lat`` and ``face_node_connectivity``

        Parameters
        ----------
        node_lon : np.ndarray
            Longitude of node coordinates
        node_lat : np.ndarray
            Latitude of node coordinates
        face_node_connectivity : np.ndarray
            Face node connectivity, mapping each face to the nodes that surround them
        fill_value: Optional
            Value used for padding connectivity variables when the maximum number of elements in a row is less than the maximum.
        start_index: Optional, default=0
            Start index (typically 0 or 1)
        dims_dict : Optional, dict
            Dictionary of dimension names mapped to the ugrid conventions (i.e. {"nVertices": "n_node})
        **kwargs :

        Usage
        -----
        >>> import uxarray as ux
        >>> node_lon, node_lat, face_node_connectivity, fill_value = ...
        >>> uxgrid = ux.Grid.from_ugrid(node_lon, node_lat, face_node_connectivity, fill_value)
        """

        if dims_dict is None:
            dims_dict = {}

        grid_ds = _read_topology(
            node_lon,
            node_lat,
            face_node_connectivity,
            fill_value,
            start_index,
            **kwargs,
        )
        grid_spec = "User Defined Topology"
        return cls(grid_ds, grid_spec, dims_dict)

    @classmethod
    def from_face_vertices(
        cls,
        face_vertices: Union[list, tuple, np.ndarray],
        latlon: Optional[bool] = True,
    ):
        """Constructs a ``Grid`` object from user-defined face vertices.

        Parameters
        ----------
        face_vertices : list, tuple, np.ndarray
            array-like input containing the face vertices to construct the grid from
        latlon : bool, default=True
            Indicates whether the inputted vertices are in lat/lon, with units in degrees
        """
        if not isinstance(face_vertices, (list, tuple, np.ndarray)):
            raise ValueError("Input must be either a list, tuple, or np.ndarray")

        face_vertices = np.asarray(face_vertices)

        if face_vertices.ndim == 3:
            grid_ds = _read_face_vertices(face_vertices, latlon)

        elif face_vertices.ndim == 2:
            grid_ds = _read_face_vertices(np.array([face_vertices]), latlon)

        else:
            raise RuntimeError(
                f"Invalid Input Dimension: {face_vertices.ndim}. Expected dimension should be "
                f"3: [n_face, n_node, two/three] or 2 when only "
                f"one face is passed in."
            )

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

        from uxarray.conventions import ugrid, descriptors

        prefix = "<uxarray.Grid>\n"
        original_grid_str = f"Original Grid Type: {self.source_grid_spec}\n"
        dims_heading = "Grid Dimensions:\n"
        dims_str = ""

        for dim_name in ugrid.DIM_NAMES:
            if dim_name in self._ds.sizes:
                dims_str += f"  * {dim_name}: {self._ds.sizes[dim_name]}\n"

        dims_str += f"  * n_nodes_per_face: {self.n_nodes_per_face.shape}\n"

        coord_heading = "Grid Coordinates (Spherical):\n"
        coords_str = ""
        for coord_name in ugrid.SPHERICAL_COORD_NAMES:
            if coord_name in self._ds:
                coords_str += f"  * {coord_name}: {getattr(self, coord_name).shape}\n"

        coords_str += "Grid Coordinates (Cartesian):\n"
        for coord_name in ugrid.CARTESIAN_COORD_NAMES:
            if coord_name in self._ds:
                coords_str += f"  * {coord_name}: {getattr(self, coord_name).shape}\n"

        connectivity_heading = "Grid Connectivity Variables:\n"
        connectivity_str = ""

        for conn_name in ugrid.CONNECTIVITY_NAMES:
            if conn_name in self._ds:
                connectivity_str += (
                    f"  * {conn_name}: {getattr(self, conn_name).shape}\n"
                )

        descriptors_heading = "Grid Descriptor Variables:\n"
        descriptors_str = ""

        for descriptor_name in descriptors.DESCRIPTOR_NAMES:
            if descriptor_name in self._ds:
                descriptors_str += (
                    f"  * {descriptor_name}: {getattr(self, descriptor_name).shape}\n"
                )

        return (
            prefix
            + original_grid_str
            + dims_heading
            + dims_str
            + coord_heading
            + coords_str
            + connectivity_heading
            + connectivity_str
            + descriptors_heading
            + descriptors_str
        )

    def __getitem__(self, item):
        """Implementation of getitem operator for indexing a grid to obtain
        variables.

        Usage
        -----
        >>> uxgrid['face_node_connectivity']
        """
        return getattr(self, item)

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

        if not (
            self.node_lon.equals(other.node_lon) or self.node_lat.equals(other.node_lat)
        ):
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
    def dims(self) -> set:
        """Names of all unstructured grid dimensions."""
        from uxarray.conventions.ugrid import DIM_NAMES

        return set([dim for dim in DIM_NAMES if dim in self._ds.dims])

    @property
    def sizes(self) -> dict:
        """Names and values of all unstructured grid dimensions."""
        return {dim: self._ds.sizes[dim] for dim in self.dims}

    @property
    def coordinates(self) -> set:
        """Names of all coordinate variables."""
        from uxarray.conventions.ugrid import (
            SPHERICAL_COORD_NAMES,
            CARTESIAN_COORD_NAMES,
        )

        return set(
            [coord for coord in SPHERICAL_COORD_NAMES if coord in self._ds]
        ).union(set([coord for coord in CARTESIAN_COORD_NAMES if coord in self._ds]))

    @property
    def connectivity(self) -> set:
        """Names of all connectivity variables."""
        from uxarray.conventions.ugrid import CONNECTIVITY_NAMES

        return set([conn for conn in CONNECTIVITY_NAMES if conn in self._ds])

    @property
    def parsed_attrs(self) -> dict:
        """Dictionary of parsed attributes from the source grid."""
        warn(
            "Grid.parsed_attrs will be deprecated in a future release. Please use Grid.attrs instead.",
            DeprecationWarning,
        )
        return self._ds.attrs

    @property
    def attrs(self) -> dict:
        """Dictionary of parsed attributes from the source grid."""
        return self._ds.attrs

    @property
    def n_node(self) -> int:
        """Total number of nodes."""
        return self._ds.sizes["n_node"]

    @property
    def n_edge(self) -> int:
        """Total number of edges."""
        if "edge_node_connectivity" not in self._ds:
            _populate_edge_node_connectivity(self)

        return self._ds.sizes["n_edge"]

    @property
    def n_face(self) -> int:
        """Total number of faces."""
        return self._ds.sizes["n_face"]

    @property
    def n_max_face_nodes(self) -> int:
        """The maximum number of nodes that can make up a single face."""
        return self.face_node_connectivity.shape[1]

    @property
    def n_max_face_edges(self) -> int:
        """The maximum number of edges that surround a single face.

        Equivalent to ``n_max_face_nodes``
        """
        return self.face_edge_connectivity.shape[1]

    @property
    def n_max_face_faces(self) -> int:
        """The maximum number of faces that surround a single face."""
        return self.face_face_connectivity.shape[1]

    @property
    def n_max_edge_edges(self) -> int:
        """The maximum number of edges that surround a single edge."""
        return self.edge_edge_connectivity.shape[1]

    @property
    def n_max_node_faces(self) -> int:
        """The maximum number of faces that surround a single node."""
        return self.node_face_connectivity.shape[1]

    @property
    def n_max_node_edges(self) -> int:
        """The maximum number of edges that surround a single node."""
        return self.node_edge_connectivity.shape[1]

    @property
    def n_nodes_per_face(self) -> xr.DataArray:
        """The number of nodes that make up each face.

        Dimensions: ``(n_node, )``
        """
        if "n_nodes_per_face" not in self._ds:
            _populate_n_nodes_per_face(self)

        return self._ds["n_nodes_per_face"]

    @property
    def node_lon(self) -> xr.DataArray:
        """Longitude of each node in degrees.

        Dimensions: ``(n_node, )``
        """
        if "node_lon" not in self._ds:
            _set_desired_longitude_range(self._ds)
            _populate_node_latlon(self)
        return self._ds["node_lon"]

    @property
    def node_lat(self) -> xr.DataArray:
        """Latitude of each node in degrees.

        Dimensions: ``(n_node, )``
        """
        if "node_lat" not in self._ds:
            _set_desired_longitude_range(self._ds)
            _populate_node_latlon(self)
        return self._ds["node_lat"]

    @property
    def node_x(self) -> xr.DataArray:
        """Cartesian x location of each node in meters.

        Dimensions: ``(n_node, )``
        """
        if "node_x" not in self._ds:
            _populate_node_xyz(self)

        return self._ds["node_x"]

    @property
    def node_y(self) -> xr.DataArray:
        """Cartesian y location of each node in meters.

        Dimensions: ``(n_node, )``
        """
        if "node_y" not in self._ds:
            _populate_node_xyz(self)
        return self._ds["node_y"]

    @property
    def node_z(self) -> xr.DataArray:
        """Cartesian z location of each node in meters.

        Dimensions: ``(n_node, )``
        """
        if "node_z" not in self._ds:
            _populate_node_xyz(self)
        return self._ds["node_z"]

    @property
    def edge_lon(self) -> xr.DataArray:
        """Longitude of the center of each edge in degrees.

        Dimensions: ``(n_edge, )``
        """
        if "edge_lon" not in self._ds:
            _populate_edge_centroids(self)
        # temp until we construct edge lon
        _set_desired_longitude_range(self._ds)
        return self._ds["edge_lon"]

    @property
    def edge_lat(self) -> xr.DataArray:
        """Latitude of the center of each edge in degrees.

        Dimensions: ``(n_edge, )``
        """
        if "edge_lat" not in self._ds:
            _populate_edge_centroids(self)
        _set_desired_longitude_range(self._ds)
        return self._ds["edge_lat"]

    @property
    def edge_x(self) -> xr.DataArray:
        """Cartesian x location of the center of each edge in meters.

        Dimensions: ``(n_edge, )``
        """
        if "edge_x" not in self._ds:
            _populate_edge_centroids(self)

        return self._ds["edge_x"]

    @property
    def edge_y(self) -> xr.DataArray:
        """Cartesian y location of the center of each edge in meters.

        Dimensions: ``(n_edge, )``
        """
        if "edge_y" not in self._ds:
            _populate_edge_centroids(self)
        return self._ds["edge_y"]

    @property
    def edge_z(self) -> xr.DataArray:
        """Cartesian z location of the center of each edge in meters.

        Dimensions: ``(n_edge, )``
        """
        if "edge_z" not in self._ds:
            _populate_edge_centroids(self)
        return self._ds["edge_z"]

    @property
    def face_lon(self) -> xr.DataArray:
        """Longitude of the center of each face in degrees.

        Dimensions: ``(n_face, )``
        """
        if "face_lon" not in self._ds:
            _populate_face_centroids(self)
            _set_desired_longitude_range(self._ds)
        return self._ds["face_lon"]

    @property
    def face_lat(self) -> xr.DataArray:
        """Latitude of the center of each face in degrees.

        Dimensions: ``(n_face, )``
        """
        if "face_lat" not in self._ds:
            _populate_face_centroids(self)
            _set_desired_longitude_range(self._ds)

        return self._ds["face_lat"]

    @property
    def face_x(self) -> xr.DataArray:
        """Cartesian x location of the center of each face in meters.

        Dimensions: ``(n_face, )``
        """
        if "face_x" not in self._ds:
            _populate_face_centroids(self)

        return self._ds["face_x"]

    @property
    def face_y(self) -> xr.DataArray:
        """Cartesian y location of the center of each face in meters.

        Dimensions: ``(n_face, )``
        """
        if "face_y" not in self._ds:
            _populate_face_centroids(self)
        return self._ds["face_y"]

    @property
    def face_z(self) -> xr.DataArray:
        """Cartesian z location of the center of each face in meters.

        Dimensions: ``(n_face, )``
        """
        if "face_z" not in self._ds:
            _populate_face_centroids(self)
        return self._ds["face_z"]

    @property
    def face_node_connectivity(self) -> xr.DataArray:
        """Indices of the nodes that make up each face.

        Dimensions: ``(n_face, n_max_face_nodes)``

        Nodes are in counter-clockwise order.
        """

        if self._ds["face_node_connectivity"].values.ndim == 1:
            face_node_connectivity_1d = self._ds["face_node_connectivity"].values
            face_node_connectivity_2d = np.expand_dims(
                face_node_connectivity_1d, axis=0
            )
            self._ds["face_node_connectivity"] = xr.DataArray(
                data=face_node_connectivity_2d,
                dims=["n_face", "n_max_face_nodes"],
                attrs=self._ds["face_node_connectivity"].attrs,
            )

        return self._ds["face_node_connectivity"]

    @property
    def edge_node_connectivity(self) -> xr.DataArray:
        """Indices of the two nodes that make up each edge.

        Dimensions: ``(n_edge, n_max_edge_nodes)``

        Nodes are in arbitrary order.
        """
        if "edge_node_connectivity" not in self._ds:
            _populate_edge_node_connectivity(self)

        return self._ds["edge_node_connectivity"]

    @property
    def node_node_connectivity(self) -> xr.DataArray:
        """Indices of the nodes that surround each node."""
        if "node_node_connectivity" not in self._ds:
            raise NotImplementedError(
                "Construction of `node_node_connectivity` not yet supported."
            )
        return self._ds["node_node_connectivity"]

    @property
    def face_edge_connectivity(self) -> xr.DataArray:
        """Indices of the edges that surround each face.

        Dimensions: ``(n_face, n_max_face_edges)``
        """
        if "face_edge_connectivity" not in self._ds:
            _populate_face_edge_connectivity(self)

        return self._ds["face_edge_connectivity"]

    @property
    def edge_edge_connectivity(self) -> xr.DataArray:
        """Indices of the edges that surround each edge.

        Dimensions: ``(n_face, n_max_edge_edges)``
        """
        if "edge_edge_connectivity" not in self._ds:
            raise NotImplementedError(
                "Construction of `edge_edge_connectivity` not yet supported."
            )

        return self._ds["edge_edge_connectivity"]

    @property
    def node_edge_connectivity(self) -> xr.DataArray:
        """Indices of the edges that surround each node."""
        if "node_edge_connectivity" not in self._ds:
            raise NotImplementedError(
                "Construction of `node_edge_connectivity` not yet supported."
            )

        return self._ds["node_edge_connectivity"]

    @property
    def face_face_connectivity(self) -> xr.DataArray:
        """Indices of the faces that surround each face.

        Dimensions ``(n_face, n_max_face_faces)``
        """
        if "face_face_connectivity" not in self._ds:
            raise NotImplementedError(
                "Construction of `face_face_connectivity` not yet supported."
            )

        return self._ds["face_face_connectivity"]

    @property
    def edge_face_connectivity(self) -> xr.DataArray:
        """Indices of the faces that saddle each edge.

        Dimensions ``(n_edge, two)``
        """
        if "edge_face_connectivity" not in self._ds:
            _populate_edge_face_connectivity(self)

        return self._ds["edge_face_connectivity"]

    @property
    def node_face_connectivity(self) -> xr.DataArray:
        """Indices of the faces that surround each node.

        Dimensions ``(n_node, n_max_node_faces)``
        """
        if "node_face_connectivity" not in self._ds:
            _populate_node_face_connectivity(self)

        return self._ds["node_face_connectivity"]

    @property
    def edge_node_distances(self):
        """Distances between the two nodes that surround each edge.

        Dimensions ``(n_edge, )``
        """
        if "edge_node_distances" not in self._ds:
            _populate_edge_node_distances(self)
        return self._ds["edge_node_distances"]

    @property
    def edge_face_distances(self):
        """Distances between the centers of the faces that saddle each edge.

        Dimensions ``(n_edge, )``
        """
        if "edge_face_distances" not in self._ds:
            _populate_edge_face_distances(self)
        return self._ds["edge_face_distances"]

    @property
    def antimeridian_face_indices(self) -> np.ndarray:
        """Index of each face that crosses the antimeridian."""
        if self._antimeridian_face_indices is None:
            self._antimeridian_face_indices = _populate_antimeridian_face_indices(self)
        return self._antimeridian_face_indices

    @property
    def face_areas(self) -> xr.DataArray:
        """The area of each face."""
        from uxarray.conventions.descriptors import FACE_AREAS_DIMS, FACE_AREAS_ATTRS

        if "face_areas" not in self._ds:
            face_areas, self._face_jacobian = self.compute_face_areas()
            self._ds["face_areas"] = xr.DataArray(
                data=face_areas, dims=FACE_AREAS_DIMS, attrs=FACE_AREAS_ATTRS
            )
        return self._ds["face_areas"]

    @property
    def bounds(self):
        """Latitude Longitude Bounds for each Face in degrees.

        Dimensions ``(n_face", two, two)``
        """
        if "bounds" not in self._ds:
            warn(
                "Constructing of `Grid.bounds` has not been optimized, which may lead to a long execution time."
            )
            _populate_bounds(self)
        return self._ds["bounds"]

    @property
    def face_jacobian(self):
        """Declare face_jacobian as a property."""
        if self._face_jacobian is None:
            _ = self.face_areas
        return self._face_jacobian

    def get_ball_tree(
        self,
        coordinates: Optional[str] = "nodes",
        coordinate_system: Optional[str] = "spherical",
        distance_metric: Optional[str] = "haversine",
        reconstruct: bool = False,
    ):
        """Get the BallTree data structure of this Grid that allows for nearest
        neighbor queries (k nearest or within some radius) on either the
        (``node_x``, ``node_y``, ``node_z``) and (``node_lon``, ``node_lat``),
        edge (``edge_x``, ``edge_y``, ``edge_z``) and (``edge_lon``,
        ``edge_lat``), or center (``face_x``, ``face_y``, ``face_z``) and
        (``face_lon``, `   `face_lat``) nodes.

        Parameters
        ----------
        coordinates : str, default="nodes"
            Selects which tree to query, with "nodes" selecting the Corner Nodes, "edge centers" selecting the Edge
            Centers of each edge, and "face centers" selecting the Face Centers of each face
        coordinate_system : str, default="cartesian"
            Selects which coordinate type to use to create the tree, "cartesian" selecting cartesian coordinates, and
            "spherical" selecting spherical coordinates.
        distance_metric : str, default="haversine"
            Distance metric used to construct the BallTree, options include:
            'euclidean', 'l2', 'minkowski', 'p','manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity', 'seuclidean',
            'mahalanobis', 'hamming', 'canberra', 'braycurtis', 'jaccard', 'dice', 'rogerstanimoto', 'russellrao',
            'sokalmichener', 'sokalsneath', 'haversine'
        reconstruct : bool, default=False
            If true, reconstructs the tree

        Returns
        -------
        self._ball_tree : grid.Neighbors.BallTree
            BallTree instance
        """

        if self._ball_tree is None or reconstruct:
            self._ball_tree = BallTree(
                self,
                coordinates=coordinates,
                distance_metric=distance_metric,
                coordinate_system=coordinate_system,
                reconstruct=reconstruct,
            )
        else:
            if coordinates != self._ball_tree._coordinates:
                self._ball_tree.coordinates = coordinates

        return self._ball_tree

    def get_kd_tree(
        self,
        coordinates: Optional[str] = "nodes",
        coordinate_system: Optional[str] = "cartesian",
        distance_metric: Optional[str] = "minkowski",
        reconstruct: bool = False,
    ):
        """Get the KDTree data structure of this Grid that allows for nearest
        neighbor queries (k nearest or within some radius) on either the
        (``node_x``, ``node_y``, ``node_z``) and (``node_lon``, ``node_lat``),
        edge (``edge_x``, ``edge_y``, ``edge_z``) and (``edge_lon``,
        ``edge_lat``), or center (``face_x``, ``face_y``, ``face_z``) and
        (``face_lon``, ``face_lat``) nodes.

        Parameters
        ----------
        coordinates : str, default="nodes"
            Selects which tree to query, with "nodes" selecting the Corner Nodes, "edge centers" selecting the Edge
            Centers of each edge, and "face centers" selecting the Face Centers of each face
        coordinate_system : str, default="cartesian"
            Selects which coordinate type to use to create the tree, "cartesian" selecting cartesian coordinates, and
            "spherical" selecting spherical coordinates.
        distance_metric : str, default="minkowski"
            Distance metric used to construct the KDTree, available options include:
            'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity'
        reconstruct : bool, default=False
            If true, reconstructs the tree

        Returns
        -------
        self._kd_tree : grid.Neighbors.KDTree
            KDTree instance
        """

        if self._kd_tree is None or reconstruct:
            self._kd_tree = KDTree(
                self,
                coordinates=coordinates,
                distance_metric=distance_metric,
                coordinate_system=coordinate_system,
                reconstruct=reconstruct,
            )

        else:
            if coordinates != self._kd_tree._coordinates:
                self._kd_tree.coordinates = coordinates

        return self._kd_tree

    def copy(self):
        """Returns a deep copy of this grid."""

        return Grid(
            self._ds,
            source_grid_spec=self.source_grid_spec,
            source_dims_dict=self._source_dims_dict,
        )

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

        warn(
            "Grid.encode_as will be deprecated in a future release. Please use Grid.to_xarray instead."
        )

        if grid_type == "UGRID":
            out_ds = _encode_ugrid(self._ds)

        elif grid_type == "Exodus":
            out_ds = _encode_exodus(self._ds)

        elif grid_type == "SCRIP":
            out_ds = _encode_scrip(
                self.face_node_connectivity,
                self.node_lon,
                self.node_lat,
                self.face_areas,
            )
        else:
            raise RuntimeError("The grid type not supported: ", grid_type)

        return out_ds

    def calculate_total_face_area(
        self, quadrature_rule: Optional[str] = "triangular", order: Optional[int] = 4
    ) -> float:
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
        face_areas, face_jacobian = self.compute_face_areas(quadrature_rule, order)

        return np.sum(face_areas)

    def compute_face_areas(
        self,
        quadrature_rule: Optional[str] = "triangular",
        order: Optional[int] = 4,
        latlon: Optional[bool] = True,
    ):
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
        x, y, z = (
            arr.astype(float) if not np.issubdtype(arr[0], np.floating) else arr
            for arr in (x, y, z)
        )

        face_nodes = self.face_node_connectivity.values
        n_nodes_per_face = self.n_nodes_per_face.values

        # call function to get area of all the faces as a np array
        self._face_areas, self._face_jacobian = get_all_face_area_from_coords(
            x,
            y,
            z,
            face_nodes,
            n_nodes_per_face,
            dim,
            quadrature_rule,
            order,
            coords_type,
        )

        min_jacobian = np.min(self._face_jacobian)
        max_jacobian = np.max(self._face_jacobian)

        if np.any(self._face_jacobian < 0):
            raise ValueError(
                "Negative jacobian found. Min jacobian: {}, Max jacobian: {}".format(
                    min_jacobian, max_jacobian
                )
            )

        return self._face_areas, self._face_jacobian

    def to_xarray(self, grid_format: Optional[str] = "ugrid"):
        """Returns a xarray Dataset representation in a specific grid format
        from the Grid object.

        Parameters
        ----------
        grid_format: str, optional
            The desired grid format for the output dataset.
            One of "ugrid", "exodus", or "scrip"

        Returns
        -------
        out_ds: xarray.Dataset
            Dataset representing the unstructured grid in a given grid format
        """

        if grid_format == "ugrid":
            out_ds = _encode_ugrid(self._ds)

        elif grid_format == "exodus":
            out_ds = _encode_exodus(self._ds)

        elif grid_format == "scrip":
            out_ds = _encode_scrip(
                self.face_node_connectivity,
                self.node_lon,
                self.node_lat,
                self.face_areas,
            )

        else:
            raise ValueError(
                f"Invalid grid_format encountered. Expected one of ['ugrid', 'exodus', 'scrip'] but received: {grid_format}"
            )

        return out_ds

    def to_geodataframe(
        self,
        override: Optional[bool] = False,
        cache: Optional[bool] = True,
        exclude_antimeridian: Optional[bool] = False,
    ):
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
            self, exclude_antimeridian=exclude_antimeridian
        )

        # cache computed geodataframe
        if cache:
            self._gdf = gdf
            self._gdf_exclude_am = exclude_antimeridian

        return gdf

    def to_polycollection(
        self,
        override: Optional[bool] = False,
        cache: Optional[bool] = True,
        correct_antimeridian_polygons: Optional[bool] = True,
    ):
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

        (
            poly_collection,
            corrected_to_original_faces,
        ) = _grid_to_matplotlib_polycollection(self)

        # cache computed polycollection
        if cache:
            self._poly_collection = poly_collection

        return poly_collection, corrected_to_original_faces

    def to_linecollection(
        self, override: Optional[bool] = False, cache: Optional[bool] = True
    ):
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

    def isel(self, **dim_kwargs):
        """Indexes an unstructured grid along a given dimension (``n_node``,
        ``n_edge``, or ``n_face``) and returns a new grid.

        Currently only supports inclusive selection, meaning that for cases where node or edge indices are provided,
        any face that contains that element is included in the resulting subset. This means that additional elements
        beyond those that were initially provided in the indices will be included. Support for more methods, such as
        exclusive and clipped indexing is in the works.

        Parameters
        **dims_kwargs: kwargs
            Dimension to index, one of ['n_node', 'n_edge', 'n_face']


        Example
        -------`
        >> grid = ux.open_grid(grid_path)
        >> grid.isel(n_face = [1,2,3,4])
        """
        from .slice import _slice_node_indices, _slice_edge_indices, _slice_face_indices

        if len(dim_kwargs) != 1:
            raise ValueError("Indexing must be along a single dimension.")

        if "n_node" in dim_kwargs:
            return _slice_node_indices(self, dim_kwargs["n_node"])

        elif "n_edge" in dim_kwargs:
            return _slice_edge_indices(self, dim_kwargs["n_edge"])

        elif "n_face" in dim_kwargs:
            return _slice_face_indices(self, dim_kwargs["n_face"])

        else:
            raise ValueError(
                "Indexing must be along a grid dimension: ('n_node', 'n_edge', 'n_face')"
            )
