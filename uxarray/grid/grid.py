import xarray as xr
import numpy as np

from html import escape

from xarray.core.options import OPTIONS

from typing import (
    Optional,
    Union,
)

# reader and writer imports
from uxarray.io._exodus import _read_exodus, _encode_exodus
from uxarray.io._mpas import _read_mpas
from uxarray.io._geopandas import _read_geodataframe
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
from uxarray.io._icon import _read_icon
from uxarray.io._structured import _read_structured_grid
from uxarray.io._voronoi import _spherical_voronoi_from_points
from uxarray.io._delaunay import (
    _spherical_delaunay_from_points,
    _regional_delaunay_from_points,
)

from uxarray.formatting_html import grid_repr

from uxarray.io.utils import _parse_grid_type
from uxarray.grid.area import get_all_face_area_from_coords
from uxarray.grid.coordinates import (
    _populate_face_centroids,
    _populate_edge_centroids,
    _populate_face_centerpoints,
    _set_desired_longitude_range,
    _populate_node_latlon,
    _populate_node_xyz,
    _normalize_xyz,
    prepare_points,
)
from uxarray.grid.connectivity import (
    _populate_edge_node_connectivity,
    _populate_face_edge_connectivity,
    _populate_n_nodes_per_face,
    _populate_node_face_connectivity,
    _populate_edge_face_connectivity,
    _populate_face_face_connectivity,
)

from uxarray.grid.geometry import (
    _populate_antimeridian_face_indices,
    _grid_to_polygon_geodataframe,
    _grid_to_matplotlib_polycollection,
    _grid_to_matplotlib_linecollection,
    _populate_bounds,
    _construct_boundary_edge_indices,
    compute_temp_latlon_array,
)

from uxarray.grid.neighbors import (
    BallTree,
    KDTree,
    _populate_edge_face_distances,
    _populate_edge_node_distances,
)

from uxarray.grid.intersections import (
    constant_lat_intersections_no_extreme,
    constant_lon_intersections_no_extreme,
    constant_lat_intersections_face_bounds,
    constant_lon_intersections_face_bounds,
)

from spatialpandas import GeoDataFrame

from uxarray.plot.accessor import GridPlotAccessor

from uxarray.subset import GridSubsetAccessor

from uxarray.cross_sections import GridCrossSectionAccessor

from uxarray.grid.validation import (
    _check_connectivity,
    _check_duplicate_nodes,
    _check_duplicate_nodes_indices,
    _check_area,
    _check_normalization,
)

from uxarray.utils.numba import is_numba_function_cached


from uxarray.conventions import ugrid

from xarray.core.utils import UncachedAccessor

from warnings import warn

import cartopy.crs as ccrs

import copy


from uxarray.constants import INT_FILL_VALUE
from uxarray.grid.dual import construct_dual


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
        self._ds.assign_attrs({"source_grid_spec": self.source_grid_spec})

        # cached parameters for GeoDataFrame conversions
        self._gdf_cached_parameters = {
            "gdf": None,
            "periodic_elements": None,
            "projection": None,
            "non_nan_polygon_indices": None,
            "engine": None,
            "exclude_am": None,
            "antimeridian_face_indices": None,
        }

        # cached parameters for PolyCollection conversions
        self._poly_collection_cached_parameters = {
            "poly_collection": None,
            "periodic_elements": None,
            "projection": None,
            "corrected_to_original_faces": None,
            "non_nan_polygon_indices": None,
            "antimeridian_face_indices": None,
        }

        # cached parameters for LineCollection conversions
        self._line_collection_cached_parameters = {
            "line_collection": None,
            "periodic_elements": None,
            "projection": None,
        }

        self._raster_data_id = None

        # initialize cached data structures (nearest neighbor operations)
        self._ball_tree = None
        self._kd_tree = None

        # flag to track if coordinates are normalized
        self._normalized = None

        # set desired longitude range to [-180, 180]
        _set_desired_longitude_range(self._ds)

    # declare plotting accessor
    plot = UncachedAccessor(GridPlotAccessor)

    # declare subset accessor
    subset = UncachedAccessor(GridSubsetAccessor)

    # declare cross section accessor
    cross_section = UncachedAccessor(GridCrossSectionAccessor)

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
            source_grid_spec, lon_name, lat_name = _parse_grid_type(dataset)
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
            elif source_grid_spec == "ICON":
                grid_ds, source_dims_dict = _read_icon(dataset, use_dual=use_dual)
            elif source_grid_spec == "Structured":
                grid_ds = _read_structured_grid(dataset[lon_name], dataset[lat_name])
                source_dims_dict = {"n_face": (lon_name, lat_name)}
            elif source_grid_spec == "Shapefile":
                raise ValueError(
                    "Use ux.Grid.from_geodataframe(<shapefile_name) instead"
                )
            else:
                raise ValueError("Unsupported Grid Format")
        else:
            # custom source grid spec is provided
            source_grid_spec = kwargs.get("source_grid_spec", None)
            grid_ds = dataset
            source_dims_dict = {}

        return cls(grid_ds, source_grid_spec, source_dims_dict)

    @classmethod
    def from_file(
        cls,
        filename: str,
        backend: Optional[str] = "geopandas",
        **kwargs,
    ):
        """Constructs a ``Grid`` object from a using the read_file method with
        a specified backend.

        Parameters
        ----------
        filename : str
            Path to grid file
        backend : str, default='geopandas'
            Backend to use to read the file, xarray or geopandas.

        Examples
        --------
        >>> import uxarray as ux
        >>> grid = ux.Grid.from_file("path/to/file.shp", backend="geopandas")

        Note
        ----
        All formats supported by `geopandas.read_file` can be used.
        See more at: https://geopandas.org/en/stable/docs/reference/api/geopandas.read_file.html#geopandas-read-file
        """

        # determine grid/mesh specification
        if backend == "geopandas":
            if str(filename).endswith(".shp"):
                source_grid_spec = "Shapefile"
            elif str(filename).endswith(".geojson"):
                source_grid_spec = "GeoJSON"
            else:
                source_grid_spec = "OtherGeoFormat"

            grid_ds, source_dims_dict = _read_geodataframe(filename)

        elif backend == "xarray":
            grid_ds, source_dims_dict = cls.from_dataset(filename)

        else:
            raise ValueError("Backend not supported")

        return cls(grid_ds, source_grid_spec, source_dims_dict)

    @classmethod
    def from_points(
        cls,
        points,
        method="spherical_delaunay",
        boundary_points=None,
        **kwargs,
    ):
        """Create a grid from unstructured points.

        This class method generates connectivity information based on the provided points.
        Depending on the chosen `method`, it constructs either a spherical Voronoi diagram
        or a spherical Delaunay triangulation. When using the Delaunay method, `boundary_points`
        can be specified to exclude triangles that span over defined holes in the data.

        Parameters
        ----------
        points : sequence of array-like
            The input points to generate connectivity from.

            - If `len(points) == 2`, `points` should be `[longitude, latitude]` in degrees,
              where each is an array-like of shape `(N,)`.
            - If `len(points) == 3`, `points` should be `[x, y, z]` coordinates,
              where each is an array-like of shape `(N,)`.

        method : str, optional
            The method to generate connectivity information. Supported methods are:

            - `'spherical_voronoi'`: Constructs a spherical Voronoi diagram.
            - `'spherical_delaunay'`: Constructs a spherical Delaunay triangulation.
            - `'regional_delaunay'`: Constructs a regional Delaunay triangulation.

            Default is `'spherical_delaunay'`.

        boundary_points : array-like of int, optional
            Indices of points that lie on a defined boundary. These are used to exclude
            Delaunay triangles that span over holes in the data. This parameter is only
            applicable when `method` is `'spherical_delaunay'`.

            Default is `None`.

        **kwargs
            Additional keyword arguments to pass to the underlying connectivity generation
            functions (`_spherical_voronoi_from_points` or `_spherical_delaunay_from_points`).

        Returns
        -------
        Grid
            An instance of a Grid created from the points
        """
        _points = prepare_points(points, normalize=True)

        if method == "spherical_voronoi":
            ds = _spherical_voronoi_from_points(_points, **kwargs)
        elif method == "spherical_delaunay":
            ds = _spherical_delaunay_from_points(_points, boundary_points)
        elif method == "regional_delaunay":
            ds = _regional_delaunay_from_points(_points, boundary_points)
        else:
            raise ValueError(
                f"Unsupported method '{method}'. Expected one of ['spherical_voronoi', 'spherical_delaunay']."
            )

        return cls.from_dataset(dataset=ds, source_grid_spec=method)

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
        To construct a UGRID-compliant grid, the user must provide at least ``node_lon``, ``node_lat`` and ``face_node_connectivity``

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

        Examples
        --------
        >>> import uxarray as ux
        >>> node_lon, node_lat, face_node_connectivity, fill_value = ...
        >>> uxgrid = ux.Grid.from_ugrid(
        ...     node_lon, node_lat, face_node_connectivity, fill_value
        ... )
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
    def from_structured(
        cls, ds: xr.Dataset = None, lon=None, lat=None, tol: Optional[float] = 1e-10
    ):
        """
        Converts a structured ``xarray.Dataset`` or longitude and latitude coordinates into an unstructured ``uxarray.Grid``.

        This class method provides flexibility in converting structured grid data into an unstructured `uxarray.UxDataset`.
        Users can either supply an existing structured `xarray.Dataset` or provide longitude and latitude coordinates
        directly.

        Parameters
        ----------
        ds : xr.Dataset, optional
            The structured `xarray.Dataset` to convert. If provided, the dataset must adhere to the
            Climate and Forecast (CF) Metadata Conventions

        lon : array-like, optional
            Longitude coordinates of the structured grid. Required if `ds` is not provided.
            Should be a one-dimensional or two-dimensional array following CF conventions.

        lat : array-like, optional
            Latitude coordinates of the structured grid. Required if `ds` is not provided.
            Should be a one-dimensional or two-dimensional array following CF conventions.

        tol : float, optional
            Tolerance for considering nodes as identical when constructing the grid from longitude and latitude.
            Default is `1e-10`.

        Returns
        -------
        Grid
            An instance of ``uxarray.Grid``
        """
        if ds is not None:
            return cls.from_dataset(ds)
        if lon is not None and lat is not None:
            grid_ds = _read_structured_grid(lon, lat, tol)
            return cls.from_dataset(
                grid_ds, source_dims_dict=None, source_grid_spec="structured"
            )
        else:
            raise ValueError(
                "No input dataset or latitude and longitude values specified."
            )

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

    def validate(self, check_duplicates=True):
        """Validates the current ``Grid``, checking for Duplicate Nodes,
        Present Connectivity, and Non-Zero Face Areas.

        Raises
        ------
        RuntimeError
            If unsupported grid type provided
        """
        # If the mesh file is loaded correctly, we have the underlying file format as UGRID
        # Test if the file is a valid ugrid file format or not
        print("Validating the mesh...")

        # call the check_connectivity and check_duplicate_nodes functions from validation.py
        checkDN = _check_duplicate_nodes(self) if check_duplicates else True
        check_C = _check_connectivity(self)
        check_A = _check_area(self)

        if checkDN and check_C and check_A:
            print("Mesh validation successful.")
            return True
        else:
            raise RuntimeError("Mesh validation failed.")

    def construct_face_centers(self, method="cartesian average"):
        """Constructs face centers, this method provides users direct control
        of the method for constructing the face centers, the default method is
        "cartesian average", but a more accurate method is "welzl" that is
        based on the recursive Welzl algorithm. It must be noted that this
        method can override the parsed/recompute the original parsed face
        centers.

        Parameters
        ----------
        method : str, default="cartesian average"
            Supported methods are "cartesian average" and "welzl"

        Returns
        -------
        None
            This method constructs the face_lon and face_lat attributes for the grid object.

        Usage
        -----
        >>> import uxarray as ux
        >>> uxgrid = ux.open_grid("GRID_FILE_NAME")
        >>> face_lat = uxgrid.construct_face_center(method="welzl")
        """
        if method == "cartesian average":
            _populate_face_centroids(self, repopulate=True)
        elif method == "welzl":
            _populate_face_centerpoints(self, repopulate=True)
        else:
            raise ValueError(
                f"Unknown method for face center calculation. Expected one of ['cartesian average', 'welzl'] but received {method}"
            )

    def __repr__(self):
        """Constructs a string representation of the contents of a ``Grid``."""

        from uxarray.conventions import descriptors

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

    def _repr_html_(self) -> str:
        if OPTIONS["display_style"] == "text":
            return f"<pre>{escape(repr(self))}</pre>"
        return grid_repr(self)

    def __getitem__(self, item):
        """Implementation of getitem operator for indexing a grid to obtain
        variables.

        Examples
        --------
        >>> uxgrid["face_node_connectivity"]
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
    def descriptors(self) -> set:
        """Names of all descriptor variables."""
        from uxarray.conventions.descriptors import DESCRIPTOR_NAMES

        return set([desc for desc in DESCRIPTOR_NAMES if desc in self._ds])

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

    @n_nodes_per_face.setter
    def n_nodes_per_face(self, value):
        """Setter for ``n_nodes_per_face``"""
        assert isinstance(value, xr.DataArray)
        self._ds["n_nodes_per_face"] = value

    @property
    def node_lon(self) -> xr.DataArray:
        """Longitude of each node in degrees.

        Dimensions: ``(n_node, )``
        """
        if "node_lon" not in self._ds:
            _set_desired_longitude_range(self._ds)
            _populate_node_latlon(self)
        return self._ds["node_lon"]

    @node_lon.setter
    def node_lon(self, value):
        """Setter for ``node_lon``"""
        assert isinstance(value, xr.DataArray)
        self._ds["node_lon"] = value

    @property
    def node_lat(self) -> xr.DataArray:
        """Latitude of each node in degrees.

        Dimensions: ``(n_node, )``
        """
        if "node_lat" not in self._ds:
            _set_desired_longitude_range(self._ds)
            _populate_node_latlon(self)
        return self._ds["node_lat"]

    @node_lat.setter
    def node_lat(self, value):
        """Setter for ``node_lat``"""
        assert isinstance(value, xr.DataArray)
        self._ds["node_lat"] = value

    @property
    def node_x(self) -> xr.DataArray:
        """Cartesian x location of each node in meters.

        Dimensions: ``(n_node, )``
        """
        if "node_x" not in self._ds:
            _populate_node_xyz(self)

        return self._ds["node_x"]

    @node_x.setter
    def node_x(self, value):
        """Setter for ``node_x``"""
        assert isinstance(value, xr.DataArray)
        self._ds["node_x"] = value

    @property
    def node_y(self) -> xr.DataArray:
        """Cartesian y location of each node in meters.

        Dimensions: ``(n_node, )``
        """
        if "node_y" not in self._ds:
            _populate_node_xyz(self)
        return self._ds["node_y"]

    @node_y.setter
    def node_y(self, value):
        """Setter for ``node_y``"""
        assert isinstance(value, xr.DataArray)
        self._ds["node_y"] = value

    @property
    def node_z(self) -> xr.DataArray:
        """Cartesian z location of each node in meters.

        Dimensions: ``(n_node, )``
        """
        if "node_z" not in self._ds:
            _populate_node_xyz(self)
        return self._ds["node_z"]

    @node_z.setter
    def node_z(self, value):
        """Setter for ``node_z``"""
        assert isinstance(value, xr.DataArray)
        self._ds["node_z"] = value

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

    @edge_lon.setter
    def edge_lon(self, value):
        """Setter for ``edge_lon``"""
        assert isinstance(value, xr.DataArray)
        self._ds["edge_lon"] = value

    @property
    def edge_lat(self) -> xr.DataArray:
        """Latitude of the center of each edge in degrees.

        Dimensions: ``(n_edge, )``
        """
        if "edge_lat" not in self._ds:
            _populate_edge_centroids(self)
        _set_desired_longitude_range(self._ds)
        return self._ds["edge_lat"]

    @edge_lat.setter
    def edge_lat(self, value):
        """Setter for ``edge_lat``"""
        assert isinstance(value, xr.DataArray)
        self._ds["edge_lat"] = value

    @property
    def edge_x(self) -> xr.DataArray:
        """Cartesian x location of the center of each edge in meters.

        Dimensions: ``(n_edge, )``
        """
        if "edge_x" not in self._ds:
            _populate_edge_centroids(self)

        return self._ds["edge_x"]

    @edge_x.setter
    def edge_x(self, value):
        """Setter for ``edge_x``"""
        assert isinstance(value, xr.DataArray)
        self._ds["edge_x"] = value

    @property
    def edge_y(self) -> xr.DataArray:
        """Cartesian y location of the center of each edge in meters.

        Dimensions: ``(n_edge, )``
        """
        if "edge_y" not in self._ds:
            _populate_edge_centroids(self)
        return self._ds["edge_y"]

    @edge_y.setter
    def edge_y(self, value):
        """Setter for ``edge_y``"""
        assert isinstance(value, xr.DataArray)
        self._ds["edge_y"] = value

    @property
    def edge_z(self) -> xr.DataArray:
        """Cartesian z location of the center of each edge in meters.

        Dimensions: ``(n_edge, )``
        """
        if "edge_z" not in self._ds:
            _populate_edge_centroids(self)
        return self._ds["edge_z"]

    @edge_z.setter
    def edge_z(self, value):
        """Setter for ``edge_z``"""
        assert isinstance(value, xr.DataArray)
        self._ds["edge_z"] = value

    @property
    def face_lon(self) -> xr.DataArray:
        """Longitude of the center of each face in degrees.

        Dimensions: ``(n_face, )``
        """
        if "face_lon" not in self._ds:
            _populate_face_centroids(self)
            _set_desired_longitude_range(self._ds)
        return self._ds["face_lon"]

    @face_lon.setter
    def face_lon(self, value):
        """Setter for ``face_lon``"""
        assert isinstance(value, xr.DataArray)
        self._ds["face_lon"] = value

    @property
    def face_lat(self) -> xr.DataArray:
        """Latitude of the center of each face in degrees.

        Dimensions: ``(n_face, )``
        """
        if "face_lat" not in self._ds:
            _populate_face_centroids(self)
            _set_desired_longitude_range(self._ds)

        return self._ds["face_lat"]

    @face_lat.setter
    def face_lat(self, value):
        """Setter for ``face_lat``"""
        assert isinstance(value, xr.DataArray)
        self._ds["face_lat"] = value

    @property
    def face_x(self) -> xr.DataArray:
        """Cartesian x location of the center of each face in meters.

        Dimensions: ``(n_face, )``
        """
        if "face_x" not in self._ds:
            _populate_face_centroids(self)

        return self._ds["face_x"]

    @face_x.setter
    def face_x(self, value):
        """Setter for ``face_x``"""
        assert isinstance(value, xr.DataArray)
        self._ds["face_x"] = value

    @property
    def face_y(self) -> xr.DataArray:
        """Cartesian y location of the center of each face in meters.

        Dimensions: ``(n_face, )``
        """
        if "face_y" not in self._ds:
            _populate_face_centroids(self)
        return self._ds["face_y"]

    @face_y.setter
    def face_y(self, value):
        """Setter for ``face_x``"""
        assert isinstance(value, xr.DataArray)
        self._ds["face_y"] = value

    @property
    def face_z(self) -> xr.DataArray:
        """Cartesian z location of the center of each face in meters.

        Dimensions: ``(n_face, )``
        """
        if "face_z" not in self._ds:
            _populate_face_centroids(self)
        return self._ds["face_z"]

    @face_z.setter
    def face_z(self, value):
        """Setter for ``face_z``"""
        assert isinstance(value, xr.DataArray)
        self._ds["face_z"] = value

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

    @face_node_connectivity.setter
    def face_node_connectivity(self, value):
        """Setter for ``face_node_connectivity``"""
        assert isinstance(value, xr.DataArray)
        self._ds["face_node_connectivity"] = value

    @property
    def edge_node_connectivity(self) -> xr.DataArray:
        """Indices of the two nodes that make up each edge.

        Dimensions: ``(n_edge, two)``

        Nodes are in arbitrary order.
        """
        if "edge_node_connectivity" not in self._ds:
            _populate_edge_node_connectivity(self)

        return self._ds["edge_node_connectivity"]

    @edge_node_connectivity.setter
    def edge_node_connectivity(self, value):
        """Setter for ``edge_node_connectivity``"""
        assert isinstance(value, xr.DataArray)
        self._ds["edge_node_connectivity"] = value

    @property
    def edge_node_x(self) -> xr.DataArray:
        """Cartesian x location for the two nodes that make up every edge.

        Dimensions: ``(n_edge, two)``
        """

        if "edge_node_x" not in self._ds:
            _edge_node_x = self.node_x.values[self.edge_node_connectivity.values]

            self._ds["edge_node_x"] = xr.DataArray(
                data=_edge_node_x,
                dims=["n_edge", "two"],
            )

        return self._ds["edge_node_x"]

    @property
    def edge_node_y(self) -> xr.DataArray:
        """Cartesian y location for the two nodes that make up every edge.

        Dimensions: ``(n_edge, two)``
        """

        if "edge_node_y" not in self._ds:
            _edge_node_y = self.node_y.values[self.edge_node_connectivity.values]

            self._ds["edge_node_y"] = xr.DataArray(
                data=_edge_node_y,
                dims=["n_edge", "two"],
            )

        return self._ds["edge_node_y"]

    @property
    def edge_node_z(self) -> xr.DataArray:
        """Cartesian z location for the two nodes that make up every edge.

        Dimensions: ``(n_edge, two)``
        """

        if "edge_node_z" not in self._ds:
            _edge_node_z = self.node_z.values[self.edge_node_connectivity.values]

            self._ds["edge_node_z"] = xr.DataArray(
                data=_edge_node_z,
                dims=["n_edge", "two"],
            )

        return self._ds["edge_node_z"]

    @property
    def node_node_connectivity(self) -> xr.DataArray:
        """Indices of the nodes that surround each node."""
        if "node_node_connectivity" not in self._ds:
            raise NotImplementedError(
                "Construction of `node_node_connectivity` not yet supported."
            )
        return self._ds["node_node_connectivity"]

    @node_node_connectivity.setter
    def node_node_connectivity(self, value):
        """Setter for ``node_node_connectivity``"""
        assert isinstance(value, xr.DataArray)
        self._ds["node_node_connectivity"] = value

    @property
    def face_edge_connectivity(self) -> xr.DataArray:
        """Indices of the edges that surround each face.

        Dimensions: ``(n_face, n_max_face_edges)``
        """
        if "face_edge_connectivity" not in self._ds:
            _populate_face_edge_connectivity(self)

        return self._ds["face_edge_connectivity"]

    @face_edge_connectivity.setter
    def face_edge_connectivity(self, value):
        """Setter for ``face_edge_connectivity``"""
        assert isinstance(value, xr.DataArray)
        self._ds["face_edge_connectivity"] = value

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

    @edge_edge_connectivity.setter
    def edge_edge_connectivity(self, value):
        """Setter for ``edge_edge_connectivity``"""
        assert isinstance(value, xr.DataArray)
        self._ds["edge_edge_connectivity"] = value

    @property
    def node_edge_connectivity(self) -> xr.DataArray:
        """Indices of the edges that surround each node."""
        if "node_edge_connectivity" not in self._ds:
            raise NotImplementedError(
                "Construction of `node_edge_connectivity` not yet supported."
            )

        return self._ds["node_edge_connectivity"]

    @node_edge_connectivity.setter
    def node_edge_connectivity(self, value):
        """Setter for ``node_edge_connectivity``"""
        assert isinstance(value, xr.DataArray)
        self._ds["node_edge_connectivity"] = value

    @property
    def face_face_connectivity(self) -> xr.DataArray:
        """Indices of the faces that surround each face.

        Dimensions ``(n_face, n_max_face_faces)``
        """
        if "face_face_connectivity" not in self._ds:
            _populate_face_face_connectivity(self)

        return self._ds["face_face_connectivity"]

    @face_face_connectivity.setter
    def face_face_connectivity(self, value):
        """Setter for ``face_face_connectivity``"""
        assert isinstance(value, xr.DataArray)
        self._ds["face_face_connectivity"] = value

    @property
    def edge_face_connectivity(self) -> xr.DataArray:
        """Indices of the faces that saddle each edge.

        Dimensions ``(n_edge, two)``
        """
        if "edge_face_connectivity" not in self._ds:
            _populate_edge_face_connectivity(self)

        return self._ds["edge_face_connectivity"]

    @edge_face_connectivity.setter
    def edge_face_connectivity(self, value):
        """Setter for ``edge_face_connectivity``"""
        assert isinstance(value, xr.DataArray)
        self._ds["edge_face_connectivity"] = value

    @property
    def node_face_connectivity(self) -> xr.DataArray:
        """Indices of the faces that surround each node.

        Dimensions ``(n_node, n_max_node_faces)``
        """
        if "node_face_connectivity" not in self._ds:
            _populate_node_face_connectivity(self)

        return self._ds["node_face_connectivity"]

    @node_face_connectivity.setter
    def node_face_connectivity(self, value):
        """Setter for ``node_face_connectivity``"""
        assert isinstance(value, xr.DataArray)
        self._ds["node_face_connectivity"] = value

    @property
    def edge_node_distances(self):
        """Distances between the two nodes that surround each edge in degrees.

        Dimensions ``(n_edge, )``
        """
        if "edge_node_distances" not in self._ds:
            _populate_edge_node_distances(self)
        return self._ds["edge_node_distances"]

    @edge_node_distances.setter
    def edge_node_distances(self, value):
        """Setter for ``edge_node_distances``"""
        assert isinstance(value, xr.DataArray)
        self._ds["edge_node_distances"] = value

    @property
    def edge_face_distances(self):
        """Distances between the centers of the faces that saddle each edge in
        degrees.

        Dimensions ``(n_edge, )``
        """
        if "edge_face_distances" not in self._ds:
            _populate_edge_face_distances(self)
        return self._ds["edge_face_distances"]

    @edge_face_distances.setter
    def edge_face_distances(self, value):
        """Setter for ``edge_face_distances``"""
        assert isinstance(value, xr.DataArray)
        self._ds["edge_face_distances"] = value

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

    @face_areas.setter
    def face_areas(self, value):
        """Setter for ``face_areas``"""
        assert isinstance(value, xr.DataArray)
        self._ds["face_areas"] = value

    @property
    def bounds(self):
        """Latitude Longitude Bounds for each Face in radians.

        Dimensions ``(n_face", two, two)``
        """
        if "bounds" not in self._ds:
            if not is_numba_function_cached(compute_temp_latlon_array):
                warn(
                    "Necessary functions for computing the bounds of each face are not yet compiled with Numba. "
                    "This initial execution will be significantly longer.",
                    RuntimeWarning,
                )

            _populate_bounds(self)

        return self._ds["bounds"]

    @bounds.setter
    def bounds(self, value):
        """Setter for ``bounds``"""
        assert isinstance(value, xr.DataArray)
        self._ds["bounds"] = value

    @property
    def face_bounds_lon(self):
        """Longitude bounds for each face in degrees."""

        if "face_bounds_lon" not in self._ds:
            bounds = self.bounds.values

            bounds_deg = np.rad2deg(bounds[:, 1, :])
            bounds_normalized = (bounds_deg + 180.0) % 360.0 - 180.0
            bounds_lon = bounds_normalized
            mask_zero = (bounds_lon[:, 0] == 0) & (bounds_lon[:, 1] == 0)
            # for faces that span all longitudes (i.e. pole faces)
            bounds_lon[mask_zero] = [-180.0, 180.0]
            self._ds["face_bounds_lon"] = xr.DataArray(
                data=bounds_lon,
                dims=["n_face", "min_max"],
            )

        return self._ds["face_bounds_lon"]

    @property
    def face_bounds_lat(self):
        """Latitude bounds for each face in degrees."""
        if "face_bounds_lat" not in self._ds:
            bounds = self.bounds.values
            bounds_lat = np.sort(np.rad2deg(bounds[:, 0, :]), axis=-1)
            self._ds["face_bounds_lat"] = xr.DataArray(
                data=bounds_lat,
                dims=["n_face", "min_max"],
            )
        return self._ds["face_bounds_lat"]

    @property
    def face_jacobian(self):
        """Declare face_jacobian as a property."""
        if self._face_jacobian is None:
            _ = self.face_areas
        return self._face_jacobian

    @property
    def boundary_edge_indices(self):
        """Indices of edges that border regions not covered by any geometry
        (holes) in a partial grid."""
        if "boundary_edge_indices" not in self._ds:
            self._ds["boundary_edge_indices"] = _construct_boundary_edge_indices(
                self.edge_face_connectivity.values
            )
        return self._ds["boundary_edge_indices"]

    @boundary_edge_indices.setter
    def boundary_edge_indices(self, value):
        """Setter for ``boundary_edge_indices``"""
        assert isinstance(value, xr.DataArray)
        self._ds["boundary_edge_indices"] = value

    @property
    def boundary_node_indices(self):
        """Indices of nodes that border regions not covered by any geometry
        (holes) in a partial grid."""
        if "boundary_node_indices" not in self._ds:
            raise ValueError

        return self._ds["boundary_node_indices"]

    @boundary_node_indices.setter
    def boundary_node_indices(self, value):
        """Setter for ``boundary_node_indices``"""
        assert isinstance(value, xr.DataArray)
        self._ds["boundary_node_indices"] = value

    @property
    def boundary_face_indices(self):
        """Indices of faces that border regions not covered by any geometry
        (holes) in a partial grid."""
        if "boundary_face_indices" not in self._ds:
            boundaries = np.unique(
                self.node_face_connectivity[
                    self.boundary_node_indices.values
                ].data.ravel()
            )
            boundaries = boundaries[boundaries != INT_FILL_VALUE]
            self._ds["boundary_face_indices"] = xr.DataArray(data=boundaries)

        return self._ds["boundary_face_indices"]

    @boundary_face_indices.setter
    def boundary_face_indices(self, value):
        """Setter for ``boundary_face_indices``"""
        assert isinstance(value, xr.DataArray)
        self._ds["boundary_face_indices"] = value

    @property
    def triangular(self):
        """Boolean indicated whether the Grid is strictly composed of
        triangular faces."""
        return self.n_max_face_nodes == 3

    @property
    def partial_sphere_coverage(self):
        """Boolean indicated whether the Grid partial covers the unit sphere
        (i.e. contains holes)"""
        return self.boundary_edge_indices.size != 0

    @property
    def global_sphere_coverage(self):
        """Boolean indicated whether the Grid completely covers the unit sphere
        (i.e. contains no holes)"""
        return not self.partial_sphere_coverage

    def chunk(self, n_node="auto", n_edge="auto", n_face="auto"):
        """Converts all arrays to dask arrays with given chunks across grid
        dimensions in-place.

        Non-dask arrays will be converted to dask arrays. Dask arrays will be chunked to the given chunk size.

        Parameters
        ----------
        n_node : int, tuple
            How to chunk node variables. Must be one of the following forms:

            - A blocksize like 1000.
            - A blockshape like (1000, 1000).
            - Explicit sizes of all blocks along all dimensions like
              ((1000, 1000, 500), (400, 400)).
            - A size in bytes, like "100 MiB" which will choose a uniform
              block-like shape
            - The word "auto" which acts like the above, but uses a configuration
              value ``array.chunk-size`` for the chunk size

            -1 or None as a blocksize indicate the size of the corresponding
            dimension.

        n_edge : int, tuple
            How to chunk edge variables. Must be one of the following forms:

            - A blocksize like 1000.
            - A blockshape like (1000, 1000).
            - Explicit sizes of all blocks along all dimensions like
              ((1000, 1000, 500), (400, 400)).
            - A size in bytes, like "100 MiB" which will choose a uniform
              block-like shape
            - The word "auto" which acts like the above, but uses a configuration
              value ``array.chunk-size`` for the chunk size

            -1 or None as a blocksize indicate the size of the corresponding
            dimension.

        n_face : int, tuple
            How to chunk face variables. Must be one of the following forms:

            - A blocksize like 1000.
            - A blockshape like (1000, 1000).
            - Explicit sizes of all blocks along all dimensions like
              ((1000, 1000, 500), (400, 400)).
            - A size in bytes, like "100 MiB" which will choose a uniform
              block-like shape
            - The word "auto" which acts like the above, but uses a configuration
              value ``array.chunk-size`` for the chunk size

            -1 or None as a blocksize indicate the size of the corresponding
            dimension.
        """

        grid_var_names = self.coordinates | self.connectivity | self.descriptors

        for var_name in grid_var_names:
            grid_var = getattr(self, var_name)

            if "n_node" in grid_var.dims:
                setattr(self, var_name, grid_var.chunk(chunks={"n_node": n_node}))
            elif "n_edge" in grid_var.dims:
                setattr(self, var_name, grid_var.chunk(chunks={"n_edge": n_edge}))
            elif "n_face" in grid_var.dims:
                setattr(self, var_name, grid_var.chunk(chunks={"n_face": n_face}))
            else:
                setattr(self, var_name, grid_var.chunk())

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

        >>> grid = ux.open_dataset(
        ...     "/home/jain/uxarray/test/meshfiles/ugrid/outCSne30/outCSne30.ug"
        ... )


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

    def normalize_cartesian_coordinates(self):
        """Normalizes Cartesian coordinates."""

        if _check_normalization(self):
            # check if coordinates are already normalized
            return

        if "node_x" in self._ds:
            # normalize node coordinates
            node_x, node_y, node_z = _normalize_xyz(
                self.node_x.values, self.node_y.values, self.node_z.values
            )
            self.node_x.data = node_x
            self.node_y.data = node_y
            self.node_z.data = node_z
        if "edge_x" in self._ds:
            # normalize edge coordinates
            edge_x, edge_y, edge_z = _normalize_xyz(
                self.edge_x.values, self.edge_y.values, self.edge_z.values
            )
            self.edge_x.data = edge_x
            self.edge_y.data = edge_y
            self.edge_z.data = edge_z
        if "face_x" in self._ds:
            # normalize face coordinates
            face_x, face_y, face_z = _normalize_xyz(
                self.face_x.values, self.face_y.values, self.face_z.values
            )
            self.face_x.data = face_x
            self.face_y.data = face_y
            self.face_z.data = face_z

    def to_xarray(self, grid_format: Optional[str] = "ugrid"):
        """Returns an ``xarray.Dataset`` with the variables stored under the
        ``Grid`` encoded in a specific grid format.

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
        periodic_elements: Optional[str] = "exclude",
        projection: Optional[ccrs.Projection] = None,
        cache: Optional[bool] = True,
        override: Optional[bool] = False,
        engine: Optional[str] = "spatialpandas",
        exclude_antimeridian: Optional[bool] = None,
        return_non_nan_polygon_indices: Optional[bool] = False,
        exclude_nan_polygons: Optional[bool] = True,
        **kwargs,
    ):
        """Constructs a ``GeoDataFrame`` consisting of polygons representing
        the faces of the current ``Grid``

        Periodic polygons (i.e. those that cross the antimeridian) can be handled using the ``periodic_elements``
        parameter. Setting ``periodic_elements='split'`` will split each periodic polygon along the antimeridian.
        Setting ``periodic_elements='exclude'`` will exclude any periodic polygon from the computed GeoDataFrame.
        Setting ``periodic_elements='ignore'`` will compute the GeoDataFrame assuming no corrections are needed, which
        is best used for grids that do not initially include any periodic polygons.


        Parameters
        ----------
        periodic_elements : str, optional
            Method for handling periodic elements. One of ['exclude', 'split', or 'ignore']:
            - 'exclude': Periodic elements will be identified and excluded from the GeoDataFrame
            - 'split': Periodic elements will be identified and split using the ``antimeridian`` package
            - 'ignore': No processing will be applied to periodic elements.
        projection: ccrs.Projection, optional
            Geographic projection used to transform polygons. Only supported when periodic_elements is set to
            'ignore' or 'exclude'
        cache: bool, optional
            Flag used to select whether to cache the computed GeoDataFrame
        override: bool, optional
            Flag used to select whether to ignore any cached GeoDataFrame
        engine: str, optional
            Selects what library to use for creating a GeoDataFrame. One of ['spatialpandas', 'geopandas']. Defaults
            to spatialpandas
        exclude_antimeridian: bool, optional
            Flag used to select whether to exclude polygons that cross the antimeridian (Will be deprecated)
        return_non_nan_polygon_indices: bool, optional
            Flag used to select whether to return the indices of any non-nan polygons
        exclude_nan_polygons: bool, optional
            Flag to select whether to exclude any nan polygons


        Returns
        -------
        gdf : spatialpandas.GeoDataFrame or geopandas.GeoDataFrame
            The output ``GeoDataFrame`` with a filled out "geometry" column of polygons.
        """

        if engine not in ["spatialpandas", "geopandas"]:
            raise ValueError(
                f"Invalid engine. Expected one of ['spatialpandas', 'geopandas'] but received {engine}"
            )

        # if project is false, projection is only used for determining central coordinates
        project = kwargs.get("project", True)

        if projection and project:
            if periodic_elements == "split":
                raise ValueError(
                    "Setting ``periodic_elements='split'`` is not supported when a "
                    "projection is provided."
                )

        if exclude_antimeridian is not None:
            warn(
                DeprecationWarning(
                    "The parameter ``exclude_antimeridian`` will be deprecated in a future release. Please "
                    "use ``periodic_elements='exclude'`` or ``periodic_elements='split'`` instead."
                ),
                stacklevel=2,
            )
            if exclude_antimeridian:
                periodic_elements = "exclude"
            else:
                periodic_elements = "split"

        if periodic_elements not in ["ignore", "exclude", "split"]:
            raise ValueError(
                f"Invalid value for 'periodic_elements'. Expected one of ['exclude', 'split', 'ignore'] but received: {periodic_elements}"
            )

        if self._gdf_cached_parameters["gdf"] is not None:
            if (
                self._gdf_cached_parameters["periodic_elements"] != periodic_elements
                or self._gdf_cached_parameters["projection"] != projection
                or self._gdf_cached_parameters["engine"] != engine
            ):
                # cached GeoDataFrame has a different projection or periodic element handling method
                override = True

        if self._gdf_cached_parameters["gdf"] is not None and not override:
            # use cached PolyCollection
            if return_non_nan_polygon_indices:
                return self._gdf_cached_parameters["gdf"], self._gdf_cached_parameters[
                    "non_nan_polygon_indices"
                ]
            else:
                return self._gdf_cached_parameters["gdf"]

        # construct a GeoDataFrame with the faces stored as polygons as the geometry
        gdf, non_nan_polygon_indices = _grid_to_polygon_geodataframe(
            self, periodic_elements, projection, project, engine
        )

        if exclude_nan_polygons and non_nan_polygon_indices is not None:
            # exclude any polygons that contain NaN values
            gdf = GeoDataFrame({"geometry": gdf["geometry"][non_nan_polygon_indices]})

        if cache:
            self._gdf_cached_parameters["gdf"] = gdf
            self._gdf_cached_parameters["non_nan_polygon_indices"] = (
                non_nan_polygon_indices
            )
            self._gdf_cached_parameters["periodic_elements"] = periodic_elements
            self._gdf_cached_parameters["projection"] = projection
            self._gdf_cached_parameters["engine"] = engine

        if return_non_nan_polygon_indices:
            return gdf, non_nan_polygon_indices

        return gdf

    def to_polycollection(
        self,
        periodic_elements: Optional[str] = "exclude",
        projection: Optional[ccrs.Projection] = None,
        return_indices: Optional[bool] = False,
        cache: Optional[bool] = True,
        override: Optional[bool] = False,
        return_non_nan_polygon_indices: Optional[bool] = False,
        **kwargs,
    ):
        """Constructs a ``matplotlib.collections.PolyCollection``` consisting
        of polygons representing the faces of the current ``Grid``

        Parameters
        ----------
        periodic_elements : str, optional
            Method for handling periodic elements. One of ['exclude', 'split', or 'ignore']:
            - 'exclude': Periodic elements will be identified and excluded from the GeoDataFrame
            - 'split': Periodic elements will be identified and split using the ``antimeridian`` package
            - 'ignore': No processing will be applied to periodic elements.
        projection: ccrs.Projection
            Cartopy geographic projection to use
        return_indices: bool
            Flag to indicate whether to return the indices of corrected polygons, if any exist
        cache: bool
            Flag to indicate whether to cache the computed PolyCollection
        override: bool
            Flag to indicate whether to override a cached PolyCollection, if it exists
        **kwargs: dict
            Key word arguments to pass into the construction of a PolyCollection
        """

        if periodic_elements not in ["ignore", "exclude", "split"]:
            raise ValueError(
                f"Invalid value for 'periodic_elements'. Expected one of ['include', 'exclude', 'split'] but received: {periodic_elements}"
            )

        if self._poly_collection_cached_parameters["poly_collection"] is not None:
            if (
                self._poly_collection_cached_parameters["periodic_elements"]
                != periodic_elements
                or self._poly_collection_cached_parameters["projection"] != projection
            ):
                # cached PolyCollection has a different projection or periodic element handling method
                override = True

        if (
            self._poly_collection_cached_parameters["poly_collection"] is not None
            and not override
        ):
            # use cached PolyCollection
            if return_indices:
                return copy.deepcopy(
                    self._poly_collection_cached_parameters["poly_collection"]
                ), self._poly_collection_cached_parameters[
                    "corrected_to_original_faces"
                ]
            else:
                return copy.deepcopy(
                    self._poly_collection_cached_parameters["poly_collection"]
                )

        (
            poly_collection,
            corrected_to_original_faces,
        ) = _grid_to_matplotlib_polycollection(
            self, periodic_elements, projection, **kwargs
        )

        if cache:
            # cache PolyCollection, indices, and state
            self._poly_collection_cached_parameters["poly_collection"] = poly_collection
            self._poly_collection_cached_parameters["corrected_to_original_faces"] = (
                corrected_to_original_faces
            )
            self._poly_collection_cached_parameters["periodic_elements"] = (
                periodic_elements
            )
            self._poly_collection_cached_parameters["projection"] = projection

        if return_indices:
            return copy.deepcopy(poly_collection), corrected_to_original_faces
        else:
            return copy.deepcopy(poly_collection)

    def to_linecollection(
        self,
        periodic_elements: Optional[str] = "exclude",
        projection: Optional[ccrs.Projection] = None,
        cache: Optional[bool] = True,
        override: Optional[bool] = False,
        **kwargs,
    ):
        """Constructs a ``matplotlib.collections.LineCollection``` consisting
        of lines representing the edges of the current ``Grid``

        Parameters
        ----------
        periodic_elements : str, optional
            Method for handling periodic elements. One of ['exclude', 'split', or 'ignore']:
            - 'exclude': Periodic elements will be identified and excluded from the GeoDataFrame
            - 'split': Periodic elements will be identified and split using the ``antimeridian`` package
            - 'ignore': No processing will be applied to periodic elements.
        projection: ccrs.Projection
            Cartopy geographic projection to use
        cache: bool
            Flag to indicate whether to cache the computed PolyCollection
        override: bool
            Flag to indicate whether to override a cached PolyCollection, if it exists
        **kwargs: dict
            Key word arguments to pass into the construction of a PolyCollection
        """
        if periodic_elements not in ["ignore", "exclude", "split"]:
            raise ValueError(
                f"Invalid value for 'periodic_elements'. Expected one of ['ignore', 'exclude', 'split'] but received: {periodic_elements}"
            )

        if self._line_collection_cached_parameters["line_collection"] is not None:
            if (
                self._line_collection_cached_parameters["periodic_elements"]
                != periodic_elements
                or self._line_collection_cached_parameters["projection"] != projection
            ):
                override = True

            if not override:
                return self._line_collection_cached_parameters["line_collection"]

        line_collection = _grid_to_matplotlib_linecollection(
            grid=self,
            periodic_elements=periodic_elements,
            projection=projection,
            **kwargs,
        )

        if cache:
            self._line_collection_cached_parameters["line_collection"] = line_collection
            self._line_collection_cached_parameters["periodic_elements"] = (
                periodic_elements
            )
            self._line_collection_cached_parameters["periodic_elements"] = (
                periodic_elements
            )

        return line_collection

    def get_dual(self):
        """Compute the dual for a grid, which constructs a new grid centered
        around the nodes, where the nodes of the primal become the face centers
        of the dual, and the face centers of the primal become the nodes of the
        dual. Returns a new `Grid` object.

        Returns
        --------
        dual : Grid
            Dual Mesh Grid constructed
        """

        if _check_duplicate_nodes_indices(self):
            raise RuntimeError("Duplicate nodes found, cannot construct dual")

        # Get dual mesh node face connectivity
        dual_node_face_conn = construct_dual(grid=self)

        # Construct dual mesh
        dual = self.from_topology(
            self.face_lon.values, self.face_lat.values, dual_node_face_conn
        )

        return dual

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

    def get_edges_at_constant_latitude(self, lat: float, use_face_bounds: bool = False):
        """Identifies the indices of edges that intersect with a line of constant latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the cross-section, in degrees.
            Must be between -90.0 and 90.0
        use_face_bounds : bool, optional
            If True, uses the bounds of each face for computing intersections.

        Returns
        -------
        faces : numpy.ndarray
            An array of edge indices that intersect with the specified latitude.
        """

        if lat > 90.0 or lat < -90.0:
            raise ValueError(
                f"Latitude must be between -90 and 90 degrees. Received {lat}"
            )

        if use_face_bounds:
            raise NotImplementedError(
                "Computing the intersection using the spherical bounding box"
                "is not yet supported."
            )
        else:
            edges = constant_lat_intersections_no_extreme(
                lat, self.edge_node_z.values, self.n_edge
            )

        return edges.squeeze()

    def get_faces_at_constant_latitude(
        self,
        lat: float,
    ):
        """
        Identifies the indices of faces that intersect with a line of constant latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the cross-section, in degrees.
            Must be between -90.0 and 90.0

        Returns
        -------
        faces : numpy.ndarray
            An array of face indices that intersect with the specified latitude.
        """

        if lat > 90.0 or lat < -90.0:
            raise ValueError(
                f"Latitude must be between -90 and 90 degrees. Received {lat}"
            )

        faces = constant_lat_intersections_face_bounds(
            lat=lat,
            face_bounds_lat=self.face_bounds_lat.values,
        )
        return faces

    def get_edges_at_constant_longitude(
        self, lon: float, use_face_bounds: bool = False
    ):
        """
        Identifies the indices of edges that intersect with a line of constant longitude.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the cross-section, in degrees.
            Must be between -90.0 and 90.0
        use_face_bounds : bool, optional
            If True, uses the bounds of each face for computing intersections.

        Returns
        -------
        faces : numpy.ndarray
            An array of edge indices that intersect with the specified longitude.
        """

        if lon > 180.0 or lon < -180.0:
            raise ValueError(
                f"Longitude must be between -180 and 180 degrees. Received {lon}"
            )

        if use_face_bounds:
            raise NotImplementedError(
                "Computing the intersection using the spherical bounding box"
                "is not yet supported."
            )
        else:
            edges = constant_lon_intersections_no_extreme(
                lon, self.edge_node_x.values, self.edge_node_y.values, self.n_edge
            )
            return edges.squeeze()

    def get_faces_at_constant_longitude(self, lon: float):
        """
        Identifies the indices of faces that intersect with a line of constant longitude.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the cross-section, in degrees.
            Must be between -90.0 and 90.0

        Returns
        -------
        faces : numpy.ndarray
            An array of face indices that intersect with the specified longitude.
        """

        if lon > 180.0 or lon < -180.0:
            raise ValueError(
                f"Longitude must be between -180 and 180 degrees. Received {lon}"
            )

        faces = constant_lon_intersections_face_bounds(lon, self.face_bounds_lon.values)
        return faces
