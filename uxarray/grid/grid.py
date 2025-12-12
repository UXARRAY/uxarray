import copy
import os
from html import escape
from typing import Sequence
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core.options import OPTIONS
from xarray.core.utils import UncachedAccessor

from uxarray.constants import INT_FILL_VALUE
from uxarray.conventions import ugrid

# Import the utility function for opening datasets with fallback
from uxarray.core.utils import _open_dataset_with_fallback
from uxarray.cross_sections import GridCrossSectionAccessor
from uxarray.formatting_html import grid_repr
from uxarray.grid.area import get_all_face_area_from_coords
from uxarray.grid.bounds import _populate_face_bounds
from uxarray.grid.connectivity import (
    _populate_edge_face_connectivity,
    _populate_edge_node_connectivity,
    _populate_face_edge_connectivity,
    _populate_face_face_connectivity,
    _populate_n_nodes_per_face,
    _populate_node_edge_connectivity,
    _populate_node_face_connectivity,
)
from uxarray.grid.coordinates import (
    _populate_edge_centroids,
    _populate_face_centerpoints,
    _populate_face_centroids,
    _populate_node_latlon,
    _populate_node_xyz,
    _set_desired_longitude_range,
    points_atleast_2d_xyz,
    prepare_points,
)
from uxarray.grid.dual import construct_dual
from uxarray.grid.geometry import (
    _construct_boundary_edge_indices,
    _grid_to_matplotlib_linecollection,
    _grid_to_matplotlib_polycollection,
    _grid_to_polygon_geodataframe,
    _populate_antimeridian_face_indices,
    _populate_max_face_radius,
)
from uxarray.grid.intersections import (
    constant_lat_intersections_face_bounds,
    constant_lat_intersections_no_extreme,
    constant_lon_intersections_face_bounds,
    constant_lon_intersections_no_extreme,
    faces_within_lat_bounds,
    faces_within_lon_bounds,
)
from uxarray.grid.neighbors import (
    BallTree,
    KDTree,
    SpatialHash,
    _populate_edge_face_distances,
    _populate_edge_node_distances,
)
from uxarray.grid.point_in_face import _point_in_face_query
from uxarray.grid.utils import make_setter
from uxarray.grid.validation import (
    _check_area,
    _check_connectivity,
    _check_duplicate_nodes,
    _check_duplicate_nodes_indices,
    _check_normalization,
)
from uxarray.io._delaunay import (
    _regional_delaunay_from_points,
    _spherical_delaunay_from_points,
)
from uxarray.io._esmf import _read_esmf

# reader and writer imports
from uxarray.io._exodus import _encode_exodus, _read_exodus
from uxarray.io._fesom2 import _read_fesom2_asci, _read_fesom2_netcdf
from uxarray.io._geopandas import _read_geodataframe
from uxarray.io._geos import _read_geos_cs
from uxarray.io._healpix import _pixels_to_ugrid, _populate_healpix_boundaries
from uxarray.io._icon import _read_icon
from uxarray.io._mpas import _read_mpas
from uxarray.io._scrip import _encode_scrip, _read_scrip
from uxarray.io._structured import _read_structured_grid
from uxarray.io._topology import _read_topology
from uxarray.io._ugrid import (
    _encode_ugrid,
    _read_ugrid,
    _validate_minimum_ugrid,
)
from uxarray.io._vertices import _read_face_vertices
from uxarray.io._voronoi import _spherical_voronoi_from_points
from uxarray.io.utils import _parse_grid_type
from uxarray.plot.accessor import GridPlotAccessor
from uxarray.subset import GridSubsetAccessor


class Grid:
    """Represents a two-dimensional unstructured grid encoded following the
    UGRID conventions and provides grid-specific functionality.

    Can be used standalone to work with unstructured grids, or can be paired with either a ``ux.UxDataArray`` or
    ``ux.UxDataset`` and accessed through the ``.uxgrid`` attribute.

    For constructing a grid from non-UGRID datasets or other types of supported data, see our ``ux.open_grid`` method or
    specific class methods (``Grid.from_dataset``, ``Grid.from_face_verticies``, etc.)

    Note on Sphere Radius:
        All internal calculations use a unit sphere (radius=1.0). The physical sphere radius
        from the source grid is preserved in the ``sphere_radius`` property for scaling results.


    Parameters
    ----------
    grid_ds : xr.Dataset
        ``xarray.Dataset`` encoded in the UGRID conventions

    source_grid_spec : str, default="UGRID"
        Original unstructured grid format (i.e. UGRID, MPAS, etc.)

    source_dims_dict : dict, default={}
        Mapping of dimensions from the source dataset to their UGRID equivalent (i.e. {nCell : n_face})

    is_subset : bool, default=False
        Flag to mark if the grid is a subset or not

    inverse_indices: xr.Dataset, default=None
        A dataset of indices that correspond to the original grid, if the grid being constructed is a subset

    Examples
    --------

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
        source_grid_spec: str | None = None,
        source_dims_dict: dict | None = None,
        is_subset: bool = False,
        inverse_indices: xr.Dataset | None = None,
    ):
        # check if inputted dataset is a minimum representable 2D UGRID unstructured grid
        if source_grid_spec != "HEALPix":
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
        self._source_dims_dict = source_dims_dict or {}

        # source grid specification (i.e. UGRID, MPAS, SCRIP, etc.)
        self.source_grid_spec = source_grid_spec

        # internal xarray dataset for storing grid variables
        self._ds = grid_ds

        # source grid specification (i.e. UGRID, MPAS, SCRIP, etc.)
        self.source_grid_spec = source_grid_spec
        self._ds = self._ds.assign_attrs({"source_grid_spec": source_grid_spec})

        # initialize attributes
        self._antimeridian_face_indices = None
        self._ds.assign_attrs({"source_grid_spec": self.source_grid_spec})
        self._is_subset = is_subset

        self._inverse_indices = inverse_indices

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

        # Cached matplotlib data structures
        self._cached_poly_collection = None
        self._cached_line_collection = None

        self._raster_data_id = None

        # Cache for KDTrees
        self._kdtrees = {}

        # initialize cached data structures (nearest neighbor operations)
        self._ball_tree = None
        self._kd_tree = None
        self._spatialhash = None

        # flag to track if coordinates are normalized
        self._normalized = None

        # set desired longitude range to [-180, 180]
        _set_desired_longitude_range(self)

    # declare plotting accessor
    plot = UncachedAccessor(GridPlotAccessor)

    # declare subset accessor
    subset = UncachedAccessor(GridSubsetAccessor)

    # declare cross section accessor
    cross_section = UncachedAccessor(GridCrossSectionAccessor)

    @classmethod
    def from_dataset(cls, dataset, use_dual: bool | None = False, **kwargs):
        """Constructs a ``Grid`` object from a dataset.

        Parameters
        ----------
        dataset : xr.Dataset or path-like
            ``xarray.Dataset`` containing unstructured grid coordinates and connectivity variables or a directory
              containing ASCII files represents a FESOM2 grid.
        use_dual : bool, default=False
            When reading in MPAS formatted datasets, indicates whether to use the Dual Mesh
        """

        if isinstance(dataset, xr.Dataset):
            # determine grid/mesh specification
            if "source_grid_spec" not in kwargs:
                # parse to detect source grid spec
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
                    grid_ds = _read_structured_grid(
                        dataset[lon_name], dataset[lat_name]
                    )
                    source_dims_dict = {"n_face": (lon_name, lat_name)}
                elif source_grid_spec == "FESOM2":
                    grid_ds, source_dims_dict = _read_fesom2_netcdf(dataset)
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
        else:
            try:
                if os.path.isdir(dataset):
                    # FESOM2 ASCII directory.
                    grid_ds, source_dims_dict = _read_fesom2_asci(dataset)
                    source_grid_spec = "FESOM2"
                    return cls(grid_ds, source_grid_spec, source_dims_dict)
            except TypeError:
                raise ValueError("Unsupported Grid Format")

        return cls(
            grid_ds,
            source_grid_spec,
            source_dims_dict,
            is_subset=kwargs.get("is_subset", False),
            inverse_indices=kwargs.get("inverse_indices"),
        )

    @classmethod
    def from_file(
        cls,
        filename: str,
        backend: str | None = "geopandas",
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
            dataset = _open_dataset_with_fallback(filename, **kwargs)
            return cls.from_dataset(dataset)

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
                f"Unsupported method '{method}'. Expected one of ['spherical_voronoi', 'spherical_delaunay', 'regional_delaunay']."
            )

        return cls.from_dataset(dataset=ds, source_grid_spec=method)

    @classmethod
    def from_topology(
        cls,
        node_lon: np.ndarray,
        node_lat: np.ndarray,
        face_node_connectivity: np.ndarray,
        fill_value: int | float | None = None,
        start_index: int | None = 0,
        dims_dict: dict | None = None,
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
        fill_value: int | float | None
            Value used for padding connectivity variables when the maximum number of elements in a row is less than the maximum.
        start_index: int | None, default=0
            Start index (typically 0 or 1)
        dims_dict : dict | None
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
        cls, ds: xr.Dataset = None, lon=None, lat=None, tol: float | None = 1e-10
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
        face_vertices: list | tuple | np.ndarray,
        latlon: bool | None = True,
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

    @classmethod
    def from_healpix(cls, zoom: int, pixels_only: bool = True, nest: bool = True):
        """Constructs a ``Grid`` object representing a given HEALPix zoom level.

        Parameters
        ----------
        zoom : int
            Zoom level of HEALPix, with 12*zoom^4 representing the number of pixels (`n_face`)
        pixels_only : bool
            Whether to only compute pixels (`face_lon`, `face_lat`) or to also construct boundaries (`face_node_connectivity`, `node_lon`, `node_lat`)

        Returns
        -------
        Grid
            An instance of ``uxarray.Grid``
        """
        grid_ds = _pixels_to_ugrid(zoom, nest)

        if not pixels_only:
            _populate_healpix_boundaries(grid_ds)

        return cls.from_dataset(grid_ds, source_grid_spec="HEALPix")

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

        coord_heading = "Grid Coordinates (Spherical):\n"
        coords_str = ""
        for coord_name in list(
            [coord for coord in ugrid.SPHERICAL_COORDS if coord in self._ds]
        ):
            coords_str += f"  * {coord_name}: {getattr(self, coord_name).shape}\n"

        coords_str += "Grid Coordinates (Cartesian):\n"
        for coord_name in list(
            [coord for coord in ugrid.CARTESIAN_COORDS if coord in self._ds]
        ):
            coords_str += f"  * {coord_name}: {getattr(self, coord_name).shape}\n"

        connectivity_heading = "Grid Connectivity Variables:\n"
        connectivity_str = ""

        for conn_name in self.connectivity:
            connectivity_str += f"  * {conn_name}: {getattr(self, conn_name).shape}\n"

        descriptors_heading = "Grid Descriptor Variables:\n"
        descriptors_str = ""
        for descriptor_name in list(
            [desc for desc in descriptors.DESCRIPTOR_NAMES if desc in self._ds]
        ):
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
            CARTESIAN_COORD_NAMES,
            SPHERICAL_COORD_NAMES,
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

    # ==================================================================================================================
    # Dimension Properties
    # ==================================================================================================================

    @property
    def n_node(self) -> int:
        """Total number of nodes.

        Returns
        -------
        n_node : int
            The total number of nodes.
        """
        if "face_node_connectivity" not in self._ds:
            _ = self.face_node_connectivity
        return self._ds.sizes["n_node"]

    @property
    def n_edge(self) -> int:
        """Total number of edges.

        Returns
        -------
        n_edge : int
            The total number of edges.
        """
        if "edge_node_connectivity" not in self._ds:
            _populate_edge_node_connectivity(self)

        return self._ds.sizes["n_edge"]

    @property
    def n_face(self) -> int:
        """Total number of faces.

        Returns
        -------
        n_face : int
            The total number of faces.
        """
        return self._ds.sizes["n_face"]

    @property
    def n_max_face_nodes(self) -> int:
        """Maximum number of nodes defining a single face.

        For example, if the grid is composed entirely of triangular faces, the value would be 3.
        If the grid is composed of a mix of triangles and hexagons, the value would be 6.

        Returns
        -------
        n_max_face_nodes : int
            The maximum number of nodes that can define a face.
        """
        return self.face_node_connectivity.shape[1]

    @property
    def n_max_face_edges(self) -> int:
        """Maximum number of edges defining a single face.

        This is equivalent to :py:attr:`~uxarray.Grid.n_max_face_nodes`.

        Returns
        -------
        n_max_face_edges : int
            The maximum number of edges that can surround a face.
        """
        return self.face_edge_connectivity.shape[1]

    @property
    def n_max_face_faces(self) -> int:
        """Maximum number of neighboring faces surrounding a single face.

        Returns
        -------
        n_max_face_faces : int
            The maximum number of faces that can surround a face.
        """
        return self.face_face_connectivity.shape[1]

    @property
    def n_max_edge_edges(self) -> int:
        """Maximum number of edges surrounding a single edge.

        Returns
        -------
        n_max_edge_edges : int
            The maximum number of edges that can surround an edge.
        """
        return self.edge_edge_connectivity.shape[1]

    @property
    def n_max_node_faces(self) -> int:
        """Maximum number of faces surrounding a single node.

        Returns
        -------
        n_max_node_faces : int
            The maximum number of faces that can surround a node.
        """
        return self.node_face_connectivity.shape[1]

    @property
    def n_max_node_edges(self) -> int:
        """Maximum number of edges surrounding a single node.

        Returns
        -------
        n_max_node_edges : int
            The maximum number of edges that can surround a node.
        """
        return self.node_edge_connectivity.shape[1]

    @property
    def n_nodes_per_face(self) -> xr.DataArray:
        """An array containing the maximum number of nodes for each face.

        Returns
        -------
        n_nodes_per_face: :py:class:`xarray.DataArray`
            An array with shape (:py:attr:`~uxarray.Grid.n_face`,)
        """
        if "n_nodes_per_face" not in self._ds:
            _populate_n_nodes_per_face(self)

        return self._ds["n_nodes_per_face"]

    n_nodes_per_face = n_nodes_per_face.setter(make_setter("n_nodes_per_face"))

    # ==================================================================================================================
    # Coordinate Properties
    # ==================================================================================================================

    @property
    def node_lon(self) -> xr.DataArray:
        """Longitude coordinate of each node (in degrees).

        Values are expected to be in the range [-180.0, 180.0].

        Returns
        -------
        node_lon: :py:class:`xarray.DataArray`
            An array with shape (:py:attr:`~uxarray.Grid.n_node`,)
        """
        if "node_lon" not in self._ds:
            if self.source_grid_spec == "HEALPix":
                _populate_healpix_boundaries(self._ds)
            else:
                _set_desired_longitude_range(self)
                _populate_node_latlon(self)
        return self._ds["node_lon"]

    node_lon = node_lon.setter(make_setter("node_lon"))

    @property
    def node_lat(self) -> xr.DataArray:
        """Latitude coordinate of each node (in degrees).

        Values are expected to be in the range [-90.0, 90.0].

        Returns
        -------
        node_lat: :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_node`,)
        """
        if "node_lat" not in self._ds:
            if self.source_grid_spec == "HEALPix":
                _populate_healpix_boundaries(self)
            else:
                _set_desired_longitude_range(self)
                _populate_node_latlon(self)
        return self._ds["node_lat"]

    node_lat = node_lat.setter(make_setter("node_lat"))

    @property
    def node_x(self) -> xr.DataArray:
        """Cartesian x coordinate of each node (in meters).

        Returns
        -------
        node_x : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_node`,)
        """
        if "node_x" not in self._ds:
            _populate_node_xyz(self)

        return self._ds["node_x"]

    node_x = node_x.setter(make_setter("node_x"))

    @property
    def node_y(self) -> xr.DataArray:
        """Cartesian y coordinate of each node (in meters).

        Returns
        -------
        node_y : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_node`,)
        """
        if "node_y" not in self._ds:
            _populate_node_xyz(self)
        return self._ds["node_y"]

    node_y = node_y.setter(make_setter("node_y"))

    @property
    def node_z(self) -> xr.DataArray:
        """Cartesian z coordinate of each node (in meters).

        Returns
        -------
        node_z : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_node`,)
        """
        if "node_z" not in self._ds:
            _populate_node_xyz(self)
        return self._ds["node_z"]

    node_z = node_z.setter(make_setter("node_z"))

    @property
    def edge_lon(self) -> xr.DataArray:
        """Longitude coordinate of the center of each edge (in degrees).

        Values are expected to be in the range [-180.0, 180.0].

        Returns
        -------
        edge_lon : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_edge`,)
        """
        if "edge_lon" not in self._ds:
            _populate_edge_centroids(self)
            _set_desired_longitude_range(self)
        return self._ds["edge_lon"]

    edge_lon = edge_lon.setter(make_setter("edge_lon"))

    @property
    def edge_lat(self) -> xr.DataArray:
        """Latitude coordinate of the center of each edge (in degrees).

        Values are expected to be in the range [-90.0, 90.0].

        Returns
        -------
        edge_lat : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_edge`,)
        """
        if "edge_lat" not in self._ds:
            _populate_edge_centroids(self)
        _set_desired_longitude_range(self)
        return self._ds["edge_lat"]

    edge_lat = edge_lat.setter(make_setter("edge_lat"))

    @property
    def edge_x(self) -> xr.DataArray:
        """Cartesian x coordinate of the center of each edge (in meters).

        Returns
        -------
        edge_x : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_edge`,)
        """
        if "edge_x" not in self._ds:
            _populate_edge_centroids(self)

        return self._ds["edge_x"]

    edge_x = edge_x.setter(make_setter("edge_x"))

    @property
    def edge_y(self) -> xr.DataArray:
        """Cartesian y coordinate of the center of each edge (in meters).

        Returns
        -------
        edge_y : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_edge`,)
        """
        if "edge_y" not in self._ds:
            _populate_edge_centroids(self)
        return self._ds["edge_y"]

    edge_y = edge_y.setter(make_setter("edge_y"))

    @property
    def edge_z(self) -> xr.DataArray:
        """Cartesian z coordinate of the center of each edge (in meters).

        Returns
        -------
        edge_z : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_edge`,)
        """
        if "edge_z" not in self._ds:
            _populate_edge_centroids(self)
        return self._ds["edge_z"]

    edge_z = edge_z.setter(make_setter("edge_z"))

    @property
    def face_lon(self) -> xr.DataArray:
        """Longitude coordinate of the center of each face (in degrees).

        Values are expected to be in the range [-180.0, 180.0].

        Returns
        -------
        face_lon : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_face`,)
        """
        if "face_lon" not in self._ds:
            _populate_face_centroids(self)
            _set_desired_longitude_range(self)
        return self._ds["face_lon"]

    face_lon = face_lon.setter(make_setter("face_lon"))

    @property
    def face_lat(self) -> xr.DataArray:
        """Latitude coordinate of the center of each face (in degrees).

        Values are expected to be in the range [-90.0, 90.0].

        Returns
        -------
        face_lat : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_face`,)
        """
        if "face_lat" not in self._ds:
            _populate_face_centroids(self)
            _set_desired_longitude_range(self)

        return self._ds["face_lat"]

    face_lat = face_lat.setter(make_setter("face_lat"))

    @property
    def face_x(self) -> xr.DataArray:
        """Cartesian x coordinate of the center of each face (in meters).

        Returns
        -------
        face_x : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_face`,)
        """
        if "face_x" not in self._ds:
            _populate_face_centroids(self)

        return self._ds["face_x"]

    face_x = face_x.setter(make_setter("face_x"))

    @property
    def face_y(self) -> xr.DataArray:
        """Cartesian y coordinate of the center of each face (in meters).

        Returns
        -------
        face_y : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_face`,)
        """
        if "face_y" not in self._ds:
            _populate_face_centroids(self)
        return self._ds["face_y"]

    face_y = face_y.setter(make_setter("face_y"))

    @property
    def face_z(self) -> xr.DataArray:
        """Cartesian z coordinate of the center of each face (in meters).

        Returns
        -------
        face_z : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_face`,)
        """
        if "face_z" not in self._ds:
            _populate_face_centroids(self)
        return self._ds["face_z"]

    face_z = face_z.setter(make_setter("face_z"))

    # ==================================================================================================================
    # Connectivity Properties
    # ==================================================================================================================

    @property
    def face_node_connectivity(self) -> xr.DataArray:
        """
        Connectivity variable representing the indices of nodes (mesh vertices) that define each face.

        Each row (i.e., each face) contains at least three node indices and up to a maximum of
        :py:attr:`~uxarray.Grid.n_max_face_nodes`. In grids with a mix of geometries (e.g., triangles and hexagons),
        rows containing fewer than :py:attr:`~uxarray.Grid.n_max_face_nodes` indices are padded with the fill value defined in
        :py:attr:`~uxarray.constants.INT_FILL_VALUE`. The node indices are stored in counter-clockwise order.

        Returns
        -------
        face_node_connectivity : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_face`, :py:attr:`~uxarray.Grid.n_max_face_nodes`)
        """

        if (
            "face_node_connectivity" not in self._ds
            and self.source_grid_spec == "HEALPix"
        ):
            _populate_healpix_boundaries(self._ds)

        if self._ds["face_node_connectivity"].ndim == 1:
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

    face_node_connectivity = face_node_connectivity.setter(
        make_setter("face_node_connectivity")
    )

    @property
    def edge_node_connectivity(self) -> xr.DataArray:
        """
        Connectivity variable representing the indices of nodes (mesh vertices) that define each edge.

        Each row (i.e., each edge) contains exactly two node indices that define the start and end points of the edge.
        The nodes are stored in an arbitrary order.

        Returns
        -------
        edge_node_connectivity : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_edge`, 2)
        """
        if "edge_node_connectivity" not in self._ds:
            _populate_edge_node_connectivity(self)

        return self._ds["edge_node_connectivity"]

    edge_node_connectivity = edge_node_connectivity.setter(
        make_setter("edge_node_connectivity")
    )

    @property
    def node_node_connectivity(self) -> xr.DataArray:
        """
        Connectivity variable representing the indices of nodes (mesh vertices) that surround each node.

        Returns
        -------
        node_node_connectivity : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_node`, n_max_node_nodes)
            representing the connectivity.
        """
        if "node_node_connectivity" not in self._ds:
            raise NotImplementedError(
                "Construction of `node_node_connectivity` not yet supported."
            )
        return self._ds["node_node_connectivity"]

    node_node_connectivity = node_node_connectivity.setter(
        make_setter("node_node_connectivity")
    )

    @property
    def face_edge_connectivity(self) -> xr.DataArray:
        """
        Connectivity variable representing the indices of edges that define each face.

        Each row (i.e., each face) contains at least three edge indices and up to a maximum of
        :py:attr:`~uxarray.Grid.n_max_face_edges`. In grids with a mix of geometries (e.g., triangles and hexagons),
        rows containing fewer than :py:attr:`~uxarray.Grid.n_max_face_edges` indices are padded with the fill value defined in
        :py:attr:`~uxarray.constants.INT_FILL_VALUE`.

        Returns
        -------
        face_edge_connectivity : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_face`, :py:attr:`~uxarray.Grid.n_max_face_edges`)
            representing the connectivity.
        """
        if "face_edge_connectivity" not in self._ds:
            _populate_face_edge_connectivity(self)

        return self._ds["face_edge_connectivity"]

    face_edge_connectivity = face_edge_connectivity.setter(
        make_setter("face_edge_connectivity")
    )

    @property
    def edge_edge_connectivity(self) -> xr.DataArray:
        """
        Connectivity variable representing the indices of edges that share at least one node.

        In grids with a mix of geometries (e.g., triangles and hexagons), rows containing fewer than the maximum number
        of edge indices are padded with the fill value defined in :py:attr:`~uxarray.constants.INT_FILL_VALUE`.

        Returns
        -------
        edge_edge_connectivity : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_edge`, :py:attr:`~uxarray.Grid.n_max_edge_edges`)
            representing the connectivity.
        """
        if "edge_edge_connectivity" not in self._ds:
            raise NotImplementedError(
                "Construction of `edge_edge_connectivity` not yet supported."
            )

        return self._ds["edge_edge_connectivity"]

    edge_edge_connectivity = edge_edge_connectivity.setter(
        make_setter("edge_edge_connectivity")
    )

    @property
    def node_edge_connectivity(self) -> xr.DataArray:
        """
        Connectivity variable representing the indices of edges that contain each node.

        In grids with a mix of geometries (e.g., triangles and hexagons), rows containing fewer than the maximum number
        of edge indices are padded with the fill value defined in :py:attr:`~uxarray.constants.INT_FILL_VALUE`.

        Returns
        -------
        node_edge_connectivity : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_node`, :py:attr:`~uxarray.Grid.n_max_node_edges`)
            representing the connectivity.
        """
        if "node_edge_connectivity" not in self._ds:
            _populate_node_edge_connectivity(self)

        return self._ds["node_edge_connectivity"]

    node_edge_connectivity = node_edge_connectivity.setter(
        make_setter("node_edge_connectivity")
    )

    @property
    def face_face_connectivity(self) -> xr.DataArray:
        """
        Connectivity variable representing the indices of faces that share edges.

        In grids with a mix of geometries (e.g., triangles and hexagons), rows containing fewer than
        :py:attr:`~uxarray.Grid.n_max_face_faces` indices are padded with the fill value defined in
        :py:attr:`~uxarray.constants.INT_FILL_VALUE`.

        Returns
        -------
        face_face_connectivity : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_face`, :py:attr:`~uxarray.Grid.n_max_face_faces`)
            representing the connectivity.
        """
        if "face_face_connectivity" not in self._ds:
            _populate_face_face_connectivity(self)

        return self._ds["face_face_connectivity"]

    face_face_connectivity = face_face_connectivity.setter(
        make_setter("face_face_connectivity")
    )

    @property
    def edge_face_connectivity(self) -> xr.DataArray:
        """
        Connectivity variable representing the indices of faces that saddle each edge.

        Each row (i.e., each edge) contains either one or two face indices. A single face indicates that there
        exists an empty region not covered by any geometry (e.g., a coastline). If an edge neighbors only one face,
        the second value is padded with :py:attr:`~uxarray.constants.INT_FILL_VALUE`.

        Returns
        -------
        edge_face_connectivity : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_edge`, 2)
            representing the connectivity.
        """
        if "edge_face_connectivity" not in self._ds:
            _populate_edge_face_connectivity(self)

        return self._ds["edge_face_connectivity"]

    edge_face_connectivity = edge_face_connectivity.setter(
        make_setter("edge_face_connectivity")
    )

    @property
    def node_face_connectivity(self) -> xr.DataArray:
        """
        Connectivity variable representing the indices of faces that share a given node.

        In grids with a mix of geometries (e.g., triangles and hexagons), rows containing fewer than
        :py:attr:`~uxarray.Grid.n_max_node_faces` indices are padded with the fill value defined in
        :py:attr:`~uxarray.constants.INT_FILL_VALUE`.

        Returns
        -------
        node_face_connectivity : :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_node`, :py:attr:`~uxarray.Grid.n_max_node_faces`)
            representing the connectivity.
        """
        if "node_face_connectivity" not in self._ds:
            _populate_node_face_connectivity(self)

        return self._ds["node_face_connectivity"]

    node_face_connectivity = node_face_connectivity.setter(
        make_setter("node_face_connectivity")
    )

    # ==================================================================================================================
    # Descriptor Properties
    # ==================================================================================================================

    @property
    def edge_node_distances(self) -> xr.DataArray:
        """Arc distance between the two nodes that make up each edge (in radians).

        Returns
        -------
        edge_node_distances: :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_edge`,)
        """
        if "edge_node_distances" not in self._ds:
            _populate_edge_node_distances(self)
        return self._ds["edge_node_distances"]

    edge_node_distances = edge_node_distances.setter(make_setter("edge_node_distances"))

    @property
    def edge_face_distances(self) -> xr.DataArray:
        """Arc distance between the faces that saddle each edge (in radians).

        Returns
        -------
        edge_face_distances: :py:class:`xarray.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_edge`,)
        """
        if "edge_face_distances" not in self._ds:
            _populate_edge_face_distances(self)
        return self._ds["edge_face_distances"]

    edge_face_distances = edge_face_distances.setter(make_setter("edge_face_distances"))

    @property
    def antimeridian_face_indices(self) -> np.ndarray:
        """Indices of faces that touch or cross the antimeridian.

        Returns
        -------
        antimeridian_face_indices: :py:class:`np.ndarray`
            An array of shape (:py:attr:`~uxarray.Grid.n_face`,)

        """
        if self._antimeridian_face_indices is None:
            self._antimeridian_face_indices = _populate_antimeridian_face_indices(self)
        return self._antimeridian_face_indices

    @property
    def face_areas(self) -> xr.DataArray:
        """Area of each face.

        Returns
        -------
        face_areas: :py:class:`xr.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_face`,)

        Notes
        -----
        For HEALPix grids, this property returns theoretical equal areas to preserve
        the equal-area property. For other grid types, areas are computed geometrically.
        """
        from uxarray.conventions.descriptors import FACE_AREAS_ATTRS, FACE_AREAS_DIMS

        if "face_areas" not in self._ds:
            # Check if this is a HEALPix grid
            if self._ds.attrs.get("source_grid_spec") == "HEALPix":
                # For HEALPix grids, use theoretical equal areas
                from uxarray.io._healpix import _compute_healpix_face_areas

                self._ds["face_areas"] = _compute_healpix_face_areas(self._ds)
            else:
                # For other grids, use calculated areas
                face_areas, self._face_jacobian = self._compute_face_areas()
                self._ds["face_areas"] = xr.DataArray(
                    data=face_areas, dims=FACE_AREAS_DIMS, attrs=FACE_AREAS_ATTRS
                )
        return self._ds["face_areas"]

    face_areas = face_areas.setter(make_setter("face_areas"))

    @property
    def bounds(self) -> xr.DataArray:
        """Spherical bounds of each face in degrees


        Returns
        -------
        bounds: :py:class:`xr.DataArray`
            An array of shape (:py:attr:`~uxarray.Grid.n_face`, `two`, `two`)
        """
        if "bounds" not in self._ds:
            _populate_face_bounds(self)
        return self._ds["bounds"]

    bounds = bounds.setter(make_setter("bounds"))

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

    boundary_edge_indices = boundary_edge_indices.setter(
        make_setter("boundary_edge_indices")
    )

    @property
    def boundary_node_indices(self):
        """Indices of nodes that border regions not covered by any geometry
        (holes) in a partial grid."""
        if "boundary_node_indices" not in self._ds:
            raise ValueError

        return self._ds["boundary_node_indices"]

    boundary_node_indices = boundary_node_indices.setter(
        make_setter("boundary_node_indices")
    )

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

    boundary_face_indices = boundary_face_indices.setter(
        make_setter("boundary_face_indices")
    )

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

    @property
    def inverse_indices(self) -> xr.Dataset:
        """Indices for a subset that map each face in the subset back to the original grid"""
        if self.is_subset:
            return self._inverse_indices
        else:
            raise Exception(
                "Grid is not a subset, therefore no inverse face indices exist"
            )

    @property
    def is_subset(self):
        """Returns `True` if the Grid is a subset, 'False' otherwise."""
        return self._is_subset

    @property
    def max_face_radius(self):
        """Maximum Euclidean distance from each face center to its nodes."""
        if "max_face_radius" not in self._ds:
            self._ds["max_face_radius"] = _populate_max_face_radius(self)
        return self._ds["max_face_radius"]

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
        coordinates: str | None = "face centers",
        coordinate_system: str | None = "spherical",
        distance_metric: str | None = "haversine",
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
        coordinates : str, default="face centers"
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

    def _get_scipy_kd_tree(
        self, coordinates: str | None = "face", reconstruct: bool = False
    ):
        """
        Build or retrieve a KDTree for efficient nearest-neighbor searches on grid points.

        In the future, this will become the default KDtree and will be moved to the Public API.

        Parameters
        ----------
        coordinates : {'node', 'edge', 'face'}, default='face'
            Which set of grid coordinates to use when constructing the KDTree:
            - 'node': corner-node positions
            - 'edge': edge-center positions
            - 'face': face-center positions
        reconstruct : bool, default=False
            If True, always rebuild the KDTree even if one exists for the given coordinate set.

        Returns
        -------
        scipy.spatial.KDTree
            A KDTree built from the specified 3D coordinates of the grid.

        Raises
        ------
        ValueError
            If `coordinates` is not one of 'node', 'edge', or 'face'.

        Notes
        -----
        - Trees are cached per-coordinate-set in `self._kdtrees` to avoid repeated construction.
        - The tree uses the (x, y, z) Cartesian values stored on each grid element.
        """
        from scipy.spatial import KDTree as SPKDTree

        if coordinates not in ("node", "edge", "face"):
            raise ValueError(
                f"Invalid coordinates='{coordinates}'; "
                "must be 'node', 'edge', or 'face'."
            )

        if reconstruct or coordinates not in self._kdtrees:
            self.normalize_cartesian_coordinates()
            x = getattr(self, f"{coordinates}_x").values
            y = getattr(self, f"{coordinates}_y").values
            z = getattr(self, f"{coordinates}_z").values

            points = np.vstack([x, y, z]).T
            self._kdtrees[coordinates] = SPKDTree(points)

        return self._kdtrees[coordinates]

    def get_kd_tree(
        self,
        coordinates: str | None = "face centers",
        coordinate_system: str | None = "cartesian",
        distance_metric: str | None = "minkowski",
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
        coordinates : str, default="face centers"
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

    def get_spatial_hash(
        self,
        reconstruct: bool = False,
    ):
        """Get the SpatialHash data structure of this Grid that allows for
        fast face search queries. Face searches are used to find the faces that
        a list of points, in spherical coordinates, are contained within.

        Parameters
        ----------
        reconstruct : bool, default=False
            If true, reconstructs the spatial hash

        Returns
        -------
        self._spatialhash : grid.Neighbors.SpatialHash
            SpatialHash instance

        Note
        ----
        Does not currently support queries on periodic elements.

        Examples
        --------
        Open a grid from a file path:

        >>> import uxarray as ux
        >>> uxgrid = ux.open_grid("grid_filename.nc")

        Obtain SpatialHash instance:

        >>> spatial_hash = uxgrid.get_spatial_hash()

        Query to find the face a point lies within in addition to its barycentric coordinates:

        >>> face_ids, bcoords = spatial_hash.query([0.0, 0.0])
        """
        if self._spatialhash is None or reconstruct:
            self._spatialhash = SpatialHash(self, reconstruct)

        return self._spatialhash

    def copy(self):
        """Returns a deep copy of this grid."""

        return Grid(
            self._ds,
            source_grid_spec=self.source_grid_spec,
            source_dims_dict=self._source_dims_dict,
        )

    def calculate_total_face_area(
        self,
        quadrature_rule: str | None = "triangular",
        order: int | None = 4,
        latitude_adjusted_area: bool | None = False,
    ) -> float:
        """Function to calculate the total surface area of all the faces in a
        mesh.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.
        latitude_adjusted_area : bool, optional
            If True, corrects the area of the faces accounting for lines of constant lattitude. Defaults to False.

        Returns
        -------
        Sum of area of all the faces in the mesh : float
        """

        return np.sum(self.face_areas.values)

    def compute_face_areas(
        self,
        quadrature_rule: str | None = "triangular",
        order: int | None = 4,
        latitude_adjusted_area: bool | None = False,
    ):
        """Face areas calculation function for grid class, calculates area of
        all faces in the grid.

        .. deprecated:: 2025.10.0
            Use the `face_areas` property instead for better performance and caching.
            This method will be removed in a future version.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.
        latitude_adjusted_area : bool, optional
            If True, corrects the area of the faces accounting for lines of constant lattitude. Defaults to False.

        Returns
        -------
        1. Area of all the faces in the mesh : np.ndarray
        2. Jacobian of all the faces in the mesh : np.ndarray

        Notes
        -----
        This method performs geometric integration to compute face areas. For HEALPix grids,
        this may not preserve the equal-area property due to differences between algorithmic
        pixel definitions and geometric representation. For HEALPix grids, use the
        ``face_areas`` property instead, which ensures mathematical correctness by using
        theoretical equal areas.
        """
        import warnings

        warnings.warn(
            "compute_face_areas() is deprecated. Use the face_areas property instead for better performance and caching.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._compute_face_areas(quadrature_rule, order, latitude_adjusted_area)

    def _compute_face_areas(
        self,
        quadrature_rule: str | None = "triangular",
        order: int | None = 4,
        latitude_adjusted_area: bool | None = False,
    ):
        """Internal face areas calculation function for grid class, calculates area of
        all faces in the grid.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.
        latitude_adjusted_area : bool, optional
            If True, corrects the area of the faces accounting for lines of constant lattitude. Defaults to False.

        Returns
        -------
        1. Area of all the faces in the mesh : np.ndarray
        2. Jacobian of all the faces in the mesh : np.ndarray

        Notes
        -----
        This method performs geometric integration to compute face areas. For HEALPix grids,
        this may not preserve the equal-area property due to differences between algorithmic
        pixel definitions and geometric representation. For HEALPix grids, use the
        ``face_areas`` property instead, which ensures mathematical correctness by using
        theoretical equal areas.
        """
        # if self._face_areas is None: # this allows for using the cached result,
        # but is not the expected behavior behavior as we are in need to recompute if this function is called with different quadrature_rule or order

        self.normalize_cartesian_coordinates()
        x = self.node_x.values
        y = self.node_y.values
        z = self.node_z.values

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
            quadrature_rule,
            order,
            latitude_adjusted_area,
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

        for prefix in ("node", "edge", "face"):
            x_var = f"{prefix}_x"
            if x_var not in self._ds:
                continue

            dx = self._ds[f"{prefix}_x"]
            dy = self._ds[f"{prefix}_y"]
            dz = self._ds[f"{prefix}_z"]

            # normalize
            norm = (dx**2 + dy**2 + dz**2) ** 0.5
            self._ds[x_var] = dx / norm
            self._ds[f"{prefix}_y"] = dy / norm
            self._ds[f"{prefix}_z"] = dz / norm

    @property
    def sphere_radius(self) -> float:
        """Physical sphere radius from the source grid (e.g., Earth's radius for MPAS ocean grids).

        Internally, all calculations use a unit sphere. This property stores the original
        radius for scaling results back to physical units.

        Returns
        -------
        sphere_radius : float
            The physical sphere radius. Defaults to 1.0 if not set.
        """
        return self._ds.attrs.get("sphere_radius", 1.0)

    @sphere_radius.setter
    def sphere_radius(self, radius: float) -> None:
        """Set the sphere radius for the grid.

        Parameters
        ----------
        radius : float
            The sphere radius to set. Must be positive.

        Raises
        ------
        ValueError
            If radius is not positive.
        """
        if radius <= 0:
            raise ValueError(f"Sphere radius must be positive, got {radius}")

        self._ds.attrs["sphere_radius"] = radius

    def to_xarray(self, grid_format: str | None = "ugrid"):
        """Returns an ``xarray.Dataset`` with the variables stored under the
        ``Grid`` encoded in a specific grid format.

        Parameters
        ----------
        grid_format: str, optional
            The desired grid format for the output dataset.
            One of "ugrid", "exodus", "scrip", or "esmf"

        Returns
        -------
        out_ds: xarray.Dataset
            Dataset representing the unstructured grid in a given grid format
        """
        # Convert to lowercase for case-insensitive comparison
        grid_format_lower = grid_format.lower()

        if grid_format_lower == "ugrid":
            out_ds = _encode_ugrid(self._ds)

        elif grid_format_lower == "exodus":
            out_ds = _encode_exodus(self._ds)

        elif grid_format_lower == "scrip":
            out_ds = _encode_scrip(
                self.face_node_connectivity,
                self.node_lon,
                self.node_lat,
                self.face_areas,
            )

        elif grid_format_lower == "esmf":
            from uxarray.io._esmf import _encode_esmf

            out_ds = _encode_esmf(self._ds)

        else:
            raise ValueError(
                f"Invalid grid_format encountered. Expected one of ['ugrid', 'exodus', 'scrip', 'esmf'] but received: {grid_format}"
            )

        return out_ds

    def to_geodataframe(
        self,
        periodic_elements: str | None = "exclude",
        projection=None,
        cache: bool | None = True,
        override: bool | None = False,
        engine: str | None = "spatialpandas",
        exclude_antimeridian: bool | None = None,
        return_non_nan_polygon_indices: bool | None = False,
        exclude_nan_polygons: bool | None = True,
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

        from spatialpandas import GeoDataFrame

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
        **kwargs,
    ):
        """Constructs a ``matplotlib.collections.PolyCollection``` consisting
        of polygons representing the faces of the unstructured grid.

        Parameters
        ----------
        **kwargs: dict
            Key word arguments to pass into the construction of a PolyCollection
        """
        import cartopy.crs as ccrs

        if "projection" in kwargs:
            proj = kwargs.pop("projection")
            warn(
                (
                    "'projection' is not a supported argument and will be ignored. "
                    "Define the desired projection on the GeoAxes that this collection will be added to. "
                    "Example:\n"
                    "    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()})\n"
                    "    ax.add_collection(poly_collection)\n"
                    f"(received projection={proj!r})"
                ),
                category=FutureWarning,
                stacklevel=2,
            )

        if self._cached_poly_collection:
            return copy.deepcopy(self._cached_poly_collection)

        (
            poly_collection,
            corrected_to_original_faces,
        ) = _grid_to_matplotlib_polycollection(
            self, periodic_elements="ignore", projection=None, **kwargs
        )

        poly_collection.set_transform(ccrs.Geodetic())
        self._cached_poly_collection = poly_collection

        return copy.deepcopy(poly_collection)

    def to_linecollection(
        self,
        **kwargs,
    ):
        """Constructs a ``matplotlib.collections.LineCollection``` consisting
        of lines representing the edges of the unstructured grid.

        Parameters
        ----------
        **kwargs: dict
            Key word arguments to pass into the construction of a LineCollection
        """
        import cartopy.crs as ccrs

        if "projection" in kwargs:
            proj = kwargs.pop("projection")
            warn(
                (
                    "'projection' is not a supported argument and will be ignored. "
                    "Define the desired projection on the GeoAxes that this collection will be added to. "
                    "Example:\n"
                    "    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()})\n"
                    "    ax.add_collection(line_collection)\n"
                    f"(received projection={proj!r})"
                ),
                category=FutureWarning,
                stacklevel=2,
            )
        if self._cached_line_collection:
            return copy.deepcopy(self._cached_line_collection)

        line_collection = _grid_to_matplotlib_linecollection(
            grid=self,
            periodic_elements="ingore",
            projection=None,
            **kwargs,
        )

        line_collection.set_transform(ccrs.Geodetic())

        self._cached_line_collection = line_collection

        return copy.deepcopy(line_collection)

    def get_dual(self, check_duplicate_nodes: bool = False):
        """Compute the dual for a grid, which constructs a new grid centered
        around the nodes, where the nodes of the primal become the face centers
        of the dual, and the face centers of the primal become the nodes of the
        dual. Returns a new `Grid` object.

        Returns
        --------
        dual : Grid
            Dual Mesh Grid constructed
        """

        if check_duplicate_nodes:
            if _check_duplicate_nodes_indices(self):
                # TODO: This is very slow
                raise RuntimeError("Duplicate nodes found, cannot construct dual")

        # Get dual mesh node face connectivity
        dual_node_face_conn = construct_dual(grid=self)

        # Construct dual mesh
        dual = self.from_topology(
            self.face_lon.values, self.face_lat.values, dual_node_face_conn
        )

        return dual

    def isel(self, inverse_indices: list[str] | set[str] | bool = False, **dim_kwargs):
        """Indexes an unstructured grid along a given dimension (``n_node``,
        ``n_edge``, or ``n_face``) and returns a new grid.

        Currently only supports inclusive selection, meaning that for cases where node or edge indices are provided,
        any face that contains that element is included in the resulting subset. This means that additional elements
        beyond those that were initially provided in the indices will be included. Support for more methods, such as
        exclusive and clipped indexing is in the works.

        Parameters
        ----------
        inverse_indices : list[str] | set[str] | bool, default=False
            Indicates whether to store the original grids indices. Passing `True` stores the original face indices,
            other reverse indices can be stored by passing any or all of the following: (["face", "edge", "node"], True)
        **dims_kwargs: kwargs
            Dimension to index, one of ['n_node', 'n_edge', 'n_face']

        Example
        -------
        >> grid = ux.open_grid(grid_path)
        >> grid.isel(n_face = [1,2,3,4])
        """
        from .slice import _slice_edge_indices, _slice_face_indices, _slice_node_indices

        if len(dim_kwargs) != 1:
            raise ValueError("Indexing must be along a single dimension.")

        if "n_node" in dim_kwargs:
            if inverse_indices:
                raise Exception(
                    "Inverse indices are not yet supported for node selection, please use face centers"
                )
            return _slice_node_indices(self, dim_kwargs["n_node"])

        elif "n_edge" in dim_kwargs:
            if inverse_indices:
                raise Exception(
                    "Inverse indices are not yet supported for edge selection, please use face centers"
                )
            return _slice_edge_indices(self, dim_kwargs["n_edge"])

        elif "n_face" in dim_kwargs:
            return _slice_face_indices(
                self, dim_kwargs["n_face"], inverse_indices=inverse_indices
            )

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
            edge_node_x = self.node_x[self.edge_node_connectivity].values
            edge_node_y = self.node_y[self.edge_node_connectivity].values
            edges = constant_lon_intersections_no_extreme(
                lon, edge_node_x, edge_node_y, self.n_edge
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

    def get_faces_between_longitudes(self, lons: tuple[float, float]):
        """Identifies the indices of faces that are strictly between two lines of constant longitude.

        Parameters
        ----------
        lons: tuple[float, float]
            A tuple of longitudes that define that minimum and maximum longitude.

        Returns
        -------
        faces : numpy.ndarray
            An array of face indices that are strictly between two lines of constant longitude.

        """
        return faces_within_lon_bounds(lons, self.face_bounds_lon.values)

    def get_faces_between_latitudes(self, lats: tuple[float, float]):
        """Identifies the indices of faces that are strictly between two lines of constant latitude.

        Parameters
        ----------
        lats: tuple[float, float]
            A tuple of latitudes that define that minimum and maximum latitudes.

        Returns
        -------
        faces : numpy.ndarray
            An array of face indices that are strictly between two lines of constant latitude.

        """
        return faces_within_lat_bounds(lats, self.face_bounds_lat.values)

    def get_faces_containing_point(
        self,
        points: Sequence[float] | np.ndarray,
        return_counts: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | list[list[int]]:
        """
        Identify which grid faces contain the given point(s).

        Parameters
        ----------
        points : array_like, shape (N, 2) or (2,) or shape (N, 3) or (3,)
            Query point(s) to locate on the grid.
            - If last dimension is 2, interpreted as (longitude, latitude) in **degrees**.
            - If last dimension is 3, interpreted as Cartesian coordinates on the unit sphere: (x, y, z).
            You may pass a single point (shape `(2,)` or `(3,)`) or multiple points (shape `(N, 2)` or `(N, 3)`).
        return_counts : bool, default=True
            - If True, returns a tuple `(face_indices, counts)`.
            - If False, returns a `list` of per-point lists of face indices (no padding).

        Returns
        -------
        If `return_counts=True`:
          face_indices : np.ndarray, shape (N, M) or (N, 1)
              2D array of face indices.  Rows are padded with `INT_FILL_VALUE` when a point
              lies on corners of multiple faces.  If every queried point falls in exactly
              one face, the result has shape `(N, 1)`.
          counts : np.ndarray, shape (N,)
              Number of valid face indices in each row of `face_indices`.

        If `return_counts=False`:
          list[list[int]]
              Python list of length `N`, where each element is the list of face
              indices for that point (no padding, in natural order).

        Notes
        -----
        - Most points will lie strictly inside exactly one face; in that case,
          `counts == 1` and `face_indices` has one column.
        - Points that lie exactly on a vertex or edge shared by multiple faces
          return multiple indices in the first `counts[i]` columns of row `i`,
          with any remaining columns filled by `INT_FILL_VALUE`.


        Examples
        --------

        Query a single spherical point

        >>> face_indices, counts = uxgrid.get_faces_containing_point(points=(0.0, 0.0))

        Query a single Cartesian point

         >>> face_indices, counts = uxgrid.get_faces_containing_point(
         ...     points=[0.0, 0.0, 1.0]
         ... )

        Query multiple points at once

        >>> pts = [(0.0, 0.0), (10.0, 20.0)]
        >>> face_indices, counts = uxgrid.get_faces_containing_point(points=pts)

        Return a list of lists

        >>> face_indices_list = uxgrid.get_faces_containing_point(
        ...     points=[0.0, 0.0, 1.0], return_counts=False
        ... )

        """

        # Determine faces containing points
        face_indices, counts = _point_in_face_query(
            source_grid=self, points=points_atleast_2d_xyz(points)
        )

        # Return a list of lists if counts are not desired
        if not return_counts:
            output: list[list[int]] = []
            for i, c in enumerate(counts):
                output.append(face_indices[i, :c].tolist())
            return output

        if (counts == 1).all():
            face_indices = face_indices[:, 0]
            face_indices = face_indices[:, None]

        return face_indices, counts
