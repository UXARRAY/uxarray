from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uxarray.errors import DimensionError

if TYPE_CHECKING:
    from uxarray.grid import Grid


#: Below this size the whole-mesh bounds computation is cheap enough that
#: the pre-filter's own cost is not repaid.
_PREFILTER_MIN_FACES = 1_000_000


class GridSubsetAccessor:
    """Accessor for performing unstructured grid subsetting, accessed through
    ``Grid.subset``"""

    def __init__(self, uxgrid: Grid) -> None:
        self.uxgrid = uxgrid

    def __repr__(self):
        prefix = "<uxarray.Grid.subset>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += "  * nearest_neighbor(center_coord, k, element)\n"
        methods_heading += "  * bounding_circle(center_coord, r, element)\n"
        methods_heading += "  * bounding_box(lon_bounds, lat_bounds)\n"
        methods_heading += "  * constant_latitude(lat, lon_range)\n"
        methods_heading += "  * constant_longitude(lon, lat_range)\n"
        methods_heading += "  * constant_latitude_interval(lats)\n"
        methods_heading += "  * constant_longitude_interval(lons)\n"

        return prefix + methods_heading

    def bounding_box(
        self,
        lon_bounds: tuple[float, float],
        lat_bounds: tuple[float, float],
        inverse_indices: list[str] | set[str] | bool = False,
    ):
        """Subsets an unstructured grid between two latitude and longitude
        points which form a bounding box.

        A bounding box may span the antimeridian, when the pair of longitude points is given in descending order (
        i.e. the first longitude point is greater than the second).

        Parameters
        ----------
        lon_bounds: tuple, list, np.ndarray
            (lon_left, lon_right) where lon_left < lon_right when the bounding box does not span
            the antimeridian, otherwise lon_left > lon_right, both between [-180, 180]
        lat_bounds: tuple, list, np.ndarray
            (lat_bottom, lat_top) where lat_top > lat_bottom and between [-90, 90]
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)
        """

        faces = _faces_in_bounding_box(self.uxgrid, lon_bounds, lat_bounds)

        return self.uxgrid.isel(n_face=faces, inverse_indices=inverse_indices)

    def bounding_circle(
        self,
        center_coord: tuple | list | np.ndarray,
        r: float | int,
        element: str | None = "face centers",
        inverse_indices: list[str] | set[str] | bool = False,
        **kwargs,
    ):
        """Subsets an unstructured grid by returning all elements within some
        radius (in degrees) from a center coord.

        Parameters
        ----------
        center_coord : tuple, list, np.ndarray
            Longitude and latitude of the center of the bounding circle
        r: scalar, int, float
            Radius of bounding circle (in degrees)
        element: str
            Element for use with `coords` comparison, one of `nodes`, `face centers`, or `edge centers`
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)
        """

        coords = np.asarray(center_coord)

        tree = self._get_tree(coords, element)

        ind = tree.query_radius(coords, r)

        if len(ind) == 0:
            raise ValueError(
                f"No elements founding within the bounding circle with radius {r} when querying {element}"
            )

        return self._index_grid(ind, element, inverse_indices)

    def nearest_neighbor(
        self,
        center_coord: tuple | list | np.ndarray,
        k: int,
        element: str | None = "face centers",
        inverse_indices: list[str] | set[str] | bool = False,
        **kwargs,
    ):
        """Subsets an unstructured grid by returning the ``k`` closest
        neighbors from a center coordinate.

        Parameters
        ----------
        center_coord : tuple, list, np.ndarray
            Longitude and latitude of the center of the bounding circle
        k: int
            Number of neighbors to query
        element: str
            Element for use with `coords` comparison, one of `nodes`, `face centers`, or `edge centers`
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)
        """

        coords = np.asarray(center_coord)

        tree = self._get_tree(coords, element)

        _, ind = tree.query(coords, k)

        return self._index_grid(ind, element, inverse_indices=inverse_indices)

    def constant_latitude(
        self,
        lat: float,
        return_face_indices: bool = False,
        inverse_indices: list[str] | set[str] | bool = False,
    ):
        """Extracts a subset of the grid by selecting all faces that
        intersect with a specified line of constant latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the subset, in degrees.
            Must be between -90.0 and 90.0
        return_face_indices : bool, optional
            If True, also returns the indices of the faces that intersect with the
            line of constant latitude.
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face  indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.Grid
            A subset of the original grid containing only the faces that intersect
            with the specified latitude.
        tuple[uxarray.Grid, numpy.ndarray], optional
            If return_face_indices=True, returns a tuple of (grid_subset, face_indices)

        Raises
        ------
        ValueError
            If no intersections are found at the specified latitude.

        Examples
        --------
        >>> # Extract grid at 25° latitude
        >>> cross_section = grid.cross_section.constant_latitude(lat=25.0)
        >>> # With face indices
        >>> cross_section, faces = grid.cross_section.constant_latitude(
        ...     lat=25.0, return_face_indices=True
        ... )

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """

        faces = self.uxgrid.get_faces_at_constant_latitude(
            lat,
        )

        if len(faces) == 0:
            raise ValueError(f"No intersections found at lat={lat}.")

        grid_at_constant_lat = self.uxgrid.isel(
            n_face=faces, inverse_indices=inverse_indices
        )

        if return_face_indices:
            return grid_at_constant_lat, faces
        else:
            return grid_at_constant_lat

    def constant_longitude(
        self,
        lon: float,
        return_face_indices: bool = False,
        inverse_indices: list[str] | set[str] | bool = False,
    ):
        """Extracts a subset of the grid by selecting all faces that
        intersect with a specified line of constant longitude.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the subset, in degrees.
            Must be between -180.0 and 180.0
        return_face_indices : bool, optional
            If True, also returns the indices of the faces that intersect with the
            line of constant longitude.
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face  indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.Grid
            A subset of the original grid containing only the faces that intersect
            with the specified longitude.
        tuple[uxarray.Grid, numpy.ndarray], optional
            If return_face_indices=True, returns a tuple of (grid_subset, face_indices)

        Raises
        ------
        ValueError
            If no intersections are found at the specified longitude.

        Examples
        --------
        >>> # Extract grid at 0° longitude (Prime Meridian)
        >>> cross_section = grid.cross_section.constant_longitude(lon=0.0)
        >>> # With face indices
        >>> cross_section, faces = grid.cross_section.constant_longitude(
        ...     lon=0.0, return_face_indices=True
        ... )

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        faces = self.uxgrid.get_faces_at_constant_longitude(
            lon,
        )

        if len(faces) == 0:
            raise ValueError(f"No intersections found at lon={lon}")

        grid_at_constant_lon = self.uxgrid.isel(
            n_face=faces, inverse_indices=inverse_indices
        )

        if return_face_indices:
            return grid_at_constant_lon, faces
        else:
            return grid_at_constant_lon

    def constant_latitude_interval(
        self,
        lats: tuple[float, float],
        return_face_indices: bool = False,
        inverse_indices: list[str] | set[str] | bool = False,
    ):
        """Extracts a subset of the grid by selecting all faces that
        are within a specified latitude interval.

        Parameters
        ----------
        lats : tuple[float, float]
            The latitude interval (min_lat, max_lat) at which to extract the subset,
            in degrees. Values must be between -90.0 and 90.0
        return_face_indices : bool, optional
            If True, also returns the indices of the faces that intersect with the
            latitude interval.
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.Grid
            A subset of the original grid containing only the faces that are within a specified latitude interval.
        tuple[uxarray.Grid, numpy.ndarray], optional
            If return_face_indices=True, returns a tuple of (grid_subset, face_indices)

        Raises
        ------
        ValueError
            If no faces are found within the specified latitude interval.

        Examples
        --------
        >>> # Extract grid between 30°S and 30°N latitude
        >>> cross_section = grid.cross_section.constant_latitude_interval(
        ...     lats=(-30.0, 30.0)
        ... )
        >>> # With face indices
        >>> cross_section, faces = grid.cross_section.constant_latitude_interval(
        ...     lats=(-30.0, 30.0), return_face_indices=True
        ... )

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        faces = self.uxgrid.get_faces_between_latitudes(lats)

        grid_between_lats = self.uxgrid.isel(
            n_face=faces, inverse_indices=inverse_indices
        )

        if return_face_indices:
            return grid_between_lats, faces
        else:
            return grid_between_lats

    def constant_longitude_interval(
        self,
        lons: tuple[float, float],
        return_face_indices: bool = False,
        inverse_indices: list[str] | set[str] | bool = False,
    ):
        """Extracts a subset of the grid by selecting all faces that are within a specified longitude interval.

        Parameters
        ----------
        lons : tuple[float, float]
            The longitude interval (min_lon, max_lon) at which to extract the subset,
            in degrees. Values must be between -180.0 and 180.0
        return_face_indices : bool, optional
            If True, also returns the indices of the faces that are within a specified longitude interval.
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.Grid
            A subset of the original grid containing only the faces that intersect
            with the specified longitude interval.
        tuple[uxarray.Grid, numpy.ndarray], optional
            If return_face_indices=True, returns a tuple of (grid_subset, face_indices)

        Raises
        ------
        ValueError
            If no faces are found within the specified longitude interval.

        Examples
        --------
        >>> # Extract grid between 0° and 45° longitude
        >>> cross_section = grid.cross_section.constant_longitude_interval(
        ...     lons=(0.0, 45.0)
        ... )
        >>> # With face indices
        >>> cross_section, faces = grid.cross_section.constant_longitude_interval(
        ...     lons=(0.0, 45.0), return_face_indices=True
        ... )

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        faces = self.uxgrid.get_faces_between_longitudes(lons)

        grid_between_lons = self.uxgrid.isel(
            n_face=faces, inverse_indices=inverse_indices
        )

        if return_face_indices:
            return grid_between_lons, faces
        else:
            return grid_between_lons

    def _get_tree(self, coords, tree_type):
        """Internal helper for obtaining the desired KDTree or BallTree."""
        if coords.ndim > 1:
            raise DimensionError("Coordinates must be one-dimensional")

        if len(coords) == 2:
            # Spherical coordinates
            tree = self.uxgrid.get_ball_tree(tree_type)
        elif len(coords) == 3:
            # Cartesian coordinates
            tree = self.uxgrid.get_kd_tree(tree_type)
        else:
            raise DimensionError("Unsupported coordinates provided.")

        return tree

    def _index_grid(self, ind, tree_type, inverse_indices=False):
        """Internal helper for indexing a grid with indices based off the
        provided tree type."""
        if tree_type == "nodes":
            return self.uxgrid.isel(inverse_indices, n_node=ind)
        elif tree_type == "edge centers":
            return self.uxgrid.isel(inverse_indices, n_edge=ind)
        else:
            return self.uxgrid.isel(inverse_indices, n_face=ind)


def _faces_in_bounding_box(uxgrid, lon_bounds, lat_bounds):
    """Face indices whose bounds fall inside a lon/lat box.

    Computing ``Grid.bounds`` costs memory proportional to the whole mesh --
    it materializes the node coordinate and connectivity arrays and builds a
    per-face box for every face, whether or not the face is anywhere near the
    region asked for. On a 300M-face grid that is larger than the machine, so
    a crop of a few thousand faces was not merely slow but impossible
    (UXARRAY/uxarray#1778).

    Face latitudes are already present and cost one array, so they are used
    first to discard faces that cannot be in the result. Exact bounds are then
    computed for the survivors only, and the exact longitude and latitude
    tests decide the answer as before.

    The answer is identical to computing bounds for every face; only the
    number of faces examined changes. Measured on a 12-corner SCRIP mesh with
    a crop of ~0.007% of the faces:

    ===========  ==================  ====================
    n_face       whole-mesh bounds   latitude pre-filter
    ===========  ==================  ====================
    1,600,000    0.84 GiB, 2.6 s     0.00 GiB, 0.8 s
    3,200,000    1.48 GiB, 5.1 s     0.00 GiB, 1.6 s
    ===========  ==================  ====================
    """
    from uxarray.grid.bounds import _construct_face_bounds_array
    from uxarray.grid.intersections import (
        faces_within_lat_bounds,
        faces_within_lon_bounds,
    )

    # The exact path, unchanged, when bounds are already paid for or the mesh
    # is small enough that the filter cannot repay its own cost.
    if "bounds" in uxgrid._ds or uxgrid.n_face <= _PREFILTER_MIN_FACES:
        return np.intersect1d(
            uxgrid.get_faces_between_longitudes(lon_bounds),
            uxgrid.get_faces_between_latitudes(lat_bounds),
        )

    face_lat = np.asarray(uxgrid.face_lat.values)

    # No margin is needed, which is worth stating because the obvious
    # assumption is the opposite. ``faces_within_lat_bounds`` keeps a face
    # only when its bounds are *fully contained* in the query interval
    # (``bounds_min >= query_min and bounds_max <= query_max``) -- despite the
    # docstring's "overlap" wording, the implementation is containment. A
    # contained face necessarily has a contained centre, so a face the exact
    # path keeps can never have a centre outside the box, and the filter is
    # conservative with equality rather than by a fudge factor.
    lat_lo, lat_hi = min(lat_bounds), max(lat_bounds)
    near_lat = (face_lat >= lat_lo) & (face_lat <= lat_hi)

    # Longitude is deliberately *not* filtered on the centre. A face near a
    # pole has a longitude span that says nothing about where its centre is:
    # measured on a HEALPix z3 grid, bounds reach 354 degrees from the centre,
    # because a face containing a pole is recorded as spanning every
    # longitude. No finite margin makes a centre test safe for those, so the
    # latitude filter -- which is well behaved, the same measurement puts its
    # worst case at 6.3 degrees -- carries the reduction on its own, and
    # longitude is settled exactly in the second stage.
    candidates = np.flatnonzero(near_lat)
    if candidates.size == 0:
        return candidates.astype(np.intp)

    # Exact bounds, for the candidates only.
    sub_bounds = _construct_face_bounds_array(
        np.asarray(uxgrid.face_node_connectivity.values)[candidates],
        np.asarray(uxgrid.n_nodes_per_face.values)[candidates],
        np.asarray(uxgrid.node_x.values),
        np.asarray(uxgrid.node_y.values),
        np.asarray(uxgrid.node_z.values),
        np.asarray(uxgrid.node_lon.values),
        np.asarray(uxgrid.node_lat.values),
        False,
        None,
    )

    bounds_lat = np.sort(np.rad2deg(sub_bounds[:, 0, :]), axis=-1)
    bounds_lon = (np.rad2deg(sub_bounds[:, 1, :]) + 180.0) % 360.0 - 180.0
    spans_all_lon = (bounds_lon[:, 0] == 0) & (bounds_lon[:, 1] == 0)
    bounds_lon[spans_all_lon] = [-180.0, 180.0]

    keep_lon = faces_within_lon_bounds(lon_bounds, bounds_lon)
    keep_lat = faces_within_lat_bounds(lat_bounds, bounds_lat)

    return candidates[np.intersect1d(keep_lon, keep_lat)]
