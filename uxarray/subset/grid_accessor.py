from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from uxarray.grid import Grid


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

        faces_between_lons = self.uxgrid.get_faces_between_longitudes(lon_bounds)
        face_between_lats = self.uxgrid.get_faces_between_latitudes(lat_bounds)

        faces = np.intersect1d(faces_between_lons, face_between_lats)

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
            raise ValueError("Coordinates must be one-dimensional")

        if len(coords) == 2:
            # Spherical coordinates
            tree = self.uxgrid.get_ball_tree(tree_type)
        elif len(coords) == 3:
            # Cartesian coordinates
            tree = self.uxgrid.get_kd_tree(tree_type)
        else:
            raise ValueError("Unsupported coordinates provided.")

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
