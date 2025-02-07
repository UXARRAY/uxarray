from __future__ import annotations

from typing import TYPE_CHECKING, Union, List, Set, Tuple

if TYPE_CHECKING:
    from uxarray.grid import Grid


class GridCrossSectionAccessor:
    """Accessor for cross-section operations on a ``Grid``"""

    def __init__(self, uxgrid: Grid) -> None:
        self.uxgrid = uxgrid

    def __repr__(self):
        prefix = "<uxarray.Grid.cross_section>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += (
            "  * constant_latitude(lat, return_face_indices, inverse_indices)\n"
        )
        methods_heading += (
            "  * constant_longitude(lon, return_face_indices, inverse_indices)\n"
        )
        methods_heading += "  * constant_latitude_interval(lats, return_face_indices, inverse_indices)\n"
        methods_heading += "  * constant_longitude_interval(lons, return_face_indices, inverse_indices)\n"
        return prefix + methods_heading

    def constant_latitude(
        self,
        lat: float,
        return_face_indices: bool = False,
        inverse_indices: Union[List[str], Set[str], bool] = False,
    ):
        """Extracts a cross-section of the grid by selecting all faces that
        intersect with a specified line of constant latitude.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the cross-section, in degrees.
            Must be between -90.0 and 90.0
        return_face_indices : bool, optional
            If True, also returns the indices of the faces that intersect with the
            line of constant latitude.
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face  indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.Grid
            A subset of the original grid containing only the faces that intersect
            with the specified latitude.
        Tuple[uxarray.Grid, numpy.ndarray], optional
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
        inverse_indices: Union[List[str], Set[str], bool] = False,
    ):
        """Extracts a cross-section of the grid by selecting all faces that
        intersect with a specified line of constant longitude.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the cross-section, in degrees.
            Must be between -180.0 and 180.0
        return_face_indices : bool, optional
            If True, also returns the indices of the faces that intersect with the
            line of constant longitude.
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face  indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.Grid
            A subset of the original grid containing only the faces that intersect
            with the specified longitude.
        Tuple[uxarray.Grid, numpy.ndarray], optional
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
        lats: Tuple[float, float],
        return_face_indices: bool = False,
        inverse_indices: Union[List[str], Set[str], bool] = False,
    ):
        """Extracts a cross-section of the grid by selecting all faces that
        are within a specified latitude interval.

        Parameters
        ----------
        lats : Tuple[float, float]
            The latitude interval (min_lat, max_lat) at which to extract the cross-section,
            in degrees. Values must be between -90.0 and 90.0
        return_face_indices : bool, optional
            If True, also returns the indices of the faces that intersect with the
            latitude interval.
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.Grid
            A subset of the original grid containing only the faces that are within a specified latitude interval.
        Tuple[uxarray.Grid, numpy.ndarray], optional
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
        lons: Tuple[float, float],
        return_face_indices: bool = False,
        inverse_indices: Union[List[str], Set[str], bool] = False,
    ):
        """Extracts a cross-section of the grid by selecting all faces are within a specifed longitude interval.

        Parameters
        ----------
        lons : Tuple[float, float]
            The longitude interval (min_lon, max_lon) at which to extract the cross-section,
            in degrees. Values must be between -180.0 and 180.0
        return_face_indices : bool, optional
            If True, also returns the indices of the faces that intersect are within a specifed longitude interval.
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.Grid
            A subset of the original grid containing only the faces that intersect
            with the specified longitude interval.
        Tuple[uxarray.Grid, numpy.ndarray], optional
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
