from __future__ import annotations

from typing import TYPE_CHECKING, List, Set, Tuple, Union

if TYPE_CHECKING:
    pass


class UxDataArrayCrossSectionAccessor:
    """Accessor for cross-section operations on a ``UxDataArray``"""

    def __init__(self, uxda) -> None:
        self.uxda = uxda

    def __repr__(self):
        prefix = "<uxarray.UxDataArray.cross_section>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += "Horizontal Cross-Sections:\n"
        methods_heading += "  * constant_latitude(lat, inverse_indices)\n"
        methods_heading += "  * constant_longitude(lon, inverse_indices)\n"
        methods_heading += "  * constant_latitude_interval(lats, inverse_indices)\n"
        methods_heading += "  * constant_longitude_interval(lons, inverse_indices)\n"

        methods_heading += "\nVertical Cross-Sections:\n"
        methods_heading += (
            "  * vertical_constant_latitude(lat, vertical_coord, inverse_indices)\n"
        )
        methods_heading += (
            "  * vertical_constant_longitude(lon, vertical_coord, inverse_indices)\n"
        )

        return prefix + methods_heading

    def _check_vertical_coord_exists(self, vertical_coord):
        """Check if a specified vertical coordinate exists in the data dimensions."""
        if vertical_coord not in self.uxda.dims:
            available_dims = list(self.uxda.dims)
            raise ValueError(
                f"Vertical coordinate '{vertical_coord}' not found in data dimensions. "
                f"Available dimensions: {available_dims}"
            )
        return vertical_coord

    def _get_vertical_coord_name(self, vertical_coord):
        """Validate the vertical coordinate dimension name."""
        if vertical_coord is None:
            raise ValueError(
                "A vertical coordinate must be explicitly specified. "
                "Use the 'vertical_coord' parameter to specify which dimension represents depth/height."
            )
        return self._check_vertical_coord_exists(vertical_coord)

    def constant_latitude(
        self,
        lat: float,
        inverse_indices: Union[
            List[str], Set[str], Tuple[List[str], bool], bool
        ] = False,
    ):
        """Extracts a horizontal cross-section of the data array by selecting all faces that
        intersect with a specified line of constant latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the cross-section, in degrees.
            Must be between -90.0 and 90.0
        inverse_indices : Union[List[str], Set[str], Tuple[List[str], bool], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - Tuple[List[str], bool]: Advanced usage for grid slicing
            - False: No index storage (default)

        Returns
        -------
        uxarray.UxDataArray
            A subset of the original data array containing only the faces that intersect
            with the specified latitude.

        Raises
        ------
        ValueError
            If no intersections are found at the specified longitude or the data variable is not face-centered.

        Examples
        --------
        >>> # Extract horizontal slice at 15.5°S latitude
        >>> cross_section = uxda.cross_section.constant_latitude(lat=-15.5)

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        if not self.uxda._face_centered():
            raise ValueError(
                "Cross sections are only supported for face-centered data variables."
            )

        faces = self.uxda.uxgrid.get_faces_at_constant_latitude(lat)

        return self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

    def constant_longitude(
        self,
        lon: float,
        inverse_indices: Union[
            List[str], Set[str], Tuple[List[str], bool], bool
        ] = False,
    ):
        """Extracts a horizontal cross-section of the data array by selecting all faces that
        intersect with a specified line of constant longitude.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the cross-section, in degrees.
            Must be between -180.0 and 180.0
        inverse_indices : Union[List[str], Set[str], Tuple[List[str], bool], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - Tuple[List[str], bool]: Advanced usage for grid slicing
            - False: No index storage (default)

        Returns
        -------
        uxarray.UxDataArray
            A subset of the original data array containing only the faces that intersect
            with the specified longitude.

        Raises
        ------
        ValueError
            If no intersections are found at the specified longitude or the data variable is not face-centered.

        Examples
        --------
        >>> # Extract horizontal slice at 0° longitude
        >>> cross_section = uxda.cross_section.constant_longitude(lon=0.0)

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        if not self.uxda._face_centered():
            raise ValueError(
                "Cross sections are only supported for face-centered data variables."
            )

        faces = self.uxda.uxgrid.get_faces_at_constant_longitude(lon)

        return self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

    def constant_latitude_interval(
        self,
        lats: Tuple[float, float],
        inverse_indices: Union[
            List[str], Set[str], Tuple[List[str], bool], bool
        ] = False,
    ):
        """Extracts a horizontal cross-section of data by selecting all faces that
        are within a specified latitude interval.

        Parameters
        ----------
        lats : Tuple[float, float]
            The latitude interval (min_lat, max_lat) at which to extract the cross-section,
            in degrees. Values must be between -90.0 and 90.0
        inverse_indices : Union[List[str], Set[str], Tuple[List[str], bool], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - Tuple[List[str], bool]: Advanced usage for grid slicing
            - False: No index storage (default)

        Returns
        -------
        uxarray.UxDataArray
            A subset of the original data array containing only the faces that are within a specified latitude interval.

        Raises
        ------
        ValueError
            If no faces are found within the specified latitude interval.

        Examples
        --------
        >>> # Extract horizontal slice between 30°S and 30°N latitude
        >>> cross_section = uxda.cross_section.constant_latitude_interval(
        ...     lats=(-30.0, 30.0)
        ... )

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        faces = self.uxda.uxgrid.get_faces_between_latitudes(lats)

        return self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

    def constant_longitude_interval(
        self,
        lons: Tuple[float, float],
        inverse_indices: Union[
            List[str], Set[str], Tuple[List[str], bool], bool
        ] = False,
    ):
        """Extracts a horizontal cross-section of data by selecting all faces are within a specifed longitude interval.

        Parameters
        ----------
        lons : Tuple[float, float]
            The longitude interval (min_lon, max_lon) at which to extract the cross-section,
            in degrees. Values must be between -180.0 and 180.0
        inverse_indices : Union[List[str], Set[str], Tuple[List[str], bool], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - Tuple[List[str], bool]: Advanced usage for grid slicing
            - False: No index storage (default)

        Returns
        -------
        uxarray.UxDataArray
            A subset of the original data array containing only the faces that intersect
            with the specified longitude interval.

        Raises
        ------
        ValueError
            If no faces are found within the specified longitude interval.

        Examples
        --------
        >>> # Extract horizontal slice between 0° and 45° longitude
        >>> cross_section = uxda.cross_section.constant_longitude_interval(
        ...     lons=(0.0, 45.0)
        ... )

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        faces = self.uxda.uxgrid.get_faces_between_longitudes(lons)

        return self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

    def vertical_constant_latitude(
        self,
        lat: float,
        vertical_coord: str,
        inverse_indices: Union[
            List[str], Set[str], Tuple[List[str], bool], bool
        ] = False,
    ):
        """Extracts a vertical cross-section (zonal transect) of the data array by selecting
        all faces that intersect with a specified line of constant latitude across all vertical levels.

        This creates a vertical slice showing how the data varies with longitude and depth/height
        at a fixed latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the vertical cross-section, in degrees.
            Must be between -90.0 and 90.0
        vertical_coord : str
            Name of the vertical coordinate dimension (e.g., 'depth', 'level', 'pressure').
            Must be explicitly specified.
        inverse_indices : Union[List[str], Set[str], Tuple[List[str], bool], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - Tuple[List[str], bool]: Advanced usage for grid slicing
            - False: No index storage (default)

        Returns
        -------
        uxarray.UxDataArray
            A vertical cross-section containing data along the specified latitude
            with dimensions (n_face_subset, vertical_coord) where n_face_subset
            are the faces intersecting the latitude line.

        Raises
        ------
        ValueError
            If no intersections are found at the specified latitude, the data variable
            is not face-centered, or no vertical dimension is found.

        Examples
        --------
        >>> # Extract vertical slice along 30°N latitude (zonal transect)
        >>> vertical_section = uxda.cross_section.vertical_constant_latitude(
        ...     lat=30.0, vertical_coord="depth"
        ... )

        >>> # Another example with a different vertical coordinate
        >>> vertical_section = uxda.cross_section.vertical_constant_latitude(
        ...     lat=30.0, vertical_coord="depth"
        ... )

        Notes
        -----
        - This method preserves all vertical levels, creating a 2D vertical transect
        - The resulting data can be plotted as longitude vs depth/height
        - The initial execution time may be longer due to Numba compilation
        """
        if not self.uxda._face_centered():
            raise ValueError(
                "Cross sections are only supported for face-centered data variables."
            )

        # Validate vertical coordinate
        self._get_vertical_coord_name(vertical_coord)

        # Get faces intersecting the latitude line
        faces = self.uxda.uxgrid.get_faces_at_constant_latitude(lat)

        if len(faces) == 0:
            raise ValueError(f"No faces found intersecting latitude {lat}°")

        # Select data along the latitude line, preserving all vertical levels
        vertical_section = self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

        return vertical_section

    def vertical_constant_longitude(
        self,
        lon: float,
        vertical_coord: str,
        inverse_indices: Union[
            List[str], Set[str], Tuple[List[str], bool], bool
        ] = False,
    ):
        """Extracts a vertical cross-section (meridional transect) of the data array by selecting
        all faces that intersect with a specified line of constant longitude across all vertical levels.

        This creates a vertical slice showing how the data varies with latitude and depth/height
        at a fixed longitude.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the vertical cross-section, in degrees.
            Must be between -180.0 and 180.0
        vertical_coord : str
            Name of the vertical coordinate dimension (e.g., 'depth', 'level', 'pressure').
            Must be explicitly specified.
        inverse_indices : Union[List[str], Set[str], Tuple[List[str], bool], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - Tuple[List[str], bool]: Advanced usage for grid slicing
            - False: No index storage (default)

        Returns
        -------
        uxarray.UxDataArray
            A vertical cross-section containing data along the specified longitude
            with dimensions (n_face_subset, vertical_coord) where n_face_subset
            are the faces intersecting the longitude line.

        Raises
        ------
        ValueError
            If no intersections are found at the specified longitude, the data variable
            is not face-centered, or no vertical dimension is found.

        Examples
        --------
        >>> # Extract vertical slice along 0° longitude (meridional transect)
        >>> vertical_section = uxda.cross_section.vertical_constant_longitude(
        ...     lon=0.0, vertical_coord="pressure"
        ... )

        >>> # Another example with a different vertical coordinate
        >>> vertical_section = uxda.cross_section.vertical_constant_longitude(
        ...     lon=0.0, vertical_coord="pressure"
        ... )

        Notes
        -----
        - This method preserves all vertical levels, creating a 2D vertical transect
        - The resulting data can be plotted as latitude vs depth/height
        - The initial execution time may be longer due to Numba compilation
        """
        if not self.uxda._face_centered():
            raise ValueError(
                "Cross sections are only supported for face-centered data variables."
            )

        # Validate vertical coordinate
        self._get_vertical_coord_name(vertical_coord)

        # Get faces intersecting the longitude line
        faces = self.uxda.uxgrid.get_faces_at_constant_longitude(lon)

        if len(faces) == 0:
            raise ValueError(f"No faces found intersecting longitude {lon}°")

        # Select data along the longitude line, preserving all vertical levels
        vertical_section = self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

        return vertical_section
