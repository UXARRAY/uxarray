from __future__ import annotations


from typing import TYPE_CHECKING, Union, List, Set, Tuple

if TYPE_CHECKING:
    pass


class UxDataArrayCrossSectionAccessor:
    """Accessor for cross-section operations on a ``UxDataArray``"""

    def __init__(self, uxda) -> None:
        self.uxda = uxda

    def __repr__(self):
        prefix = "<uxarray.UxDataArray.cross_section>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += "  * constant_latitude(lat, inverse_indices)\n"
        methods_heading += "  * constant_longitude(lon, inverse_indices)\n"
        methods_heading += "  * constant_latitude_interval(lats, inverse_indices)\n"
        methods_heading += "  * constant_longitude_interval(lons, inverse_indices)\n"

        return prefix + methods_heading

    def constant_latitude(
        self, lat: float, inverse_indices: Union[List[str], Set[str], bool] = False
    ):
        """Extracts a cross-section of the data array by selecting all faces that
        intersect with a specified line of constant latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the cross-section, in degrees.
            Must be between -90.0 and 90.0
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
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
        >>> # Extract data at 15.5°S latitude
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
        self, lon: float, inverse_indices: Union[List[str], Set[str], bool] = False
    ):
        """Extracts a cross-section of the data array by selecting all faces that
        intersect with a specified line of constant longitude.

        Parameters
        ----------
        lon : float
            The latitude at which to extract the cross-section, in degrees.
            Must be between -180.0 and 180.0
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
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
        >>> # Extract data at 0° longitude
        >>> cross_section = uxda.cross_section.constant_latitude(lon=0.0)

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        if not self.uxda._face_centered():
            raise ValueError(
                "Cross sections are only supported for face-centered data variables."
            )

        faces = self.uxda.uxgrid.get_faces_at_constant_longitude(
            lon,
        )

        return self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

    def constant_latitude_interval(
        self,
        lats: Tuple[float, float],
        inverse_indices: Union[List[str], Set[str], bool] = False,
    ):
        """Extracts a cross-section of data by selecting all faces that
        are within a specified latitude interval.

        Parameters
        ----------
        lats : Tuple[float, float]
            The latitude interval (min_lat, max_lat) at which to extract the cross-section,
            in degrees. Values must be between -90.0 and 90.0
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
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
        >>> # Extract data between 30°S and 30°N latitude
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
        inverse_indices: Union[List[str], Set[str], bool] = False,
    ):
        """Extracts a cross-section of data by selecting all faces are within a specifed longitude interval.

        Parameters
        ----------
        lons : Tuple[float, float]
            The longitude interval (min_lon, max_lon) at which to extract the cross-section,
            in degrees. Values must be between -180.0 and 180.0
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
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
        >>> # Extract data between 0° and 45° longitude
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
